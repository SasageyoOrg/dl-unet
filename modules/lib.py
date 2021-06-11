
# TODO: brief description of all methods
# ---------------------------------------------------------------------------- #
#                                 "lib" module
# contains all necessary tools:
#     - load_data
#     - split_data
#     - data_augmentation
#     - train
#     - cleanup
#     - get_predictions
#     - dice
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
#                                    Imports                                   #
# ---------------------------------------------------------------------------- #

import pickle
import numpy as np
import os
import cv2
import copy
import glob
import json

from sklearn.metrics import accuracy_score
from tensorflow.keras.optimizers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger
from tensorflow.keras import backend as k
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras import backend as K

from itertools import groupby
from itertools import chain



# ---------------------------------------------------------------------------- #
#                                    Methods                                   #
# ---------------------------------------------------------------------------- #

# ------------------------------------ INIT ---------------------------------- #
def get_dataset_list(mode, path_, imgs_list_path, params, ratio_test, ratio_val, seed, modality):
  if mode == 0:
    # RANDOM SPLIT
    x = split_patients(imgs_list_path, ratio_test, ratio_val, modality)
    X, Y = load_data_patients(path_, x, params, modality)
  else:
    # SPLIT PATIENTS
    imgs, msks = load_data_random(path_, params)
    X, Y = split_data_random(imgs, msks, ratio_test, ratio_val, seed)

  dataset_list = [X, Y]
  return dataset_list

# --------------------------------- load_data -------------------------------- #
def load_data_random(path_, params):
  dim = (params['x'],params['y'])

  N = params['length_data'][0]
  X = np.empty((N, params['y'],params['x'], params['n_channels']))
  Y = np.empty((N, params['y'],params['x'], params['n_channels_mask']))
  imgs_list_path = os.listdir(path_[0])[:N]
  
  i = 0
  for im in sorted(imgs_list_path):
    image = cv2.imread(os.path.join(path_[0],im),0)
    image = cv2.resize(image,dim)

    mea = np.mean(image)
    ss = np.std(image)
    image = (image - mea)/ss

    mask = cv2.imread(os.path.join(path_[1],im),0)
    mask = cv2.resize(mask,dim)
    mask = mask / 255

    X[i,:,:,0] = image
    Y[i,:,:,0] = mask
    i += 1
  return X, Y

def load_data_patients(path_, splitting, params, modality): 
  dim = (params['x'],params['y'])
  N = params['length_data']

  Xlist = {}
  Ylist = {}
  for mode in modality:
    lenght_mode = len(splitting[mode])
    imgs = np.empty((lenght_mode, params['y'],params['x'], params['n_channels']))
    msks = np.empty((lenght_mode, params['y'],params['x'], params['n_channels_mask']))
    i = 0
    for im in sorted(splitting[mode]):
      # Get image------------------------------------
      image = cv2.imread(os.path.join(path_[0],im),0)
      image = cv2.resize(image,dim)
      # Normalize image
      mea = np.mean(image)
      ss  = np.std(image)
      image = (image - mea)/ss
      # ---------------------------------------------

      # Get mask-------------------------------------
      mask = cv2.imread(os.path.join(path_[1],im),0)
      mask = cv2.resize(mask,dim)
      # Normalize mask
      mask = mask / 255
      # ---------------------------------------------

      imgs[i,:,:,0] = image
      msks[i,:,:,0] = mask
      i += 1 
      
    Xlist[mode] = imgs
    Ylist[mode] = msks 
    
    return Xlist, Ylist

# -------------------------------- split_data -------------------------------- #

def split_data_random(imgs, msks, ratio_test, ratio_val, seed):

  X = {}
  Y = {}

  # Split in train / test
  X_train, X['test'], Y['train'], Y['test'] = train_test_split(
                                                imgs,
                                                msks,
                                                test_size = (1 - ratio_test),
                                                random_state = seed)

  # Split in train / val
  X['train'], X['validation'], Y['train'], Y['validation'] = train_test_split(
                                                              X_train,
                                                              Y['train'],
                                                              test_size = (1 - ratio_val),
                                                              random_state = seed)

  return X, Y

def split_patients(path, ratio_test, ratio_val, modality):
  # clean patient name file: "Pz001-001.bmp" -> "Pz001"
  patient = [x[:-8] for x in path]
  patient.sort()

  '''
  # creating dictionary with patient image number
  count_distinct_patient = {}
  for key, value in groupby(patient):
      count = len(list(value))
      count_distinct_patient[key] = count
  '''

  patient = list(dict.fromkeys(patient)) # delete duplicate name patient

  x = {}
  # Splitting
  x_train, x['test'] = train_test_split(patient, 
                                        test_size = 0.20, 
                                        random_state = 42)
  
  x['train'], x['validation'] = train_test_split(x_train, 
                                                 test_size = 0.10, 
                                                 random_state = 42)
  
  x_path = {}
  # Get each path of the image, for each mode
  for mode in modality:
    path_ = []
    for elm in x[mode]:
      path_.append(subfinder(path, elm)) # get all image name 
    x_path[mode] = list(chain.from_iterable(path_)) # merging sublist

  return x_path

# find sublist
def subfinder(mylist, pattern):
  return [x for x in mylist if x[:-8] in pattern]

# ----------------------------- data_augmentation ---------------------------- #

def data_augmentation(seed, dataset_list, params):
# ---------------------------------------------------------------------------- #
  # * Train
  train_img_datagen  = ImageDataGenerator(rotation_range = 25, fill_mode='constant')
  train_msk_datagen  = ImageDataGenerator(rotation_range = 25, fill_mode='constant')

  train_img_datagen.fit(dataset_list[0]['train'], augment=True, seed=seed)
  train_msk_datagen.fit(dataset_list[1]['train'], augment=True, seed=seed)

  train_img_generator = train_img_datagen.flow(dataset_list[0]['train'], batch_size = params['batch_size'], shuffle = True, seed=seed)
  train_msk_generator = train_msk_datagen.flow(dataset_list[1]['train'], batch_size = params['batch_size'], shuffle = True, seed=seed)

  train_generator = zip(train_img_generator, train_msk_generator)
# ---------------------------------------------------------------------------- #
  # * Validation
  vali_img_datagen = ImageDataGenerator()
  vali_msk_datagen = ImageDataGenerator()

  vali_img_datagen.fit(dataset_list[0]['validation'], seed=seed)
  vali_msk_datagen.fit(dataset_list[1]['validation'], seed=seed)

  val_img_generator = vali_img_datagen.flow(dataset_list[0]['validation'], batch_size = params['batch_size'], seed=seed)
  val_msk_generator = vali_msk_datagen.flow(dataset_list[1]['validation'], batch_size = params['batch_size'], seed=seed)

  val_generator = zip(val_img_generator, val_msk_generator)

  return train_generator, val_generator

# ----------------------------------- train ---------------------------------- #
# TODO: manage history and model_json_tuned     [V]
# TODO: auto select best model                  [V]

def train(root, params, model, train_generator, val_generator):
  model_weights_path = root
  # model_weights_path = root + 'unet/Model{epoch:03d}-{loss:2f}.hdf5'

  checkPoint = ModelCheckpoint(model_weights_path,
                              monitor = 'val_loss',
                              verbose = 1,
                              save_best_only = True,
                              mode = 'min')
  callbacks_list = checkPoint

  # for fine tuning purposes
  #model.load_weights(root + '/results/Unetprova/Model008-0.161614.hdf5')
  #model.load_weights("/content/gdrive/My Drive/CVDL/Segmentation/weights2.h5")

  model.compile(optimizer = Adam(learning_rate = params["learningRate"]),
                                  loss = "binary_crossentropy",
                                  metrics = ['accuracy'])

  steps_per_epoch = np.ceil(params["length_training"]/params["batch_size"])

  # csv_logger = CSVLogger(root + '/unet.csv', append = False, separator = ';')

  history = model.fit(train_generator,
                      steps_per_epoch = steps_per_epoch,
                      epochs = params["nEpoches"],
                      validation_data = val_generator,
                      verbose = 1,
                      validation_steps = (params['length_validation']/params["batch_size"]),
                      callbacks = [callbacks_list])

  #this saves the weights of the last epoch, only!
  # model.save_weights("./weights.h5")
  print("Saved model and weights in the directory")
  model.load_weights(model_weights_path)

  print("Loaded best weights: ", model_weights_path)

  return history, model

# ---------------------------------- cleanup --------------------------------- #

def cleanup(root, best_weights):
  best_weights_path = root + '/unet'

  if os.path.isfile(best_weights_path):
      #Verifies CSV file was created, then deletes unneeded files.
      for CleanUp in glob.glob(root+'/*.*'):
          print(CleanUp)

          if not CleanUp.endswith(best_weights):
              os.remove(CleanUp)

# ------------------------------ get_preditcions ----------------------------- #

def get_preditcions(test_image, test_mask, params, model, threshold):

  mean_ = np.mean(test_image)
  test_image = test_image - mean_
  std = np.std(test_image)
  test_image = test_image/std

  test_mask = test_mask

  test_image = img_to_array(test_image)
  test_image = test_image.reshape((1, test_image.shape[0],
                                      test_image.shape[1],
                                      test_image.shape[2]))
  #model.predict_classes(X_test)

  # prediction
  prediction = model.predict(test_image)
  # copia per ridefinire la segmentazione ar di intensità dei valori
  pred = copy.copy(prediction[0,:,:,0])

  pred[pred >= threshold] = 1
  pred[pred < threshold] = 0

  test_results = {
    'test_image': test_image,
    'test_mask': test_mask,
    'pred_image': prediction,
    'pred_mask': pred
    }
  return test_results

# --------------------------------- acccuracy -------------------------------- #

# @ test_mask: the target mask to make the test on
# @ pred_mask: mask predicted by the model

def accuracy(test_mask, pred_mask):

    test_mask = test_mask/255
    a = test_mask.flatten()
    b = pred_mask.flatten()
    a = a.astype(int)
    b = b.astype(int)
    return accuracy_score(a, b)


# ----------------------------------- dice ----------------------------------- #

def dice(im1, im2, empty_score=1.0):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        Both are empty (sum eq to zero) = empty_score
    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    # Test usando Keras.sum() ...
    #dice_numerator = K.sum(2 * im1 * im2, axis=(0, 1, 2)) + empty_score
    #dice_denominator = K.sum(im1, axis= axis) + K.sum(im2, axis =(0, 1, 2)) + empty_score
    #dice_coefficient = dice_numerator / dice_denominator

    return 2. * intersection.sum() / im_sum

# ----------------------------------- dump ----------------------------------- #

def create_dump(data, dump_name):
  pik = dump_name
  with open(pik, "wb") as f:
    pickle.dump(data, f)

# --------------------------------- load_dump -------------------------------- #
def load_dump(dump_name):
  pik = dump_name
  with open(pik, "rb") as f:
    return pickle.load(data, f)
