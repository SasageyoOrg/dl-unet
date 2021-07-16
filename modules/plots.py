import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import lib
import cv2

def plot_train(history, epochs):
    fig = plt.figure(figsize=(10, 10))
    plt.title('Training and validation loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.plot(np.arange(0, epochs), history.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), history.history["val_loss"], label="val_loss")

    plt.title('Training and validation accuracy')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.plot(np.arange(0, epochs), history.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), history.history["val_accuracy"], label="val_acc")
    return fig

def plot_train_metric(history, epochs, metric):
    '''
    fig, ax = plt.subplots(1, 2, figsize=(32, 16))
    
    ax[0].set_title('Training and Validation Loss')
    ax[0].set(xlabel='epochs', ylabel='loss')
    ax[0].plot(np.arange(0, epochs), history.history["loss"], label="train_loss")
    ax[0].plot(np.arange(0, epochs), history.history["val_loss"], label="val_loss")
    ax[0].legend()

    ax[1].set_title('Training and Validation Metrics')
    ax[1].set(xlabel='epochs', ylabel='loss')
    ax[1].plot(np.arange(0, epochs), history.history[metric], label = "train_" + metric)
    ax[1].plot(np.arange(0, epochs), history.history["val_" + metric], label = "val_" + metric)
    ax[1].legend()
    '''
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    
    ax[0].set_title('Training and Validation Loss', size=18)
    ax[0].set_xlabel('epochs',fontsize = 18)
    ax[0].set_ylabel('loss',fontsize = 18)
    ax[0].plot(np.arange(0, epochs), history["loss"], label="train_loss")
    ax[0].plot(np.arange(0, epochs), history["val_loss"], label="val_loss")
    ax[0].legend(prop={'size': 18})

    ax[1].set_title('Training and Validation Metrics', size=18)
    ax[1].set_xlabel('epochs',fontsize = 18)
    ax[1].set_ylabel('loss',fontsize = 18)
    ax[1].plot(np.arange(0, epochs), history[metric], label = "train_" + metric)
    ax[1].plot(np.arange(0, epochs), history["val_" + metric], label = "val_" + metric)
    ax[1].legend(prop={'size': 18})
    return fig

def plot_augmented_batch(train_generator):
    # Print some examples of random data agumentation
    batchx, batchy = next(train_generator)

    for i in range(batchx.shape[0]):
        fig, ax = plt.subplots(1, 3, figsize=(8,16))
        ax[0].set_title('image')
        ax[0].imshow(batchx[i][:,:,0], cmap='gray')
        ax[1].set_title('label')
        ax[1].imshow(batchy[i][:,:,0], cmap='gray')
        ax[2].set_title('image+label')
        ax[2].imshow(batchx[i][:,:,0], cmap='gray')
        ax[2].imshow(batchy[i][:,:,0], 'jet', interpolation='none', alpha=0.5)
    return fig

def plot_test_results(test_results):
    fig = plt.figure(figsize=(15, 15))

    fig.add_subplot(2,2,1)
    plt.title("Test Image")
    plt.imshow(test_results['test_image'][0,:,:,0], cmap='gray')

    fig.add_subplot(2,2,2)
    plt.title("Test Mask")
    plt.imshow(test_results['test_mask'], cmap='gray')


    fig.add_subplot(2,2,3)
    plt.title("Predicted Image")
    plt.imshow(test_results['pred_image'][0,:,:,0], cmap='gray')

    fig.add_subplot(2,2,4)
    plt.title("Predicted Mask")
    plt.imshow(test_results['pred_mask'], cmap='gray')

    return fig



def get_results(dataset_list, params, model, threshold):

    results = {}    # all metrics per image
    length_test = len(dataset_list[0]['test'])

    for image_index in range (length_test):
        test_image = cv2.resize(dataset_list[0]['test'][image_index,:,:,0],(params["x"],params["y"]))
        test_mask = cv2.resize(dataset_list[1]['test'][image_index,:,:,0],(params["x"],params["y"]))
        test_results = lib.get_preditcions(test_image, test_mask, params, model, threshold)
        results[image_index] = {'dice': lib.dice(test_mask, test_results["pred_mask"])}
    
    results_df = pd.DataFrame(results).transpose()  
    avgs = {'dice': results_df['dice'].mean()}      # metrics averages
    return results_df, avgs

def plot_bad_predictions(bad_res, dataset_list, params, model):
  threshold = 0.5
  for idx in bad_res.index:
    print('\n')
    print('\n')
    print(f"Image index {idx} - dice: {bad_res.loc[idx]}")
    threshold = 0.5
    test_image = dataset_list[0]['test'][idx,:,:,0]
    test_mask = dataset_list[1]['test'][idx,:,:,0]
    test_results = lib.get_preditcions(test_image, test_mask, params, model, threshold)
    results_figure = plot_test_results(test_results)

    plt.show()



