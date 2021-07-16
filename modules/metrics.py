
import numpy
import keras
import keras.backend as K
import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy

# Dice Coefficient
def dice(output, target, smooth = 1e-7):
    output = tf.keras.layers.Flatten()(output)
    #output[output < 0] = 0
    target = tf.keras.layers.Flatten()(target)
    intersection = tf.reduce_sum(output * target)

    return (2. * intersection + smooth) / (tf.reduce_sum(output) + tf.reduce_sum(target) + smooth)

# IoU Coefficient
# source: https://tensorlayer.readthedocs.io/en/latest/_modules/tensorlayer/cost.html#dice_coe
def iou(output, target, threshold=0.5, axis=(1, 2, 3), smooth=1e-5):
    """Non-differentiable Intersection over Union (IoU) for comparing the
    similarity of two batch of data, usually be used for evaluating binary image segmentation.
    The coefficient between 0 to 1, and 1 means totally match.

    Notes
    ------
    - IoU cannot be used as training loss, people usually use dice coefficient for training, 
      IoU and hard-dice for evaluating.
    """
    pre = tf.cast(output > threshold, dtype=tf.float32)
    truth = tf.cast(target > threshold, dtype=tf.float32)
    inse = tf.reduce_sum(tf.multiply(pre, truth), axis=axis)  # AND
    union = tf.reduce_sum(tf.cast(tf.add(pre, truth) >= 1, dtype=tf.float32), axis=axis)  # OR
    
    batch_iou = (inse + smooth) / (union + smooth)
    iou = tf.reduce_mean(batch_iou, name='iou')
    return iou  # , pre, truth, inse, union

# Dice Loss
def dice_loss(output, target):
  return 1 - dice(output, target)

# BCE-Dice Loss
# source: https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch#BCE-Dice-Loss
def bce_dice_loss(output, target):
    return binary_crossentropy(output, target) + dice_loss(output, target)