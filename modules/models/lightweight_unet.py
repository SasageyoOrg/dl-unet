from tensorflow.keras.layers import *
from tensorflow.keras.models import *

def build_model(initializer, input_size):
  inputs = Input(input_size) 

  #-------------------------------ENCODING-------------------------------------
  conv1 = Activation('relu')(BatchNormalization()(Conv2D(32, (3, 3), padding='same', kernel_initializer = initializer)(inputs)))
  #conv1 = Activation('relu')(BatchNormalization()(Conv2D(32, (3, 3), padding='same', kernel_initializer = initializer)(conv1)))
  conv1 = Activation('relu')(Conv2D(32, (3, 3), padding='same', kernel_initializer = initializer)(conv1))
  pool1 = MaxPool2D((2, 2))(conv1)

  conv2 = Activation('relu')(BatchNormalization()(Conv2D(64, (3, 3), padding='same', kernel_initializer = initializer)(pool1)))
  #conv2 = Activation('relu')(BatchNormalization()(Conv2D(64, (3, 3), padding='same', kernel_initializer = initializer)(conv2)))
  conv2 = Activation('relu')(Conv2D(64, (3, 3), padding='same', kernel_initializer = initializer)(conv2))
  pool2 = MaxPool2D((2, 2))(conv2)

  conv3 = Activation('relu')(BatchNormalization()(Conv2D(128, (3, 3), padding='same', kernel_initializer = initializer)(pool2)))
  #conv3 = Activation('relu')(BatchNormalization()(Conv2D(128, (3, 3), padding='same', kernel_initializer = initializer)(conv3)))
  conv3 = Activation('relu')(Conv2D(128, (3, 3), padding='same', kernel_initializer = initializer)(conv3))
  pool3 = MaxPool2D((2, 2))(conv3)
  #---------------------------------------------------------------------------

  #---------------------------BOTTLENECK--------------------------------------
  conv4 = Activation('relu')(BatchNormalization()(Conv2D(256, (3, 3), padding='same', kernel_initializer = initializer)(pool3)))
  #conv4 = Activation('relu')(BatchNormalization()(Conv2D(256, (3, 3), padding='same', kernel_initializer = initializer)(conv4)))
  conv4 = Activation('relu')(Conv2D(256, (3, 3), padding='same', kernel_initializer = initializer)(conv4))
  drop4 = Dropout(0.2)(conv4)
  #----------------------------------------------------------------------------

  #-------------------------------DECODING-------------------------------------
  up1 = UpSampling2D((2, 2), interpolation='bilinear')(drop4)
  merge1 = Concatenate()([up1, conv3])
  dconv1 = Activation('relu')(BatchNormalization()(Conv2D(128, (3, 3), padding='same', kernel_initializer = initializer)(merge1)))
  #dconv1 = Activation('relu')(BatchNormalization()(Conv2D(128, (3, 3), padding='same', kernel_initializer = initializer)(dconv1)))
  dconv1 = Activation('relu')(Conv2D(128, (3, 3), padding='same', kernel_initializer = initializer)(dconv1))
  
  up2 = UpSampling2D((2, 2), interpolation='bilinear')(dconv1)
  merge2 = Concatenate()([up2, conv2])
  dconv2 = Activation('relu')(BatchNormalization()(Conv2D(64, (3, 3), padding='same', kernel_initializer = initializer)(merge2)))
  #dconv2 = Activation('relu')(BatchNormalization()(Conv2D(64, (3, 3), padding='same', kernel_initializer = initializer)(dconv2)))
  dconv2 = Activation('relu')(Conv2D(64, (3, 3), padding='same', kernel_initializer = initializer)(dconv2))

  up3 = UpSampling2D((2, 2), interpolation='bilinear')(dconv2)
  merge3 = Concatenate()([up3, conv1])
  dconv3 = Activation('relu')(BatchNormalization()(Conv2D(32, (3, 3), padding='same', kernel_initializer = initializer)(merge3)))
  #dconv3 = Activation('relu')(BatchNormalization()(Conv2D(32, (3, 3), padding='same', kernel_initializer = initializer)(dconv3)))
  dconv3 = Activation('relu')(Conv2D(32, (3, 3), padding='same', kernel_initializer = initializer)(dconv3))
  #----------------------------------------------------------------------------

  #-------------------------------OUTPUT---------------------------------------
  dconv3   = Activation('relu')(Conv2D(2, (3, 3), padding = 'same', kernel_initializer = initializer)(dconv3))
  out_conv =  Activation('sigmoid')(Conv2D(1, (1, 1), padding='same')(dconv3))
  #----------------------------------------------------------------------------    

  model = Model(inputs, out_conv)
  return model