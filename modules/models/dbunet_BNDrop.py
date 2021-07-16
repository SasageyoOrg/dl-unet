import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.applications import *
from keras.initializers import glorot_uniform

# Squeeze and Excitation Block 
# source: https://paperswithcode.com/method/squeeze-and-excitation-block
def SqueezeExciteBlock(inputs, ratio=8):
    x = inputs           
    filters = x.shape[-1]
    
    se = GlobalAveragePooling2D()(x)
    se = Reshape((1, 1, filters))(se)
    se = Dense(filters // ratio, activation='relu', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', use_bias=False)(se)

    y = Multiply()([x, se])

    return y

# Atrous Spatial Pyramid Pooling (ASPP)
# source: https://paperswithcode.com/method/aspp
def ASPP(x, filter):
    shape = x.shape

    y1 = AveragePooling2D(pool_size=(shape[1], shape[2]))(x)
    y1 = Activation('relu')(BatchNormalization()(Conv2D(filter, 1, padding='same')(y1)))
    y1 = UpSampling2D((shape[1], shape[2]), interpolation='bilinear')(y1)

    y2 = Activation('relu')(BatchNormalization()(Conv2D(filter, 1, dilation_rate=1, padding='same', use_bias=False)(x)))
    y3 = Activation('relu')(BatchNormalization()(Conv2D(filter, 3, dilation_rate=6, padding='same', use_bias=False)(x)))
    y4 = Activation('relu')(BatchNormalization()(Conv2D(filter, 3, dilation_rate=12, padding='same', use_bias=False)(x)))
    y5 = Activation('relu')(BatchNormalization()(Conv2D(filter, 3, dilation_rate=18, padding='same', use_bias=False)(x)))

    y = Concatenate()([y1, y2, y3, y4, y5])
    y = Activation('relu')(BatchNormalization()(Conv2D(filter, 1, dilation_rate=1, padding='same', use_bias=False)(y)))

    return y
# Unet 1 -----------------------------------------------------------------------
def encoder_1(inputs):
  # skip connections array
  skip_connections = []

  # convert to 3-channel input
  inputs_3ch = Concatenate()([inputs, inputs, inputs])  
  # vgg19
  model = VGG19(include_top=False, weights='imagenet', input_tensor=inputs_3ch)

  # conv blocks
  names = ['block1_conv2', 'block2_conv2', 'block3_conv4']
  for name in names:
      skip_connections.append(model.get_layer(name).output)

  # output
  output = model.get_layer('block4_conv4').output

  return output, skip_connections


def decoder_1(inputs, skip_connections):
  # reverse the skip connections order
  #skip_connections.reverse()

  x = inputs

  ''' 
      4 blocks of these layers:
      -> UpSampling2D 
      -> Concatenate 
      -> Conv2D + BatchNormalization
      -> Conv2D + BatchNormalization
      -> SqueezeExciteBlock
  '''
  # conv_block_1 ->
  up1 = UpSampling2D((2, 2), interpolation='bilinear')(x)
  merge1 = Concatenate()([up1, skip_connections[2]])
  conv1 = Activation('relu')(BatchNormalization()(Conv2D(128, (3, 3), padding='same')(merge1)))
  conv1 = Activation('relu')(BatchNormalization()(Conv2D(128, (3, 3), padding='same')(conv1)))
  seb1 = SqueezeExciteBlock(conv1)
  # <- conv_block_1

  # conv_block_2 ->
  up2 = UpSampling2D((2, 2), interpolation='bilinear')(seb1)
  merge2 = Concatenate()([up2, skip_connections[1]])
  conv2 = Activation('relu')(BatchNormalization()(Conv2D(64, (3, 3), padding='same')(merge2)))
  conv2 = Activation('relu')(BatchNormalization()(Conv2D(64, (3, 3), padding='same')(conv2)))
  seb2 = SqueezeExciteBlock(conv2)
  # <- conv_block_2

  # conv_block_3 ->
  up3 = UpSampling2D((2, 2), interpolation='bilinear')(seb2)
  merge3 = Concatenate()([up3, skip_connections[0]])
  conv3 = Activation('relu')(BatchNormalization()(Conv2D(32, (3, 3), padding='same')(merge3)))
  conv3 = Activation('relu')(BatchNormalization()(Conv2D(32, (3, 3), padding='same')(conv3)))
  seb3 = SqueezeExciteBlock(conv3)
  # <- conv_block_3

  # output
  # conv5 = Conv2D(2, 3, activation = 'relu', padding = 'same')(seb4)
  # conv6 = Conv2D(1, 1, activation = 'sigmoid')(conv5)

  conv4 = Activation('relu')(Conv2D(2, (3, 3), padding='same')(seb3))
  conv4 = Activation('sigmoid')(Conv2D(1, (1, 1), padding='same')(conv4))

  return conv4

# Unet 2 -----------------------------------------------------------------------
def unet(inputs, initializer, skip_connections):

  x = inputs

  ''' 
      ### U-Net Encoder ###
      composed by 4 blocks of these layers:
      -> Conv2D + BatchNormalization
      -> Conv2D + BatchNormalization
      -> MaxPooling2D

      note: the last block integrates a Dropout layer
  '''
  # encoder_conv_block_1 ->
  econv1 = Activation('relu')(BatchNormalization()(Conv2D(32, (3, 3), padding='same', kernel_initializer = initializer)(x)))
  econv1 = Activation('relu')(BatchNormalization()(Conv2D(32, (3, 3), padding='same', kernel_initializer = initializer)(econv1)))
  eseb1 = SqueezeExciteBlock(econv1)
  pool1 = MaxPool2D((2, 2))(eseb1)
  # <- encoder_conv_block_1

  # encoder_conv_block_2 ->
  econv2 = Activation('relu')(BatchNormalization()(Conv2D(64, (3, 3), padding='same', kernel_initializer = initializer)(pool1)))
  econv2 = Activation('relu')(BatchNormalization()(Conv2D(64, (3, 3), padding='same', kernel_initializer = initializer)(econv2)))
  eseb2 = SqueezeExciteBlock(econv2)
  pool2 = MaxPool2D((2, 2))(eseb2)
  # <- encoder_conv_block_2

  '''
    Added dropout between eseb3 & pool 3
  '''

  # encoder_conv_block_3 ->
  econv3 = Activation('relu')(BatchNormalization()(Conv2D(128, (3, 3), padding='same', kernel_initializer = initializer)(pool2)))
  econv3 = Activation('relu')(BatchNormalization()(Conv2D(128, (3, 3), padding='same', kernel_initializer = initializer)(econv3)))
  drop1 = Dropout(0.2)(econv3)
  eseb3 = SqueezeExciteBlock(drop1)
  pool3 = MaxPool2D((2, 2))(eseb3)
  aspp1 = ASPP(pool3, 64)
  # <- encoder_conv_block_3

 

  '''
    Added bottleneck
  '''
  # center_block (bottleneck) ->
  conv4 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(aspp1)
  conv4 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv4)
  drop2 = Dropout(0.2)(conv4)
  # <- center_block (bottleneck)

  ''' 
      ### U-Net Decoder ###
      composed by 4 blocks of these layers:
      -> UpSampling2D
      -> Concatenate
      -> Conv2D + BatchNormalization
      -> Conv2D + BatchNormalization
  '''
  # decoder_conv_block_1 ->
  up1 = UpSampling2D((2, 2), interpolation='bilinear')(drop2)
  merge1 = Concatenate()([up1, skip_connections[2], eseb3])
  dconv1 = Activation('relu')(BatchNormalization()(Conv2D(128, (3, 3), padding='same', kernel_initializer = initializer)(merge1)))
  dconv1 = Activation('relu')(BatchNormalization()(Conv2D(128, (3, 3), padding='same', kernel_initializer = initializer)(dconv1)))
  dseb1 = SqueezeExciteBlock(dconv1)
  # <- decoder_conv_block_1

  # decoder_conv_block_2 ->
  up2 = UpSampling2D((2, 2), interpolation='bilinear')(dseb1)
  merge2 = Concatenate()([up2, skip_connections[1], eseb2])
  dconv2 = Activation('relu')(BatchNormalization()(Conv2D(64, (3, 3), padding='same', kernel_initializer = initializer)(merge2)))
  dconv2 = Activation('relu')(BatchNormalization()(Conv2D(64, (3, 3), padding='same', kernel_initializer = initializer)(dconv2)))
  dseb2 = SqueezeExciteBlock(dconv2)
  # <- decoder_conv_block_2

  # decoder_conv_block_2 ->
  up3 = UpSampling2D((2, 2), interpolation='bilinear')(dseb2)
  merge3 = Concatenate()([up3, skip_connections[0], eseb1])
  dconv3 = Activation('relu')(BatchNormalization()(Conv2D(32, (3, 3), padding='same', kernel_initializer = initializer)(merge3)))
  dconv3 = Activation('relu')(BatchNormalization()(Conv2D(32, (3, 3), padding='same', kernel_initializer = initializer)(dconv3)))
  dseb3 = SqueezeExciteBlock(dconv3)
  # <- decoder_conv_block_2

  # output
  #conv6 = Conv2D(2, 3, activation = 'relu', padding = 'same')(dseb4)
  #conv7 = Conv2D(1, 1, activation = 'sigmoid')(conv6)

  dconv4= Activation('relu')(Conv2D(2, (3, 3), padding='same', kernel_initializer = initializer)(dseb3))
  dconv5 = Activation('sigmoid')(Conv2D(1, (1, 1), padding="same")(dconv4))

  return dconv5


def build_model(initializer, input_size):
  inputs = Input(input_size)

  # encoder 1 (vgg19) 
  x, skips = encoder_1(inputs)
  x = ASPP(x, 64)
  # decoder 1
  output_1 = decoder_1(x, skips)
  
  # multiply input and output 1
  x = inputs * output_1
  
  # unet
  output_2 = unet(x, initializer, skips)
  
  # concatenate output 1 and output 2
  #outputs = Concatenate()([output_1, output_2])

  # model
  model = Model(inputs, output_2)

  return model