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

def encoder_1(inputs):
  # skip connections array
  skip_connections = []

  # convert to 3-channel input
  inputs_3ch = Concatenate()([inputs, inputs, inputs])  
  # VGG19
  model = VGG19(include_top=False, weights='imagenet', input_tensor=inputs_3ch)
  # ResNet50V2
  #model = ResNet50V2(include_top=False, weights='imagenet', input_tensor=inputs_3ch)

  # conv blocks
  names = ['block1_conv2', 'block2_conv2', 'block3_conv4', 'block4_conv4']
  for name in names:
      skip_connections.append(model.get_layer(name).output)

  # output
  output = model.get_layer('block5_conv4').output

  return output, skip_connections


def decoder_1(inputs, skip_connections):
  # reverse the skip connections order
  skip_connections.reverse()

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
  merge1 = Concatenate()([up1, skip_connections[0]])
  conv1 = Activation('relu')(BatchNormalization()(Conv2D(256, (3, 3), padding='same')(merge1)))
  conv1 = Activation('relu')(BatchNormalization()(Conv2D(256, (3, 3), padding='same')(conv1)))
  seb1 = SqueezeExciteBlock(conv1)
  # <- conv_block_1

  # conv_block_2 ->
  up2 = UpSampling2D((2, 2), interpolation='bilinear')(seb1)
  merge2 = Concatenate()([up2, skip_connections[1]])
  conv2 = Activation('relu')(BatchNormalization()(Conv2D(128, (3, 3), padding='same')(merge2)))
  conv2 = Activation('relu')(BatchNormalization()(Conv2D(128, (3, 3), padding='same')(conv2)))
  seb2 = SqueezeExciteBlock(conv2)
  # <- conv_block_2

  # conv_block_3 ->
  up3 = UpSampling2D((2, 2), interpolation='bilinear')(seb2)
  merge3 = Concatenate()([up3, skip_connections[2]])
  conv3 = Activation('relu')(BatchNormalization()(Conv2D(64, (3, 3), padding='same')(merge3)))
  conv3 = Activation('relu')(BatchNormalization()(Conv2D(64, (3, 3), padding='same')(conv3)))
  seb3 = SqueezeExciteBlock(conv3)
  # <- conv_block_3

  # conv_block_4 ->
  up4 = UpSampling2D((2, 2), interpolation='bilinear')(seb3)
  merge4 = Concatenate()([up4, skip_connections[3]])
  conv4 = Activation('relu')(BatchNormalization()(Conv2D(32, (3, 3), padding='same')(merge4)))
  conv4 = Activation('relu')(BatchNormalization()(Conv2D(32, (3, 3), padding='same')(conv4)))
  seb4 = SqueezeExciteBlock(conv4)
  # <- conv_block_4

  # output
  # conv5 = Conv2D(2, 3, activation = 'relu', padding = 'same')(seb4)
  # conv6 = Conv2D(1, 1, activation = 'sigmoid')(conv5)

  conv6 = Activation('sigmoid')(Conv2D(1, (1, 1), padding="same")(seb4))

  return conv6


def unet(inputs, skip_connections):
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
  econv1 = Activation('relu')(BatchNormalization()(Conv2D(32, (3, 3), padding='same')(x)))
  econv1 = Activation('relu')(BatchNormalization()(Conv2D(32, (3, 3), padding='same')(econv1)))
  eseb1 = SqueezeExciteBlock(econv1)
  pool1 = MaxPool2D((2, 2))(eseb1)
  # <- encoder_conv_block_1

  # encoder_conv_block_2 ->
  econv2 = Activation('relu')(BatchNormalization()(Conv2D(64, (3, 3), padding='same')(pool1)))
  econv2 = Activation('relu')(BatchNormalization()(Conv2D(64, (3, 3), padding='same')(econv2)))
  eseb2 = SqueezeExciteBlock(econv2)
  pool2 = MaxPool2D((2, 2))(eseb2)
  # <- encoder_conv_block_2

  # encoder_conv_block_3 ->
  econv3 = Activation('relu')(BatchNormalization()(Conv2D(128, (3, 3), padding='same')(pool2)))
  econv3 = Activation('relu')(BatchNormalization()(Conv2D(128, (3, 3), padding='same')(econv3)))
  eseb3 = SqueezeExciteBlock(econv3)
  pool3 = MaxPool2D((2, 2))(eseb3)
  # <- encoder_conv_block_3

  # encoder_conv_block_4 ->
  econv4 = Activation('relu')(BatchNormalization()(Conv2D(256, (3, 3), padding='same')(pool3)))
  econv4 = Activation('relu')(BatchNormalization()(Conv2D(256, (3, 3), padding='same')(econv4)))
  eseb4 = SqueezeExciteBlock(econv4)
  #drop1 = Dropout(0.5)(eseb4)
  pool4 = MaxPool2D((2, 2))(eseb4)
  aspp1 = ASPP(pool4, 64)
  # <- encoder_conv_block_4

  # center_block (bottleneck) ->
  #conv5 = Conv2D(64, 3, activation = 'relu', padding = 'same')(aspp1)
  #conv5 = Conv2D(64, 3, activation = 'relu', padding = 'same')(conv5)
  #drop2 = Dropout(0.5)(conv5)
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
  up1 = UpSampling2D((2, 2), interpolation='bilinear')(aspp1)
  merge1 = Concatenate()([up1, skip_connections[0], eseb4])
  dconv1 = Activation('relu')(BatchNormalization()(Conv2D(256, (3, 3), padding='same')(merge1)))
  dconv1 = Activation('relu')(BatchNormalization()(Conv2D(256, (3, 3), padding='same')(dconv1)))
  dseb1 = SqueezeExciteBlock(dconv1)
  # <- decoder_conv_block_1

  # decoder_conv_block_2 ->
  up2 = UpSampling2D((2, 2), interpolation='bilinear')(dseb1)
  merge2 = Concatenate()([up2, skip_connections[1], eseb3])
  dconv2 = Activation('relu')(BatchNormalization()(Conv2D(128, (3, 3), padding='same')(merge2)))
  dconv2 = Activation('relu')(BatchNormalization()(Conv2D(128, (3, 3), padding='same')(dconv2)))
  dseb2 = SqueezeExciteBlock(dconv2)
  # <- decoder_conv_block_2

  # decoder_conv_block_2 ->
  up3 = UpSampling2D((2, 2), interpolation='bilinear')(dseb2)
  merge3 = Concatenate()([up3, skip_connections[2], eseb2])
  dconv3 = Activation('relu')(BatchNormalization()(Conv2D(64, (3, 3), padding='same')(merge3)))
  dconv3 = Activation('relu')(BatchNormalization()(Conv2D(64, (3, 3), padding='same')(dconv3)))
  dseb3 = SqueezeExciteBlock(dconv3)
  # <- decoder_conv_block_2

  # decoder_conv_block_2 ->
  up4 = UpSampling2D((2, 2), interpolation='bilinear')(dseb3)
  merge4 = Concatenate()([up4, skip_connections[3], eseb1])
  dconv4 = Activation('relu')(BatchNormalization()(Conv2D(32, (3, 3), padding='same')(merge4)))
  dconv4 = Activation('relu')(BatchNormalization()(Conv2D(32, (3, 3), padding='same')(dconv4)))
  dseb4 = SqueezeExciteBlock(dconv4)
  # <- decoder_conv_block_2
  
  # output
  #conv6 = Conv2D(2, 3, activation = 'relu', padding = 'same')(dseb4)
  #conv7 = Conv2D(1, 1, activation = 'sigmoid')(conv6)

  conv7 = Activation('sigmoid')(Conv2D(1, (1, 1), padding="same")(dseb4))

  return conv7


def build_model(input_size):
  inputs = Input(input_size)

  # encoder 1 (vgg19) 
  x, skips = encoder_1(inputs)
  x = ASPP(x, 64)
  # decoder 1
  output_1 = decoder_1(x, skips)
  
  # multiply input and output 1
  x = inputs * output_1
  
  # unet
  output_2 = unet(x, skips)
  
  # concatenate output 1 and output 2
  outputs = Concatenate()([output_1, output_2])

  # model
  model = Model(inputs, outputs)

  return model
