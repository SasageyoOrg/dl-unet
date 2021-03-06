
from tensorflow.keras.layers import *
from tensorflow.keras.models import *

def unet(initializer, input_size):
    inputs = Input(input_size)
    #---------------------------ENCODING----------------------------------------
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    #---------------------------------------------------------------------------

    #---------------------------BOTTLENECK--------------------------------------
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv5)
    drop5 = Dropout(0.5)(conv5)
    #---------------------------------------------------------------------------

    #---------------------------DECODING--------------------------------------
    up6    = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = initializer)(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6  = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(merge6)
    conv6  = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv6)
    up7    = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = initializer)(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7  = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(merge7)
    conv7  = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv7)

    up8    = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = initializer)(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8  = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(merge8)
    conv8  = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv8)


    up9    = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = initializer)(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9  = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(merge9)
    conv9  = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv9)
    conv9  = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)
    #---------------------------------------------------------------------------
    model = Model(inputs,conv10)
    #model.summary()
    return model


def lightweight_unet(initializer, input_size):
#initializer = glorot_uniform()
#def unet(input_size = (160,224,1)):
    inputs = Input(input_size) 

    #-------------------------------ENCODING-------------------------------------
    # conv + bn + ReLU (red right arrow) - nr.1
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(inputs)
    batch1 = BatchNormalization(axis=3)(conv1)
    # conv + ReLU (orange right arrow) - nr.1
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(batch1)

    # max-pooling (yellow downwards arrow) - nr.1
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) 

    # conv + bn + ReLU (red right arrow)  - nr.2
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(pool1)
    batch2 = BatchNormalization(axis=3)(conv2)
    # conv + ReLU 2 (orange right arrow) - nr.2
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(batch2)

    # max-pooling (yellow downwards arrow) - nr.2
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) 

    # conv + bn + ReLU (red right arrow)  - nr.3
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(pool2)
    batch3 = BatchNormalization(axis=3)(conv3)
    # conv + ReLU (orange right arrow)  - nr.3
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(batch3)

    # max-pooling (yellow downwards arrow)  - nr.3
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3) 

    # conv + bn + ReLU (red right arrow) - nr.4
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(pool3)
    batch4 = BatchNormalization(axis=3)(conv4)
    # conv + ReLU (orange right arrow) - nr.4
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(batch4)
    #----------------------------------------------------------------------------

    #-------------------------------DECODING-------------------------------------
    # deconv + ReLU (blue towards arrow) - nr.1
    up5    = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = initializer)(UpSampling2D(size = (2,2))(conv4))
    # concatenation (white right arrow) - nr.1
    merge5 = concatenate([conv3,up5], axis = 3) 

    # conv + bn + ReLU (red right arrow) - nr.5
    conv5  = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(merge5)
    batch5 = BatchNormalization(axis=3)(conv5)
    # conv + ReLU (orange right arrow) - nr.5
    conv5  = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(batch5)
    
    # deconv + ReLU (blue towards arrow) - nr.2
    up6    = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = initializer)(UpSampling2D(size = (2,2))(conv5))
    # concatenation (white right arrow) - nr.2
    merge6 = concatenate([conv2,up6], axis = 3) 

    # conv + bn + ReLU (red right arrow) - nr.6
    conv6  = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(merge6)
    batch6 = BatchNormalization(axis=3)(conv6)
    # conv + ReLU (orange right arrow) - nr.6
    conv6  = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(batch6)

    # deconv + ReLU (blue towards arrow) - nr.3
    up7    = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = initializer)(UpSampling2D(size = (2,2))(conv6))
    # concatenation (white right arrow) - nr.3
    merge7 = concatenate([conv1,up7], axis = 3) 

    # conv + bn + ReLU (red right arrow) - nr.7
    conv7  = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(merge7)
    batch7 = BatchNormalization(axis=3)(conv7)
    # conv + ReLU (orange right arrow) - nr.7
    conv7  = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(batch7)

    # conv + sigmoid (gray right arrow)
    conv8 = Conv2D(1, 1, activation = 'sigmoid')(conv7)
    #----------------------------------------------------------------------------

    model = Model(inputs, conv8)

    model.summary()

    return model
