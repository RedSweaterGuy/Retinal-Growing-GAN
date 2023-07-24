import os

from keras import backend as K
from keras import losses, Sequential
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dense, GlobalAveragePooling2D
from keras.layers import Input
from keras.layers import LeakyReLU
from keras.layers.core import Activation, Flatten
from keras.layers import Concatenate
from keras.layers import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
#M: remove typeerror in keras
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()


os.environ['KERAS_BACKEND'] = 'tensorflow'
K.set_image_data_format('channels_last')

def generator(img_size, n_filters, name='g'):
    """
    generate network based on unet
    """

    # set image specifics
    k=3 # kernel size
    s=2 # stride
    img_ch=3 # image channels
    out_ch=1 # output channel
    img_height, img_width = img_size[0], img_size[1]
    padding='same'

    inputs = Input((img_height, img_width, img_ch), name="inputs")
    conv1 = Conv2D(n_filters, (k, k), padding=padding, name="conv1c")(inputs)
    conv1 = BatchNormalization(scale=False, axis=3, name="conv1b")(conv1)
    conv1 = Activation('relu',name="conv1a")(conv1)
    conv1 = Conv2D(n_filters, (k, k),  padding=padding, name="conv1c2")(conv1)
    conv1 = BatchNormalization(scale=False, axis=3, name="conv1b2")(conv1)
    conv1 = Activation('relu', name="conv1a2")(conv1)
    pool1 = MaxPooling2D(pool_size=(s, s), name="conv1m")(conv1)

    conv2 = Conv2D(2*n_filters, (k, k),  padding=padding, name="conv2c")(pool1)
    conv2 = BatchNormalization(scale=False, axis=3, name="conv2b")(conv2)
    conv2 = Activation('relu', name="conv2a")(conv2)
    conv2 = Conv2D(2*n_filters, (k, k),  padding=padding, name="conv2c2")(conv2)
    conv2 = BatchNormalization(scale=False, axis=3, name="conv2b2")(conv2)
    conv2 = Activation('relu', name="conv2a2")(conv2)
    pool2 = MaxPooling2D(pool_size=(s, s), name="conv2m")(conv2)

    conv3 = Conv2D(4*n_filters, (k, k),  padding=padding, name="conv3c")(pool2)
    conv3 = BatchNormalization(scale=False, axis=3, name="conv3b")(conv3)
    conv3 = Activation('relu', name="conv3a")(conv3)
    conv3 = Conv2D(4*n_filters, (k, k),  padding=padding, name="conv3c2")(conv3)
    conv3 = BatchNormalization(scale=False, axis=3, name="conv3b2")(conv3)
    conv3 = Activation('relu', name="conv3a2")(conv3)
    pool3 = MaxPooling2D(pool_size=(s, s), name="conv3m")(conv3)

    #80 80 128 geht hier rein
    conv4 = Conv2D(8*n_filters, (k, k),  padding=padding, name="conv4c")(pool3)
    conv4 = BatchNormalization(scale=False, axis=3, name="conv4b")(conv4)
    conv4 = Activation('relu', name="conv4a")(conv4)
    conv4 = Conv2D(8*n_filters, (k, k),  padding=padding, name="conv4c2")(conv4)
    conv4 = BatchNormalization(scale=False, axis=3, name="conv4b2")(conv4)
    conv4 = Activation('relu', name="conv4a2")(conv4)
    pool4 = MaxPooling2D(pool_size=(s, s), name="conv4m")(conv4)

    conv5 = Conv2D(16*n_filters, (k, k),  padding=padding, name="conv5c")(pool4)
    conv5 = BatchNormalization(scale=False, axis=3, name="conv5b")(conv5)
    conv5 = Activation('relu', name="conv5a")(conv5)
    conv5 = Conv2D(16*n_filters, (k, k),  padding=padding, name="conv5c2")(conv5)
    conv5 = BatchNormalization(scale=False, axis=3, name="conv5b2")(conv5)
    conv5 = Activation('relu', name="conv5a2")(conv5)

    up1 = Concatenate(axis=3, name="conv6cat")([UpSampling2D(size=(s, s))(conv5), conv4])
    conv6 = Conv2D(8*n_filters, (k, k),  padding=padding, name="conv6c")(up1)
    conv6 = BatchNormalization(scale=False, axis=3, name="conv6b")(conv6)
    conv6 = Activation('relu', name="conv6a")(conv6)
    conv6 = Conv2D(8*n_filters, (k, k),  padding=padding, name="conv6c2")(conv6)
    conv6 = BatchNormalization(scale=False, axis=3, name="conv6b2")(conv6)
    conv6 = Activation('relu', name="conv6a2")(conv6)
    #gibt 80 80 256 aus

    up2 = Concatenate(axis=3, name="conv7cat")([UpSampling2D(size=(s, s))(conv6), conv3])
    conv7 = Conv2D(4*n_filters, (k, k),  padding=padding, name="conv7c")(up2)
    conv7 = BatchNormalization(scale=False, axis=3, name="conv7b")(conv7)
    conv7 = Activation('relu', name="conv7a")(conv7)
    conv7 = Conv2D(4*n_filters, (k, k),  padding=padding, name="conv7c2")(conv7)
    conv7 = BatchNormalization(scale=False, axis=3, name="conv7b2")(conv7)
    conv7 = Activation('relu', name="conv7a2")(conv7)

    up3 = Concatenate(axis=3, name="conv8cat")([UpSampling2D(size=(s, s))(conv7), conv2])
    conv8 = Conv2D(2*n_filters, (k, k),  padding=padding, name="conv8c")(up3)
    conv8 = BatchNormalization(scale=False, axis=3, name="conv8b")(conv8)
    conv8 = Activation('relu', name="conv8a")(conv8)
    conv8 = Conv2D(2*n_filters, (k, k),  padding=padding, name="conv8c2")(conv8)
    conv8 = BatchNormalization(scale=False, axis=3, name="conv8b2")(conv8)
    conv8 = Activation('relu', name="conv8a2")(conv8)

    up4 = Concatenate(axis=3, name="conv9cat")([UpSampling2D(size=(s, s))(conv8), conv1])
    conv9 = Conv2D(n_filters, (k, k),  padding=padding, name="conv9c")(up4)
    conv9 = BatchNormalization(scale=False, axis=3, name="conv9b")(conv9)
    conv9 = Activation('relu', name="conv9a")(conv9)
    conv9 = Conv2D(n_filters, (k, k),  padding=padding, name="conv9c2")(conv9)
    conv9 = BatchNormalization(scale=False, axis=3, name="conv9b2")(conv9)
    conv9 = Activation('relu', name="conv9a2")(conv9)

    outputs = Conv2D(out_ch, (1, 1), padding=padding, activation='sigmoid', name="outputs")(conv9)

    g = Model(inputs, outputs, name=name)

    return g


def generator1(img_size, n_filters, name='g'):
    """
    generate network based on unet
    """

    # set image specifics
    k = 3  # kernel size
    s = 2  # stride
    img_ch = 3  # image channels
    out_ch = 1  # output channel
    img_height, img_width = img_size[0], img_size[1]
    padding = 'same'

    inputs = Input((img_height, img_width, img_ch), name="inputs1")

    # 80 80 128 geht hier rein
    conv4 = Conv2D(8 * n_filters, (k, k), padding=padding, name="conv4c")(inputs)
    conv4 = BatchNormalization(scale=False, axis=3, name="conv4b")(conv4)
    conv4 = Activation('relu', name="conv4a")(conv4)
    conv4 = Conv2D(8 * n_filters, (k, k), padding=padding, name="conv4c2")(conv4)
    conv4 = BatchNormalization(scale=False, axis=3, name="conv4b2")(conv4)
    conv4 = Activation('relu', name="conv4a2")(conv4)
    pool4 = MaxPooling2D(pool_size=(s, s), name="conv4m")(conv4)

    conv5 = Conv2D(16 * n_filters, (k, k), padding=padding, name="conv5c")(pool4)
    conv5 = BatchNormalization(scale=False, axis=3, name="conv5b")(conv5)
    conv5 = Activation('relu', name="conv5a")(conv5)
    conv5 = Conv2D(16 * n_filters, (k, k), padding=padding, name="conv5c2")(conv5)
    conv5 = BatchNormalization(scale=False, axis=3, name="conv5b2")(conv5)
    conv5 = Activation('relu', name="conv5a2")(conv5)

    up1 = Concatenate(axis=3, name="conv6cat")([UpSampling2D(size=(s, s), name="up1")(conv5), conv4])
    conv6 = Conv2D(8 * n_filters, (k, k), padding=padding, name="conv6c")(up1)
    conv6 = BatchNormalization(scale=False, axis=3, name="conv6b")(conv6)
    conv6 = Activation('relu', name="conv6a")(conv6)
    conv6 = Conv2D(8 * n_filters, (k, k), padding=padding, name="conv6c2")(conv6)
    conv6 = BatchNormalization(scale=False, axis=3, name="conv6b2")(conv6)
    conv6 = Activation('relu', name="conv6a2")(conv6)
    # gibt 80 80 256 aus

    outputs = Conv2D(out_ch, (1, 1), padding=padding, activation='sigmoid', name="outputs1")(conv6)

    g = Model(inputs, outputs, name=name)

    return g


def generator2(img_size, n_filters, name='g'):
    """
    generate network based on unet
    """

    # set image specifics
    k = 3  # kernel size
    s = 2  # stride
    img_ch = 3  # image channels
    out_ch = 1  # output channel
    img_height, img_width = img_size[0], img_size[1]
    padding = 'same'

    inputs = Input((img_height, img_width, img_ch), name="inputs2")

    conv3 = Conv2D(4 * n_filters, (k, k), padding=padding, name="conv3c")(inputs)
    conv3 = BatchNormalization(scale=False, axis=3, name="conv3b")(conv3)
    conv3 = Activation('relu', name="conv3a")(conv3)
    conv3 = Conv2D(4 * n_filters, (k, k), padding=padding, name="conv3c2")(conv3)
    conv3 = BatchNormalization(scale=False, axis=3, name="conv3b2")(conv3)
    conv3 = Activation('relu', name="conv3a2")(conv3)
    pool3 = MaxPooling2D(pool_size=(s, s), name="conv3m")(conv3)

    # 80 80 128 geht hier rein
    conv4 = Conv2D(8 * n_filters, (k, k), padding=padding, name="conv4c")(pool3)
    conv4 = BatchNormalization(scale=False, axis=3, name="conv4b")(conv4)
    conv4 = Activation('relu', name="conv4a")(conv4)
    conv4 = Conv2D(8 * n_filters, (k, k), padding=padding, name="conv4c2")(conv4)
    conv4 = BatchNormalization(scale=False, axis=3, name="conv4b2")(conv4)
    conv4 = Activation('relu', name="conv4a2")(conv4)
    pool4 = MaxPooling2D(pool_size=(s, s), name="conv4m")(conv4)

    conv5 = Conv2D(16 * n_filters, (k, k), padding=padding, name="conv5c")(pool4)
    conv5 = BatchNormalization(scale=False, axis=3, name="conv5b")(conv5)
    conv5 = Activation('relu', name="conv5a")(conv5)
    conv5 = Conv2D(16 * n_filters, (k, k), padding=padding, name="conv5c2")(conv5)
    conv5 = BatchNormalization(scale=False, axis=3, name="conv5b2")(conv5)
    conv5 = Activation('relu', name="conv5a2")(conv5)

    up1 = Concatenate(axis=3, name="conv6cat")([UpSampling2D(size=(s, s), name="up1")(conv5), conv4])
    conv6 = Conv2D(8 * n_filters, (k, k), padding=padding, name="conv6c")(up1)
    conv6 = BatchNormalization(scale=False, axis=3, name="conv6b")(conv6)
    conv6 = Activation('relu', name="conv6a")(conv6)
    conv6 = Conv2D(8 * n_filters, (k, k), padding=padding, name="conv6c2")(conv6)
    conv6 = BatchNormalization(scale=False, axis=3, name="conv6b2")(conv6)
    conv6 = Activation('relu', name="conv6a2")(conv6)
    # gibt 80 80 256 aus

    up2 = Concatenate(axis=3, name="conv7cat")([UpSampling2D(size=(s, s), name="up2")(conv6), conv3])
    conv7 = Conv2D(4 * n_filters, (k, k), padding=padding, name="conv7c")(up2)
    conv7 = BatchNormalization(scale=False, axis=3, name="conv7b")(conv7)
    conv7 = Activation('relu', name="conv7a")(conv7)
    conv7 = Conv2D(4 * n_filters, (k, k), padding=padding, name="conv7c2")(conv7)
    conv7 = BatchNormalization(scale=False, axis=3, name="conv7b2")(conv7)
    conv7 = Activation('relu', name="conv7a2")(conv7)

    outputs = Conv2D(out_ch, (1, 1), padding=padding, activation='sigmoid', name="outputs2")(conv7)

    g = Model(inputs, outputs, name=name)

    return g


def generator3(img_size, n_filters, name='g'):
    """
    generate network based on unet
    """

    # set image specifics
    k = 3  # kernel size
    s = 2  # stride
    img_ch = 3  # image channels
    out_ch = 1  # output channel
    img_height, img_width = img_size[0], img_size[1]
    padding = 'same'

    inputs = Input((img_height, img_width, img_ch), name="inputs3")

    conv2 = Conv2D(2 * n_filters, (k, k), padding=padding, name="conv2c")(inputs)
    conv2 = BatchNormalization(scale=False, axis=3, name="conv2b")(conv2)
    conv2 = Activation('relu', name="conv2a")(conv2)
    conv2 = Conv2D(2 * n_filters, (k, k), padding=padding, name="conv2c2")(conv2)
    conv2 = BatchNormalization(scale=False, axis=3, name="conv2b2")(conv2)
    conv2 = Activation('relu', name="conv2a2")(conv2)
    pool2 = MaxPooling2D(pool_size=(s, s), name="conv2m")(conv2)

    conv3 = Conv2D(4 * n_filters, (k, k), padding=padding, name="conv3c")(pool2)
    conv3 = BatchNormalization(scale=False, axis=3, name="conv3b")(conv3)
    conv3 = Activation('relu', name="conv3a")(conv3)
    conv3 = Conv2D(4 * n_filters, (k, k), padding=padding, name="conv3c2")(conv3)
    conv3 = BatchNormalization(scale=False, axis=3, name="conv3b2")(conv3)
    conv3 = Activation('relu', name="conv3a2")(conv3)
    pool3 = MaxPooling2D(pool_size=(s, s), name="conv3m")(conv3)

    # 80 80 128 geht hier rein
    conv4 = Conv2D(8 * n_filters, (k, k), padding=padding, name="conv4c")(pool3)
    conv4 = BatchNormalization(scale=False, axis=3, name="conv4b")(conv4)
    conv4 = Activation('relu', name="conv4a")(conv4)
    conv4 = Conv2D(8 * n_filters, (k, k), padding=padding, name="conv4c2")(conv4)
    conv4 = BatchNormalization(scale=False, axis=3, name="conv4b2")(conv4)
    conv4 = Activation('relu', name="conv4a2")(conv4)
    pool4 = MaxPooling2D(pool_size=(s, s), name="conv4m")(conv4)

    conv5 = Conv2D(16 * n_filters, (k, k), padding=padding, name="conv5c")(pool4)
    conv5 = BatchNormalization(scale=False, axis=3, name="conv5b")(conv5)
    conv5 = Activation('relu', name="conv5a")(conv5)
    conv5 = Conv2D(16 * n_filters, (k, k), padding=padding, name="conv5c2")(conv5)
    conv5 = BatchNormalization(scale=False, axis=3, name="conv5b2")(conv5)
    conv5 = Activation('relu', name="conv5a2")(conv5)

    up1 = Concatenate(axis=3, name="conv6cat")([UpSampling2D(size=(s, s), name="up1")(conv5), conv4])
    conv6 = Conv2D(8 * n_filters, (k, k), padding=padding, name="conv6c")(up1)
    conv6 = BatchNormalization(scale=False, axis=3, name="conv6b")(conv6)
    conv6 = Activation('relu', name="conv6a")(conv6)
    conv6 = Conv2D(8 * n_filters, (k, k), padding=padding, name="conv6c2")(conv6)
    conv6 = BatchNormalization(scale=False, axis=3, name="conv6b2")(conv6)
    conv6 = Activation('relu', name="conv6a2")(conv6)
    # gibt 80 80 256 aus

    up2 = Concatenate(axis=3, name="conv7cat")([UpSampling2D(size=(s, s), name="up2")(conv6), conv3])
    conv7 = Conv2D(4 * n_filters, (k, k), padding=padding, name="conv7c")(up2)
    conv7 = BatchNormalization(scale=False, axis=3, name="conv7b")(conv7)
    conv7 = Activation('relu', name="conv7a")(conv7)
    conv7 = Conv2D(4 * n_filters, (k, k), padding=padding, name="conv7c2")(conv7)
    conv7 = BatchNormalization(scale=False, axis=3, name="conv7b2")(conv7)
    conv7 = Activation('relu', name="conv7a2")(conv7)

    up3 = Concatenate(axis=3, name="conv8cat")([UpSampling2D(size=(s, s), name="up3")(conv7), conv2])
    conv8 = Conv2D(2 * n_filters, (k, k), padding=padding, name="conv8c")(up3)
    conv8 = BatchNormalization(scale=False, axis=3, name="conv8b")(conv8)
    conv8 = Activation('relu', name="conv8a")(conv8)
    conv8 = Conv2D(2 * n_filters, (k, k), padding=padding, name="conv8c2")(conv8)
    conv8 = BatchNormalization(scale=False, axis=3, name="conv8b2")(conv8)
    conv8 = Activation('relu', name="conv8a2")(conv8)

    outputs = Conv2D(out_ch, (1, 1), padding=padding, activation='sigmoid', name="outputs3")(conv8)

    g = Model(inputs, outputs, name=name)

    return g


def generator4(img_size, n_filters, name='g'):
    """
    generate network based on unet
    """

    # set image specifics
    k = 3  # kernel size
    s = 2  # stride
    img_ch = 3  # image channels
    out_ch = 1  # output channel
    img_height, img_width = img_size[0], img_size[1]
    padding = 'same'

    inputs = Input((img_height, img_width, img_ch), name="inputs4")
    conv1 = Conv2D(n_filters, (k, k), padding=padding, name="conv1c")(inputs)
    conv1 = BatchNormalization(scale=False, axis=3, name="conv1b")(conv1)
    conv1 = Activation('relu', name="conv1a")(conv1)
    conv1 = Conv2D(n_filters, (k, k), padding=padding, name="conv1c2")(conv1)
    conv1 = BatchNormalization(scale=False, axis=3, name="conv1b2")(conv1)
    conv1 = Activation('relu', name="conv1a2")(conv1)
    pool1 = MaxPooling2D(pool_size=(s, s), name="conv1m")(conv1)

    conv2 = Conv2D(2 * n_filters, (k, k), padding=padding, name="conv2c")(pool1)
    conv2 = BatchNormalization(scale=False, axis=3, name="conv2b")(conv2)
    conv2 = Activation('relu', name="conv2a")(conv2)
    conv2 = Conv2D(2 * n_filters, (k, k), padding=padding, name="conv2c2")(conv2)
    conv2 = BatchNormalization(scale=False, axis=3, name="conv2b2")(conv2)
    conv2 = Activation('relu', name="conv2a2")(conv2)
    pool2 = MaxPooling2D(pool_size=(s, s), name="conv2m")(conv2)

    conv3 = Conv2D(4 * n_filters, (k, k), padding=padding, name="conv3c")(pool2)
    conv3 = BatchNormalization(scale=False, axis=3, name="conv3b")(conv3)
    conv3 = Activation('relu', name="conv3a")(conv3)
    conv3 = Conv2D(4 * n_filters, (k, k), padding=padding, name="conv3c2")(conv3)
    conv3 = BatchNormalization(scale=False, axis=3, name="conv3b2")(conv3)
    conv3 = Activation('relu', name="conv3a2")(conv3)
    pool3 = MaxPooling2D(pool_size=(s, s), name="conv3m")(conv3)

    # 80 80 128 geht hier rein
    conv4 = Conv2D(8 * n_filters, (k, k), padding=padding, name="conv4c")(pool3)
    conv4 = BatchNormalization(scale=False, axis=3, name="conv4b")(conv4)
    conv4 = Activation('relu', name="conv4a")(conv4)
    conv4 = Conv2D(8 * n_filters, (k, k), padding=padding, name="conv4c2")(conv4)
    conv4 = BatchNormalization(scale=False, axis=3, name="conv4b2")(conv4)
    conv4 = Activation('relu', name="conv4a2")(conv4)
    pool4 = MaxPooling2D(pool_size=(s, s), name="conv4m")(conv4)

    conv5 = Conv2D(16 * n_filters, (k, k), padding=padding, name="conv5c")(pool4)
    conv5 = BatchNormalization(scale=False, axis=3, name="conv5b")(conv5)
    conv5 = Activation('relu', name="conv5a")(conv5)
    conv5 = Conv2D(16 * n_filters, (k, k), padding=padding, name="conv5c2")(conv5)
    conv5 = BatchNormalization(scale=False, axis=3, name="conv5b2")(conv5)
    conv5 = Activation('relu', name="conv5a2")(conv5)

    up1 = Concatenate(axis=3, name="conv6cat")([UpSampling2D(size=(s, s), name="up1")(conv5), conv4])
    conv6 = Conv2D(8 * n_filters, (k, k), padding=padding, name="conv6c")(up1)
    conv6 = BatchNormalization(scale=False, axis=3, name="conv6b")(conv6)
    conv6 = Activation('relu', name="conv6a")(conv6)
    conv6 = Conv2D(8 * n_filters, (k, k), padding=padding, name="conv6c2")(conv6)
    conv6 = BatchNormalization(scale=False, axis=3, name="conv6b2")(conv6)
    conv6 = Activation('relu', name="conv6a2")(conv6)
    # gibt 80 80 256 aus

    up2 = Concatenate(axis=3, name="conv7cat")([UpSampling2D(size=(s, s), name="up2")(conv6), conv3])
    conv7 = Conv2D(4 * n_filters, (k, k), padding=padding, name="conv7c")(up2)
    conv7 = BatchNormalization(scale=False, axis=3, name="conv7b")(conv7)
    conv7 = Activation('relu', name="conv7a")(conv7)
    conv7 = Conv2D(4 * n_filters, (k, k), padding=padding, name="conv7c2")(conv7)
    conv7 = BatchNormalization(scale=False, axis=3, name="conv7b2")(conv7)
    conv7 = Activation('relu', name="conv7a2")(conv7)

    up3 = Concatenate(axis=3, name="conv8cat")([UpSampling2D(size=(s, s), name="up3")(conv7), conv2])
    conv8 = Conv2D(2 * n_filters, (k, k), padding=padding, name="conv8c")(up3)
    conv8 = BatchNormalization(scale=False, axis=3, name="conv8b")(conv8)
    conv8 = Activation('relu', name="conv8a")(conv8)
    conv8 = Conv2D(2 * n_filters, (k, k), padding=padding, name="conv8c2")(conv8)
    conv8 = BatchNormalization(scale=False, axis=3, name="conv8b2")(conv8)
    conv8 = Activation('relu', name="conv8a2")(conv8)

    up4 = Concatenate(axis=3, name="conv9cat")([UpSampling2D(size=(s, s), name="up4")(conv8), conv1])
    conv9 = Conv2D(n_filters, (k, k), padding=padding, name="conv9c")(up4)
    conv9 = BatchNormalization(scale=False, axis=3, name="conv9b")(conv9)
    conv9 = Activation('relu', name="conv9a")(conv9)
    conv9 = Conv2D(n_filters, (k, k), padding=padding, name="conv9c2")(conv9)
    conv9 = BatchNormalization(scale=False, axis=3, name="conv9b2")(conv9)
    conv9 = Activation('relu', name="conv9a2")(conv9)

    outputs = Conv2D(out_ch, (1, 1), padding=padding, activation='sigmoid', name="outputs4")(conv9)

    g = Model(inputs, outputs, name=name)

    return g


def discriminator_pixel(img_size, n_filters, init_lr, name='d'):
    """
    discriminator network (pixel GAN)
    """
    
    # set image specifics
    k=3 # kernel size
    img_ch=3 # image channels
    out_ch=1 # output channel
    img_height, img_width = img_size[0], img_size[1]
    
    inputs = Input((img_height, img_width, img_ch + out_ch))

    conv1 = Conv2D(n_filters, kernel_size=(k, k), padding="same")(inputs) 
    conv1 = LeakyReLU(0.2)(conv1)
    
    conv2 = Conv2D(2*n_filters, kernel_size=(k, k), padding="same")(conv1) 
    conv2 = LeakyReLU(0.2)(conv2)
    
    conv3 = Conv2D(4*n_filters, kernel_size=(k, k), padding="same")(conv2) 
    conv3 = LeakyReLU(0.2)(conv3)

    conv4 =  Conv2D(out_ch, kernel_size=(1, 1), padding="same")(conv3)
    outputs = Activation('sigmoid')(conv4)

    d = Model(inputs, outputs, name=name)

    def d_loss(y_true, y_pred):
        L = losses.binary_crossentropy(K.batch_flatten(y_true),
                                           K.batch_flatten(y_pred))
        return L

    d.compile(optimizer=Adam(lr=init_lr, beta_1=0.5), loss=d_loss, metrics=['accuracy'])
    
    return d, d.layers[-1].output_shape[1:]

def discriminator_patch1(img_size, n_filters, init_lr, name='d'):
    """
    discriminator network (patch GAN)
    stride 2 conv X 1
    max pooling X 2
    """
    
    # set image specifics
    k=3 # kernel size
    s=2 # stride
    img_ch=3 # image channels
    out_ch=1 # output channel
    img_height, img_width = img_size[0], img_size[1]
    padding='same'#'valid'

    inputs = Input((img_height, img_width, img_ch + out_ch))

    conv1 = Conv2D(n_filters, kernel_size=(k, k), strides=(s,s), padding=padding)(inputs)
    conv1 = BatchNormalization(scale=False, axis=3)(conv1)
    conv1 = Activation('relu')(conv1)    
    conv1 = Conv2D(n_filters, kernel_size=(k, k), padding=padding)(conv1) 
    conv1 = BatchNormalization(scale=False, axis=3)(conv1)
    conv1 = Activation('relu')(conv1)    
    pool1 = MaxPooling2D(pool_size=(s, s))(conv1)
    
    conv2 = Conv2D(2*n_filters, kernel_size=(k, k), padding=padding)(pool1) 
    conv2 = BatchNormalization(scale=False, axis=3)(conv2)
    conv2 = Activation('relu')(conv2)    
    conv2 = Conv2D(2*n_filters, kernel_size=(k, k), padding=padding)(conv2) 
    conv2 = BatchNormalization(scale=False, axis=3)(conv2)
    conv2 = Activation('relu')(conv2)    
    pool2 = MaxPooling2D(pool_size=(s, s))(conv2)
    
    conv3 = Conv2D(4*n_filters, kernel_size=(k, k), padding=padding)(pool2) 
    conv3 = BatchNormalization(scale=False, axis=3)(conv3)
    conv3 = Activation('relu')(conv3)    
    conv3 = Conv2D(4*n_filters, kernel_size=(k, k), padding=padding)(conv3) 
    conv3 = BatchNormalization(scale=False, axis=3)(conv3)
    conv3 = Activation('relu')(conv3)
    
    outputs=Conv2D(out_ch, kernel_size=(1, 1), padding=padding, activation='sigmoid')(conv3)

    d = Model(inputs, outputs, name=name)

    def d_loss(y_true, y_pred):
        L = losses.binary_crossentropy(K.batch_flatten(y_true),
                                            K.batch_flatten(y_pred))
#         L = losses.mean_squared_error(K.batch_flatten(y_true),
#                                            K.batch_flatten(y_pred))
        return L

    d.compile(optimizer=Adam(lr=init_lr, beta_1=0.5), loss=d_loss, metrics=['accuracy'])
    
    return d, d.layers[-1].output_shape[1:]

def discriminator_patch2(img_size, n_filters, init_lr, name='d'):
    """
    discriminator network (patch GAN)
     stride 2 conv X 2
       max pooling X 4
    """
    
    # set image specifics
    k=3 # kernel size
    s=2 # stride
    img_ch=3 # image channels
    out_ch=1 # output channel
    img_height, img_width = img_size[0], img_size[1]
    padding='same'#'valid'

    inputs = Input((img_height, img_width, img_ch + out_ch))

    conv1 = Conv2D(n_filters, kernel_size=(k, k), strides=(s,s), padding=padding)(inputs)
    conv1 = BatchNormalization(scale=False, axis=3)(conv1)
    conv1 = Activation('relu')(conv1)    
    conv1 = Conv2D(n_filters, kernel_size=(k, k), padding=padding)(conv1) 
    conv1 = BatchNormalization(scale=False, axis=3)(conv1)
    conv1 = Activation('relu')(conv1)    
    pool1 = MaxPooling2D(pool_size=(s, s))(conv1)
    
    conv2 = Conv2D(2*n_filters, kernel_size=(k, k), strides=(s,s), padding=padding)(pool1) 
    conv2 = BatchNormalization(scale=False, axis=3)(conv2)
    conv2 = Activation('relu')(conv2)    
    conv2 = Conv2D(2*n_filters, kernel_size=(k, k), padding=padding)(conv2) 
    conv2 = BatchNormalization(scale=False, axis=3)(conv2)
    conv2 = Activation('relu')(conv2)    
    pool2 = MaxPooling2D(pool_size=(s, s))(conv2)
    
    conv3 = Conv2D(4*n_filters, kernel_size=(k, k), padding=padding)(pool2) 
    conv3 = BatchNormalization(scale=False, axis=3)(conv3)
    conv3 = Activation('relu')(conv3)    
    conv3 = Conv2D(4*n_filters, kernel_size=(k, k), padding=padding)(conv3) 
    conv3 = BatchNormalization(scale=False, axis=3)(conv3)
    conv3 = Activation('relu')(conv3)    
    pool3 = MaxPooling2D(pool_size=(s, s))(conv3)
    
    conv4 = Conv2D(8*n_filters, kernel_size=(k, k), padding=padding)(pool3) 
    conv4 = BatchNormalization(scale=False, axis=3)(conv4)
    conv4 = Activation('relu')(conv4)    
    conv4 = Conv2D(8*n_filters, kernel_size=(k, k), padding=padding)(conv4) 
    conv4 = BatchNormalization(scale=False, axis=3)(conv4)
    conv4 = Activation('relu')(conv4)    
    pool4 = MaxPooling2D(pool_size=(s, s))(conv4)
    
    conv5 = Conv2D(16*n_filters, kernel_size=(k, k), padding=padding)(pool4) 
    conv5 = BatchNormalization(scale=False, axis=3)(conv5)
    conv5 = Activation('relu')(conv5)    
    conv5 = Conv2D(16*n_filters, kernel_size=(k, k), padding=padding)(conv5) 
    conv5 = BatchNormalization(scale=False, axis=3)(conv5)
    conv5 = Activation('relu')(conv5)    
    
    outputs=Conv2D(out_ch, kernel_size=(1, 1), padding=padding, activation='sigmoid')(conv5)

    d = Model(inputs, outputs, name=name)

    def d_loss(y_true, y_pred):
        L = losses.binary_crossentropy(K.batch_flatten(y_true),
                                            K.batch_flatten(y_pred))
#         L = losses.mean_squared_error(K.batch_flatten(y_true),
#                                            K.batch_flatten(y_pred))
        return L

    d.compile(optimizer=Adam(lr=init_lr, beta_1=0.5), loss=d_loss, metrics=['accuracy'])
    
    return d, d.layers[-1].output_shape[1:]

def discriminator_image(img_size, n_filters, init_lr, name='d'):
    """
    discriminator network (patch GAN)
      stride 2 conv X 2
        max pooling X 4
    fully connected X 1
    """

    # set image specifics
    k = 3  # kernel size
    s = 2  # stride
    img_ch = 3  # image channels
    out_ch = 1  # output channel
    img_height, img_width = img_size[0], img_size[1]
    padding = 'same'  # 'valid'

    inputs = Input((img_height, img_width, img_ch + out_ch))

    conv1 = Conv2D(n_filters, kernel_size=(k, k), padding=padding)(inputs)
    conv1 = BatchNormalization(scale=False, axis=3)(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv2D(n_filters, kernel_size=(k, k), padding=padding)(conv1)
    conv1 = BatchNormalization(scale=False, axis=3)(conv1)
    conv1 = Activation('relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(s, s))(conv1)

    conv2 = Conv2D(2 * n_filters, kernel_size=(k, k), padding=padding)(pool1)
    conv2 = BatchNormalization(scale=False, axis=3)(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Conv2D(2 * n_filters, kernel_size=(k, k), padding=padding)(conv2)
    conv2 = BatchNormalization(scale=False, axis=3)(conv2)
    conv2 = Activation('relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(s, s))(conv2)

    conv3 = Conv2D(4 * n_filters, kernel_size=(k, k), padding=padding)(pool2)
    conv3 = BatchNormalization(scale=False, axis=3)(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Conv2D(4 * n_filters, kernel_size=(k, k), padding=padding)(conv3)
    conv3 = BatchNormalization(scale=False, axis=3)(conv3)
    conv3 = Activation('relu')(conv3)
    pool3 = MaxPooling2D(pool_size=(s, s))(conv3)

    conv4 = Conv2D(8 * n_filters, kernel_size=(k, k), padding=padding)(pool3)
    conv4 = BatchNormalization(scale=False, axis=3)(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = Conv2D(8 * n_filters, kernel_size=(k, k), padding=padding)(conv4)
    conv4 = BatchNormalization(scale=False, axis=3)(conv4)
    conv4 = Activation('relu')(conv4)
    pool4 = MaxPooling2D(pool_size=(s, s))(conv4)

    conv5 = Conv2D(16 * n_filters, kernel_size=(k, k), strides=(s, s), padding=padding)(pool4)
    conv5 = BatchNormalization(scale=False, axis=3)(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Conv2D(16 * n_filters, kernel_size=(k, k), padding=padding)(conv5)
    conv5 = BatchNormalization(scale=False, axis=3)(conv5)
    conv5 = Activation('relu')(conv5)

    conv6 = Conv2D(16 * n_filters, kernel_size=(k, k), strides=(s, s), padding=padding)(conv5)
    conv6 = BatchNormalization(scale=False, axis=3)(conv6)
    conv6 = Activation('relu')(conv6)
    conv6 = Conv2D(16 * n_filters, kernel_size=(k, k), padding=padding)(conv6)
    conv6 = BatchNormalization(scale=False, axis=3)(conv6)
    conv6 = Activation('relu')(conv6)

    gap = GlobalAveragePooling2D()(conv6)
    outputs = Dense(1, activation='sigmoid')(gap)

    d = Model(inputs, outputs, name=name)

    def d_loss(y_true, y_pred):
        L = losses.binary_crossentropy(K.batch_flatten(y_true),
                                       K.batch_flatten(y_pred))
        #         L = losses.mean_squared_error(K.batch_flatten(y_true),
        #                                            K.batch_flatten(y_pred))
        return L

    d.compile(optimizer=Adam(lr=init_lr, beta_1=0.5), loss=d_loss, metrics=['accuracy'])

    return d, d.layers[-1].output_shape[1:]


def discriminator_image1(img_size, n_filters, init_lr, name='d'):
    """
    discriminator network (patch GAN)
      stride 2 conv X 2
        max pooling X 4
    fully connected X 1
    """

    # set image specifics
    k = 3  # kernel size
    s = 2  # stride
    img_ch = 3  # image channels
    out_ch = 1  # output channel
    img_height, img_width = img_size[0], img_size[1]
    padding = 'same'  # 'valid'

    inputs = Input((img_height, img_width, img_ch + out_ch), name="inputs1")

    conv4 = Conv2D(8 * n_filters, kernel_size=(k, k), padding=padding, name="conv4c")(inputs)
    conv4 = BatchNormalization(scale=False, axis=3, name="conv4b")(conv4)
    conv4 = Activation('relu', name="conv4a")(conv4)
    conv4 = Conv2D(8 * n_filters, kernel_size=(k, k), padding=padding, name="conv4c2")(conv4)
    conv4 = BatchNormalization(scale=False, axis=3, name="conv4b2")(conv4)
    conv4 = Activation('relu', name="conv4a2")(conv4)
    pool4 = MaxPooling2D(pool_size=(s, s), name="conv4m")(conv4)

    conv5 = Conv2D(16 * n_filters, kernel_size=(k, k), strides=(s, s), padding=padding, name="conv5c")(pool4)
    conv5 = BatchNormalization(scale=False, axis=3, name="conv5b")(conv5)
    conv5 = Activation('relu', name="conv5a")(conv5)
    conv5 = Conv2D(16 * n_filters, kernel_size=(k, k), padding=padding, name="conv5c2")(conv5)
    conv5 = BatchNormalization(scale=False, axis=3, name="conv5b2")(conv5)
    conv5 = Activation('relu', name="conv5a2")(conv5)

    conv6 = Conv2D(16 * n_filters, kernel_size=(k, k), strides=(s, s), padding=padding, name="conv6c")(conv5)
    conv6 = BatchNormalization(scale=False, axis=3, name="conv6b")(conv6)
    conv6 = Activation('relu', name="conv6a")(conv6)
    conv6 = Conv2D(16 * n_filters, kernel_size=(k, k), padding=padding, name="conv6c2")(conv6)
    conv6 = BatchNormalization(scale=False, axis=3, name="conv6b2")(conv6)
    conv6 = Activation('relu', name="conv6a2")(conv6)

    gap = GlobalAveragePooling2D(name="gap")(conv6)
    outputs = Dense(1, activation='sigmoid', name="output")(gap)

    d = Model(inputs, outputs, name=name)

    def d_loss(y_true, y_pred):
        L = losses.binary_crossentropy(K.batch_flatten(y_true),
                                       K.batch_flatten(y_pred))
        #         L = losses.mean_squared_error(K.batch_flatten(y_true),
        #                                            K.batch_flatten(y_pred))
        return L

    d.compile(optimizer=Adam(lr=init_lr, beta_1=0.5), loss=d_loss, metrics=['accuracy'])

    return d, d.layers[-1].output_shape[1:]

def discriminator_image2(img_size, n_filters, init_lr, name='d'):
    """
    discriminator network (patch GAN)
      stride 2 conv X 2
        max pooling X 4
    fully connected X 1
    """

    # set image specifics
    k = 3  # kernel size
    s = 2  # stride
    img_ch = 3  # image channels
    out_ch = 1  # output channel
    img_height, img_width = img_size[0], img_size[1]
    padding = 'same'  # 'valid'

    inputs = Input((img_height, img_width, img_ch + out_ch), name="inputs2")

    conv3 = Conv2D(4 * n_filters, kernel_size=(k, k), padding=padding, name="conv3c")(inputs)
    conv3 = BatchNormalization(scale=False, axis=3, name="conv3b")(conv3)
    conv3 = Activation('relu', name="conv3a")(conv3)
    conv3 = Conv2D(4 * n_filters, kernel_size=(k, k), padding=padding, name="conv3c2")(conv3)
    conv3 = BatchNormalization(scale=False, axis=3, name="conv3b2")(conv3)
    conv3 = Activation('relu', name="conv3a2")(conv3)
    pool3 = MaxPooling2D(pool_size=(s, s), name="conv3m")(conv3)

    conv4 = Conv2D(8 * n_filters, kernel_size=(k, k), padding=padding, name="conv4c")(pool3)
    conv4 = BatchNormalization(scale=False, axis=3, name="conv4b")(conv4)
    conv4 = Activation('relu', name="conv4a")(conv4)
    conv4 = Conv2D(8 * n_filters, kernel_size=(k, k), padding=padding, name="conv4c2")(conv4)
    conv4 = BatchNormalization(scale=False, axis=3, name="conv4b2")(conv4)
    conv4 = Activation('relu', name="conv4a2")(conv4)
    pool4 = MaxPooling2D(pool_size=(s, s), name="conv4m")(conv4)

    conv5 = Conv2D(16 * n_filters, kernel_size=(k, k), strides=(s, s), padding=padding, name="conv5c")(pool4)
    conv5 = BatchNormalization(scale=False, axis=3, name="conv5b")(conv5)
    conv5 = Activation('relu', name="conv5a")(conv5)
    conv5 = Conv2D(16 * n_filters, kernel_size=(k, k), padding=padding, name="conv5c2")(conv5)
    conv5 = BatchNormalization(scale=False, axis=3, name="conv5b2")(conv5)
    conv5 = Activation('relu', name="conv5a2")(conv5)

    conv6 = Conv2D(16 * n_filters, kernel_size=(k, k), strides=(s, s), padding=padding, name="conv6c")(conv5)
    conv6 = BatchNormalization(scale=False, axis=3, name="conv6b")(conv6)
    conv6 = Activation('relu', name="conv6a")(conv6)
    conv6 = Conv2D(16 * n_filters, kernel_size=(k, k), padding=padding, name="conv6c2")(conv6)
    conv6 = BatchNormalization(scale=False, axis=3, name="conv6b2")(conv6)
    conv6 = Activation('relu', name="conv6a2")(conv6)

    gap = GlobalAveragePooling2D(name="gap")(conv6)
    outputs = Dense(1, activation='sigmoid', name="output")(gap)

    d = Model(inputs, outputs, name=name)

    def d_loss(y_true, y_pred):
        L = losses.binary_crossentropy(K.batch_flatten(y_true),
                                       K.batch_flatten(y_pred))
        #         L = losses.mean_squared_error(K.batch_flatten(y_true),
        #                                            K.batch_flatten(y_pred))
        return L

    d.compile(optimizer=Adam(lr=init_lr, beta_1=0.5), loss=d_loss, metrics=['accuracy'])

    return d, d.layers[-1].output_shape[1:]

def discriminator_image3(img_size, n_filters, init_lr, name='d'):
    """
    discriminator network (patch GAN)
      stride 2 conv X 2
        max pooling X 4
    fully connected X 1
    """

    # set image specifics
    k = 3  # kernel size
    s = 2  # stride
    img_ch = 3  # image channels
    out_ch = 1  # output channel
    img_height, img_width = img_size[0], img_size[1]
    padding = 'same'  # 'valid'

    inputs = Input((img_height, img_width, img_ch + out_ch), name="inputs3")

    conv2 = Conv2D(2 * n_filters, kernel_size=(k, k), padding=padding, name="conv2c")(inputs)
    conv2 = BatchNormalization(scale=False, axis=3, name="conv2b")(conv2)
    conv2 = Activation('relu', name="conv2a")(conv2)
    conv2 = Conv2D(2 * n_filters, kernel_size=(k, k), padding=padding, name="conv2c2")(conv2)
    conv2 = BatchNormalization(scale=False, axis=3, name="conv2b2")(conv2)
    conv2 = Activation('relu', name="conv2a2")(conv2)
    pool2 = MaxPooling2D(pool_size=(s, s), name="conv2m")(conv2)

    conv3 = Conv2D(4 * n_filters, kernel_size=(k, k), padding=padding, name="conv3c")(pool2)
    conv3 = BatchNormalization(scale=False, axis=3, name="conv3b")(conv3)
    conv3 = Activation('relu', name="conv3a")(conv3)
    conv3 = Conv2D(4 * n_filters, kernel_size=(k, k), padding=padding, name="conv3c2")(conv3)
    conv3 = BatchNormalization(scale=False, axis=3, name="conv3b2")(conv3)
    conv3 = Activation('relu', name="conv3a2")(conv3)
    pool3 = MaxPooling2D(pool_size=(s, s), name="conv3m")(conv3)

    conv4 = Conv2D(8 * n_filters, kernel_size=(k, k), padding=padding, name="conv4c")(pool3)
    conv4 = BatchNormalization(scale=False, axis=3, name="conv4b")(conv4)
    conv4 = Activation('relu', name="conv4a")(conv4)
    conv4 = Conv2D(8 * n_filters, kernel_size=(k, k), padding=padding, name="conv4c2")(conv4)
    conv4 = BatchNormalization(scale=False, axis=3, name="conv4b2")(conv4)
    conv4 = Activation('relu', name="conv4a2")(conv4)
    pool4 = MaxPooling2D(pool_size=(s, s), name="conv4m")(conv4)

    conv5 = Conv2D(16 * n_filters, kernel_size=(k, k), strides=(s, s), padding=padding, name="conv5c")(pool4)
    conv5 = BatchNormalization(scale=False, axis=3, name="conv5b")(conv5)
    conv5 = Activation('relu', name="conv5a")(conv5)
    conv5 = Conv2D(16 * n_filters, kernel_size=(k, k), padding=padding, name="conv5c2")(conv5)
    conv5 = BatchNormalization(scale=False, axis=3, name="conv5b2")(conv5)
    conv5 = Activation('relu', name="conv5a2")(conv5)

    conv6 = Conv2D(16 * n_filters, kernel_size=(k, k), strides=(s, s), padding=padding, name="conv6c")(conv5)
    conv6 = BatchNormalization(scale=False, axis=3, name="conv6b")(conv6)
    conv6 = Activation('relu', name="conv6a")(conv6)
    conv6 = Conv2D(16 * n_filters, kernel_size=(k, k), padding=padding, name="conv6c2")(conv6)
    conv6 = BatchNormalization(scale=False, axis=3, name="conv6b2")(conv6)
    conv6 = Activation('relu', name="conv6a2")(conv6)

    gap = GlobalAveragePooling2D(name="gap")(conv6)
    outputs = Dense(1, activation='sigmoid', name="output")(gap)

    d = Model(inputs, outputs, name=name)

    def d_loss(y_true, y_pred):
        L = losses.binary_crossentropy(K.batch_flatten(y_true),
                                       K.batch_flatten(y_pred))
        #         L = losses.mean_squared_error(K.batch_flatten(y_true),
        #                                            K.batch_flatten(y_pred))
        return L

    d.compile(optimizer=Adam(lr=init_lr, beta_1=0.5), loss=d_loss, metrics=['accuracy'])

    return d, d.layers[-1].output_shape[1:]


def discriminator_image4(img_size, n_filters, init_lr, name='d'):
    """
    discriminator network (patch GAN)
      stride 2 conv X 2
        max pooling X 4
    fully connected X 1
    """

    # set image specifics
    k = 3  # kernel size
    s = 2  # stride
    img_ch = 3  # image channels
    out_ch = 1  # output channel
    img_height, img_width = img_size[0], img_size[1]
    padding = 'same'  # 'valid'

    inputs = Input((img_height, img_width, img_ch + out_ch), name="inputs4")

    conv1 = Conv2D(n_filters, kernel_size=(k, k), padding=padding, name="conv1c")(inputs)
    conv1 = BatchNormalization(scale=False, axis=3, name="conv1b")(conv1)
    conv1 = Activation('relu', name="conv1a")(conv1)
    conv1 = Conv2D(n_filters, kernel_size=(k, k), padding=padding, name="conv1c2")(conv1)
    conv1 = BatchNormalization(scale=False, axis=3, name="conv1b2")(conv1)
    conv1 = Activation('relu', name="conv1a2")(conv1)
    pool1 = MaxPooling2D(pool_size=(s, s), name="conv1m")(conv1)

    conv2 = Conv2D(2 * n_filters, kernel_size=(k, k), padding=padding, name="conv2c")(pool1)
    conv2 = BatchNormalization(scale=False, axis=3, name="conv2b")(conv2)
    conv2 = Activation('relu', name="conv2a")(conv2)
    conv2 = Conv2D(2 * n_filters, kernel_size=(k, k), padding=padding, name="conv2c2")(conv2)
    conv2 = BatchNormalization(scale=False, axis=3, name="conv2b2")(conv2)
    conv2 = Activation('relu', name="conv2a2")(conv2)
    pool2 = MaxPooling2D(pool_size=(s, s), name="conv2m")(conv2)

    conv3 = Conv2D(4 * n_filters, kernel_size=(k, k), padding=padding, name="conv3c")(pool2)
    conv3 = BatchNormalization(scale=False, axis=3, name="conv3b")(conv3)
    conv3 = Activation('relu', name="conv3a")(conv3)
    conv3 = Conv2D(4 * n_filters, kernel_size=(k, k), padding=padding, name="conv3c2")(conv3)
    conv3 = BatchNormalization(scale=False, axis=3, name="conv3b2")(conv3)
    conv3 = Activation('relu', name="conv3a2")(conv3)
    pool3 = MaxPooling2D(pool_size=(s, s), name="conv3m")(conv3)

    conv4 = Conv2D(8 * n_filters, kernel_size=(k, k), padding=padding, name="conv4c")(pool3)
    conv4 = BatchNormalization(scale=False, axis=3, name="conv4b")(conv4)
    conv4 = Activation('relu', name="conv4a")(conv4)
    conv4 = Conv2D(8 * n_filters, kernel_size=(k, k), padding=padding, name="conv4c2")(conv4)
    conv4 = BatchNormalization(scale=False, axis=3, name="conv4b2")(conv4)
    conv4 = Activation('relu', name="conv4a2")(conv4)
    pool4 = MaxPooling2D(pool_size=(s, s), name="conv4m")(conv4)

    conv5 = Conv2D(16 * n_filters, kernel_size=(k, k), strides=(s, s), padding=padding, name="conv5c")(pool4)
    conv5 = BatchNormalization(scale=False, axis=3, name="conv5b")(conv5)
    conv5 = Activation('relu', name="conv5a")(conv5)
    conv5 = Conv2D(16 * n_filters, kernel_size=(k, k), padding=padding, name="conv5c2")(conv5)
    conv5 = BatchNormalization(scale=False, axis=3, name="conv5b2")(conv5)
    conv5 = Activation('relu', name="conv5a2")(conv5)

    conv6 = Conv2D(16 * n_filters, kernel_size=(k, k), strides=(s, s), padding=padding, name="conv6c")(conv5)
    conv6 = BatchNormalization(scale=False, axis=3, name="conv6b")(conv6)
    conv6 = Activation('relu', name="conv6a")(conv6)
    conv6 = Conv2D(16 * n_filters, kernel_size=(k, k), padding=padding, name="conv6c2")(conv6)
    conv6 = BatchNormalization(scale=False, axis=3, name="conv6b2")(conv6)
    conv6 = Activation('relu', name="conv6a2")(conv6)

    gap = GlobalAveragePooling2D(name="gap")(conv6)
    outputs = Dense(1, activation='sigmoid', name="output")(gap)

    d = Model(inputs, outputs, name=name)

    def d_loss(y_true, y_pred):
        L = losses.binary_crossentropy(K.batch_flatten(y_true),
                                       K.batch_flatten(y_pred))
        #         L = losses.mean_squared_error(K.batch_flatten(y_true),
        #                                            K.batch_flatten(y_pred))
        return L

    d.compile(optimizer=Adam(lr=init_lr, beta_1=0.5), loss=d_loss, metrics=['accuracy'])

    return d, d.layers[-1].output_shape[1:]


def discriminator_dummy(img_size, n_filters, init_lr, name='d'):    # naive unet without GAN
    # set image specifics
    img_ch=3 # image channels
    out_ch=1 # output channel
    img_height, img_width = img_size[0], img_size[1]

    inputs = Input((img_height, img_width, img_ch + out_ch))

    d = Model(inputs, inputs, name=name)

    def d_loss(y_true, y_pred):
        L = losses.binary_crossentropy(K.batch_flatten(y_true),
                                            K.batch_flatten(y_pred))
#         L = losses.mean_squared_error(K.batch_flatten(y_true),
#                                            K.batch_flatten(y_pred))
        return L
    
    d.compile(optimizer=Adam(lr=init_lr, beta_1=0.5), loss=d_loss, metrics=['accuracy'])
    
    return d, d.layers[-1].output_shape[1:]

def GAN(g,d,img_size,n_filters_g, n_filters_d, alpha_recip, init_lr, name='gan'):
    """
    GAN (that binds generator and discriminator)
    """
    img_h, img_w=img_size[0], img_size[1]

    img_ch=3
    seg_ch=1
    
    fundus = Input((img_h, img_w, img_ch))
    vessel = Input((img_h, img_w, seg_ch))
    
    fake_vessel=g(fundus)
    fake_pair=Concatenate(axis=3)([fundus, fake_vessel])
    
    gan=Model([fundus, vessel], d(fake_pair), name=name)

    def gan_loss(y_true, y_pred):
        y_true_flat = K.batch_flatten(y_true)
        y_pred_flat = K.batch_flatten(y_pred)

        L_adv = losses.binary_crossentropy(y_true_flat, y_pred_flat)
#         L_adv = losses.mean_squared_error(y_true_flat, y_pred_flat)

        vessel_flat = K.batch_flatten(vessel)
        fake_vessel_flat = K.batch_flatten(fake_vessel)
        L_seg = losses.binary_crossentropy(vessel_flat, fake_vessel_flat)
#         L_seg = losses.mean_absolute_error(vessel_flat, fake_vessel_flat)

        return alpha_recip*L_adv + L_seg
    
    
    gan.compile(optimizer=Adam(lr=init_lr, beta_1=0.5), loss=gan_loss, metrics=['accuracy'])
        
    return gan
    
def pretrain_g(g, img_size, n_filters_g, init_lr):
    img_h, img_w=img_size[0], img_size[1]

    img_ch=3
    fundus = Input((img_h, img_w, img_ch))
    generator=Model(fundus, g(fundus))

    def g_loss(y_true, y_pred):
        L_seg = losses.binary_crossentropy(K.batch_flatten(y_true), K.batch_flatten(y_pred))
        return L_seg
    
    
    generator.compile(optimizer=Adam(lr=init_lr, beta_1=0.5), loss=g_loss, metrics=['accuracy'])
        
    return generator