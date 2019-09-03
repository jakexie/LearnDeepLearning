# u-net
import keras
from keras.models import Model
from keras.layers import Conv2D, Dense, MaxPooling2D, UpSampling2D
from keras.layers import Activation, Input, BatchNormalization


def conv2d_bn(input, nums_kernal, size, strides=1, padding = 'same'):
    x = Conv2D(nums_kernal, size, padding=padding, strides=strides)(input)
    x = BatchNormalization()(x)
    return Activation('relu')(x)


def upsample_contract(input_1, input_2, nums_kernel, kernel_size=(3,3)):
    up_ratio = (int(input_2.shape[1].value/input_1.shape[1].value),
                int(input_2.shape[2].value/input_1.shape[2].value))
    upsample = UpSampling2D(up_ratio)(input_1)
    conv = conv2d_bn(upsample, nums_kernel, kernel_size)
    concat = keras.layers.concatenate([conv, input_2])
    return concat


def create_unet(input_shape=(512,512,3)):
    input = Input(input_shape)
    # layer 1
    conv_1a = conv2d_bn(input, 64, (3,3))
    conv_1b = conv2d_bn(conv_1a, 64, (3,3))
    pooling_1 = MaxPooling2D()(conv_1b)
    # layer 2
    conv_2a = conv2d_bn(pooling_1, 128, (3,3))
    conv_2b = conv2d_bn(conv_2a, 128, (3,3))
    pooling_2 = MaxPooling2D()(conv_2b)
    # layer 3
    conv_3a = conv2d_bn(pooling_2, 256, (3,3))
    conv_3b = conv2d_bn(conv_3a, 256, (3,3))
    pooling_3 = MaxPooling2D()(conv_3b)
    # layer 4
    conv_4a = conv2d_bn(pooling_3, 512, (3,3))
    conv_4b = conv2d_bn(conv_4a, 512, (3,3))
    pooling_4 = MaxPooling2D()(conv_4b)
    # layer 5
    conv_5a = conv2d_bn(pooling_4, 1024, (3,3))
    conv_5b = conv2d_bn(conv_5a, 1024, (3,3))
    #pooling_5 = MaxPooling2D()(conv_5b)

    # inv_layer 4
    de_conv_4a = upsample_contract(conv_5b, conv_4b, 512)
    de_conv_4b = conv2d_bn(de_conv_4a, 512, (3,3))
    de_conv_4c = conv2d_bn(de_conv_4b, 512, (3,3))

    # inv_layer 3
    de_conv_3a = upsample_contract(de_conv_4c, conv_3b, 256)
    de_conv_3b = conv2d_bn(de_conv_3a, 256, (3,3))
    de_conv_3c = conv2d_bn(de_conv_3b, 256, (3,3))

    # inv_layer 2
    de_conv_2a = upsample_contract(de_conv_3c, conv_2b, 128)
    de_conv_2b = conv2d_bn(de_conv_2a, 128, (3,3))
    de_conv_2c = conv2d_bn(de_conv_2b, 128, (3,3))

    # inv_layer 1
    de_conv_1a = upsample_contract(de_conv_2c, conv_1b, 128)
    de_conv_1b = conv2d_bn(de_conv_1a, 128, (3,3))
    de_conv_1c = conv2d_bn(de_conv_1b, 128, (3,3))

    # output
    output = conv2d_bn(de_conv_1c, 21, (1,1))

    unet_model = Model(input, output, name='u-net')
    return unet_model

from config import *
from test_train_data import *


def main(argv):
    config = Config()
    config.batch_size = 1
    config.steps_per_epoch = 500
    config.validation_steps = 100
    config.epochs = 10
    config.image_min_dims = 256
    config.image_max_dims = 256
    model = create_unet((config.image_min_dims, config.image_min_dims, 3))
    train_data(model, config, argv[1])

import sys
if __name__ == "__main__":
    main(sys.argv)
