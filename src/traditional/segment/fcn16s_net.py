# fcn 16s
# 直接从 pool4 上采样（反卷级）到原尺寸
# 5 layers
from keras.models import Model, Sequential
from keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, Dropout, UpSampling2D
from keras.layers import Input, ZeroPadding2D, BatchNormalization, Activation


def conv2d_bn(input, nums_kernal, size, strides=1, padding='same'):
    x = Conv2D(nums_kernal, size, padding=padding, strides=strides)(input)
    x = BatchNormalization()(x)
    return Activation('relu')(x)


def create_fcn16s(input_size=(512, 512, 3)):
    # model = Sequential()
    # preprocessing
    input = Input(shape=(input_size))
    # x = ZeroPadding2D(100)(input)
    # layer1 2conv  1/2
    x = conv2d_bn(input, 64, (3, 3))
    x = conv2d_bn(x, 64, (3, 3))
    pool1 = MaxPooling2D()(x)

    # layer2 2conv 1/4
    x = conv2d_bn(pool1, 128, (3, 3))
    x = conv2d_bn(x, 128, (3, 3))
    pool2 = MaxPooling2D()(x)

    # layer3 3conv 1/8
    x = conv2d_bn(pool2, 256, (3, 3))
    x = conv2d_bn(x, 256, (3, 3))
    x = conv2d_bn(x, 256, (3, 3))
    pool3 = MaxPooling2D()(x)

    # layer4 3conv 1/16
    x = conv2d_bn(pool3, 512, (3, 3))
    x = conv2d_bn(x, 512, (3, 3))
    x = conv2d_bn(x, 512, (3, 3))
    pool4 = MaxPooling2D()(x)

    # layer5 3conv 1/32
    x = conv2d_bn(pool4, 512, (3, 3))
    x = conv2d_bn(x, 512, (3, 3))
    x = conv2d_bn(x, 512, (3, 3))
    pool5 = MaxPooling2D()(x)

    # layer6 2full
    full_1 = conv2d_bn(pool5, 4096, (7, 7))
    drop_1 = Dropout(0.5)(full_1)
    full_2 = conv2d_bn(drop_1, 4096, (1, 1))
    drop_2 = Dropout(0.5)(full_2)

    # 上采样2倍 = pool4 size
    drop_2_u = conv2d_bn(drop_2, 21, (1, 1))
    # deconv_1 = Conv2DTranspose(21, (64,64), strides=32)(drop_2_n)

    ratio_1 = (int(pool4.shape[1].value / drop_2_u.shape[1].value), int(pool4.shape[2].value / drop_2_u.shape[2].value))
    print("ratio_1: ", ratio_1)
    bilinear_inter_1 = UpSampling2D(ratio_1)(drop_2_u)
    deconv_1 = conv2d_bn(bilinear_inter_1, 21, (1, 1))

    # merge(+)
    pool4_u = conv2d_bn(pool4, 21, (1, 1))
    merge_1 = keras.layers.add([deconv_1, pool4_u])

    # upsample 16
    ratio_2 = (int(input.shape[1].value / merge_1.shape[1].value), int(input.shape[2].value / merge_1.shape[2].value))
    print("ratio_2: ", ratio_2)
    bilinear_inter_2 = UpSampling2D(ratio_2)(merge_1)
    deconv_2 = conv2d_bn(bilinear_inter_2, 21, (1, 1))

    output = Activation('softmax')(deconv_2)

    fcn16s_model = Model(input, output, name='fcn16s_net')
    return fcn16s_model

from config import *
from test_train_data import *


def main(argv):
    config = Config()
    config.batch_size = 10
    config.steps_per_epoch = 100
    config.validation_steps = 100
    config.epochs = 10
    config.image_min_dims = 256
    config.image_max_dims = 256
    model = create_fcn16s((config.image_min_dims, config.image_min_dims, 3))
    train_data(model, config, argv[1])

import sys
if __name__ == "__main__":
    main(sys.argv)