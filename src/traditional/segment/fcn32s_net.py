# fcn 32s
# 直接从 pool5 上采样（反卷级）到原尺寸
# 5 layers
from keras.models import Model
from keras.layers import MaxPooling2D, Dropout
from keras.layers import Input, Activation
from dpl.utils import conv2d_bn, deconv2d_bn


def create_fcn32s(input_size=(512, 512, 3)):
    # model = Sequential()
    # preprocessing
    input_layer = Input(shape=(input_size))
    #x = ZeroPadding2D(100)(input)
    # layer1 2conv  1/2
    x = conv2d_bn(input_layer, 64, (3, 3))
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

    # normalize to length == class_nums
    drop_2_n = conv2d_bn(drop_2, 21, (1, 1))

    # 上采样32倍
    deconv_1 = deconv2d_bn(drop_2_n, 21, size=(64, 64), strides=(32, 32),
                           output_shape=input_layer.shape)

    output = Activation('softmax')(deconv_1)

    fcn32s_model = Model(input_layer, output, name='fcn32s_net')
    return fcn32s_model


from config import *
from test_train_data import *


def main(argv):
    config = Config()
    config.batch_size = 10
    config.steps_per_epoch = 100
    config.validation_steps = 100
    config.epochs = 1
    config.image_min_dims = 256
    config.image_max_dims = 256
    model = create_fcn32s((config.image_min_dims, config.image_min_dims, 3))
    train_data(model, config, argv[1])
    #print(model.summary())

import sys
if __name__ == "__main__":
    main(sys.argv)