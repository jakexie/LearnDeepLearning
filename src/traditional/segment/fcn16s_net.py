# fcn 16s
# 直接从 pool4 上采样（反卷级）到原尺寸
# 5 layers
from keras.models import Model
from keras.layers import MaxPooling2D, Dropout
from keras.layers import Input, Activation
from dpl.utils import conv2d_bn, deconv2d_bn


def create_fcn16s(input_size=(512, 512, 3)):
    input_layer = Input(shape=(input_size))
    # x = ZeroPadding2D(100)(input)
    # block1 2conv  1/2
    x = conv2d_bn(input_layer, 64, (3, 3), name='block1_conv1')
    x = conv2d_bn(x, 64, (3, 3), name='block1_conv2')
    pool1 = MaxPooling2D(name='block1_pool')(x)

    # block2 2conv 1/4
    x = conv2d_bn(pool1, 128, (3, 3), name='block2_conv1')
    x = conv2d_bn(x, 128, (3, 3), name='block2_conv2')
    pool2 = MaxPooling2D(name='block2_pool')(x)

    # block3 3conv 1/8
    x = conv2d_bn(pool2, 256, (3, 3), name='block3_conv1')
    x = conv2d_bn(x, 256, (3, 3), name='block3_conv2')
    x = conv2d_bn(x, 256, (3, 3), name='block3_conv3')
    pool3 = MaxPooling2D(name='block3_pool')(x)

    # block4 3conv 1/16
    x = conv2d_bn(pool3, 512, (3, 3), name='block4_conv1')
    x = conv2d_bn(x, 512, (3, 3), name='block4_conv2')
    x = conv2d_bn(x, 512, (3, 3), name='block4_conv3')
    pool4 = MaxPooling2D(name='block4_pool')(x)

    # block5 3conv 1/32
    x = conv2d_bn(pool4, 512, (3, 3), name='block5_conv1')
    x = conv2d_bn(x, 512, (3, 3), name='block5_conv2')
    x = conv2d_bn(x, 512, (3, 3), name='block5_conv3')
    pool5 = MaxPooling2D(name='block5_pool')(x)

    # block6 2full
    full_1 = conv2d_bn(pool5, 4096, (7, 7), name='fc1')
    drop_1 = Dropout(0.5)(full_1)
    full_2 = conv2d_bn(drop_1, 4096, (1, 1), name='fc2')
    drop_2 = Dropout(0.5)(full_2)

    # 上采样2倍 = pool4 size
    drop_2_u = conv2d_bn(drop_2, 21, (1, 1))

    deconv_1 = deconv2d_bn(drop_2_u, 21, output_shape=pool4.shape)

    # merge(+)
    pool4_u = conv2d_bn(pool4, 21, (1, 1))
    merge_1 = keras.layers.add([deconv_1, pool4_u])

    # upsample 16
    deconv_2 = deconv2d_bn(merge_1, 21, size=32, strides=16, output_shape=input_layer.shape)

    output = Activation('softmax')(deconv_2)

    fcn16s_model = Model(input_layer, output, name='fcn16s_net')
    return fcn16s_model

from config import *
from test_train_data import *


def main(argv):
    config = Config()
    config.batch_size = 20
    config.steps_per_epoch = 70
    config.validation_steps = 10
    config.epochs = 176
    config.image_min_dims = 224
    config.image_max_dims = 224
    model = create_fcn16s((config.image_min_dims, config.image_min_dims, 3))
    pretrained_path = "./pretrained_weights/vgg16_weights_transfered2fcn.h5"
    model.load_weights(pretrained_path, by_name=True)
    train_data(model, config, argv[1])

import sys
if __name__ == "__main__":
    main(sys.argv)