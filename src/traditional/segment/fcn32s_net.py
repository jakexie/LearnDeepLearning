# fcn 32s
# 直接从 pool5 上采样（反卷级）到原尺寸
# 5 layers
from keras.models import Model
from keras.layers import MaxPooling2D, Dropout
from keras.layers import Input, Activation
from dpl.utils import conv2d_bn, deconv2d_bn


def create_fcn32s(input_size=(512, 512, 3), classes=21):
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

    # normalize to length == class_nums
    drop_2_n = conv2d_bn(drop_2, classes, (1, 1), name='score_fr')

    # 上采样32倍
    deconv_1 = deconv2d_bn(drop_2_n, classes, size=(64, 64), strides=(32, 32),
                           output_shape=input_layer.shape, name='upscore')

    output = Activation('softmax')(deconv_1)

    fcn32s_model = Model(input_layer, output, name='fcn32s_net')
    return fcn32s_model


from config import *
from test_train_data import *
from transfer_fcn import *


def main(argv):
    #transfer_FCN_Vgg16()
    config = Config()
    config.batch_size = 32
    config.steps_per_epoch = 265
    config.validation_steps = 23 # total 736 val data
    config.epochs = 50
    config.image_min_dims = 224
    config.image_max_dims = 224
    model = create_fcn32s((config.image_min_dims, config.image_min_dims, 3))
    #pretrained_path = "./pretrained_weights/vgg16_weights_transfered2fcn.h5"
    pretrained_path = "../check_points/fcn32s_net_weights_epoch10.hdf5"
    model.load_weights(pretrained_path, by_name=True)
    layer = model.get_layer(name='block1_conv1')
    print(layer.get_weights())
    train_data(model, config)
    # print(model.summary())

import sys
if __name__ == "__main__":
    main(sys.argv)