# vgg 16
from keras.models import Model
from keras.layers import Dense, MaxPooling2D, Dropout, Flatten, Input
from dpl.utils import conv2d_bn


def create_vgg16_net(input_shape=(224, 224, 3), num_classes=1000):
    # model = Sequential()
    # preprocessing
    input_layer = Input(input_shape)
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

    x = Flatten(name='flatten')(pool5)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dropout(0.5)(x)
    x = Dense(num_classes, activation='softmax', name='predictions')(x)

    model = Model(input_layer, x, name='vgg16_net')

    return model


def main():
    model = create_vgg16_net()
    model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())


if __name__ == '__main__':
    main()

