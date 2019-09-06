# google net 2014
# 训练时需要副分类器， 预测时无视
import keras
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Input, AveragePooling2D, Dense, Flatten
from keras.layers import BatchNormalization, Input, Dropout, Activation


def conv2d_bn(input, nums_kernal, size, strides=1, padding='same'):
    x = Conv2D(nums_kernal, size, padding=padding, strides=strides)(input)
    x = BatchNormalization()(x)
    return Activation('relu')(x)


def getMaxPool(input):
    return MaxPooling2D(pool_size=(3, 3), strides=1, padding='same')(input)

# conv strides=1 padding=same
# maxpool size(3,3) strides=2 padding='same'
# nums --> nums of filter for special size(1x1, 3x3, 5x5, pool)
def Inception(input, nums_11, nums_11_33, nums_33, nums_11_55, nums_55, nums_11_pool):
    conv_11 = conv2d_bn(input, nums_11, (1, 1))

    conv_11_33 = conv2d_bn(input, nums_11_33, (1, 1))
    conv_33 = conv2d_bn(conv_11_33, nums_33, (3, 3))

    conv_11_55 = conv2d_bn(input, nums_11_55, (1, 1))
    conv_55 = conv2d_bn(conv_11_55, nums_55, (5, 5))

    conv_max_pool = getMaxPool(input)
    conv_max_pool_11 = conv2d_bn(conv_max_pool, nums_11_pool, (1, 1))

    output = keras.layers.concatenate([conv_11, conv_33, conv_55, conv_max_pool_11], axis=-1)
    return output


def AuxiliaryClassifier(input):
    x = AveragePooling2D(pool_size=(5, 5), strides=3)(input)
    x = conv2d_bn(x, 128, (1, 1))
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.7)(x)
    # output = Dense(1000, activation='softmax')(x)
    # add 1 full layer for mnist
    x = Dense(1000, activation='relu')(x)
    x = Dropout(0.7)(x)
    output = Dense(10, activation='softmax')(x)
    return output


def MainClassifier(input):
    x = AveragePooling2D(pool_size=(7, 7), strides=1)(input)
    x = Dropout(0.4)(x)
    x = Flatten()(x)
    # output = Dense(1000, activation='softmax')(x)
    # add 1 full layer for mnist
    x = Dense(1000, activation='relu')(x)
    x = Dropout(0.4)(x)
    output = Dense(10, activation='softmax')(x)
    return output

# create google net inception v1
# 仅实现了主分类器
def create_google_net(input_shape):
    input_layer = Input(shape=input_shape)
    x = Conv2D(64, (7, 7), strides=2, padding='same', activation='relu')(input_layer)
    # x = conv2d_bn(input, 64, size=(7,7), strides=2)
    x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)
    x = BatchNormalization()(x)

    x = Conv2D(64, (1, 1), activation='relu')(x)
    x = Conv2D(192, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)

    # inception_3a
    x = Inception(x, 64, 96, 128, 16, 32, 32)
    # inception_3b
    x = Inception(x, 128, 128, 192, 32, 96, 64)

    x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)

    # inception_4a
    x = Inception(x, 192, 96, 208, 16, 48, 64)
    # Auxiliary Classifier 0
    softmax0 = AuxiliaryClassifier(x)
    # inception_4b
    x = Inception(x, 160, 112, 224, 24, 64, 64)
    # inception_4c
    x = Inception(x, 128, 128, 256, 24, 64, 64)
    # inception_4d
    x = Inception(x, 112, 114, 288, 32, 64, 64)
    # Auxiliary Classifier 1
    softmax1 = AuxiliaryClassifier(x)
    # inception_4e
    x = Inception(x, 256, 160, 320, 32, 128, 128)

    x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)

    # inception_5a
    x = Inception(x, 256, 160, 320, 32, 128, 128)
    # inception_5b
    x = Inception(x, 384, 192, 384, 48, 128, 128)

    # Main classifier
    softmax2 = MainClassifier(x)

    # [softmax0, softmax1, softmax2]
    model = Model(input_layer, outputs=softmax2, name="googLenet")
    return model


def main():
    # model.build(input)
    input_shape = (224, 224, 3)
    model = create_google_net(input_shape)
    model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    #model.get_config()
    print(model.summary())


if __name__ == '__main__':
    main()