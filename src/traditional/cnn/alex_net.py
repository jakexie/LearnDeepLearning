########## Alexnet 2012 ###################
# 8 layers 5 conv + 3 maxpool + 2 full + 1 output
# 1-2 layers: conv(11*11, 5*5) + relu + normal + maxpool
# 3-4 layers: conv(3*3, 3*3) + relu
# 5 layers: conv + relu + maxpool
# 6 layers: full(4096) + relu
# 7 layers: full(4096)
# 8 layers: full(1000 output)
# 9 layers: full(10) 为mnist数据额外添加（因为mnist手写数据有10个类别）
import keras
from keras.layers import Conv2D, Dense, MaxPool2D, BatchNormalization, Dropout, Flatten
from keras.models import Sequential


def create_alex_net(input_shape=(227,227,3), num_classes=10):
    model = Sequential()
    # 1
    # conv1
    # output dim = (227-11)/4 + 1 = 55 --> 55*55*96
    model.add(Conv2D(96, (11,11), strides=4, activation='relu', input_shape=input_shape))
    print("conv_1（input）: ", model.output_shape)
    # normalize layers
    model.add(BatchNormalization())
    # max_pool_1 27*27*96
    model.add(MaxPool2D(pool_size=(3,3), strides=2))
    print("max_pool_1: ", model.output_shape)
    # 2
    # conv2 27*27*256
    model.add(Conv2D(256, (5,5), padding='same', activation='relu'))
    print("conv_2: ", model.output_shape)
    # normalize layers
    model.add(BatchNormalization())
    # max_pool_2 13*13*256
    model.add(MaxPool2D(pool_size=(3,3), strides=2))
    print("max_pool_2: ", model.output_shape)
    # 3
    # conv3 13*13*384
    model.add(Conv2D(384, (3,3), padding='same', activation='relu'))
    print("conv_3: ", model.output_shape)
    # 4
    # conv4 13*13*384
    model.add(Conv2D(384, (3,3), padding='same', activation='relu'))
    print("conv_4: ", model.output_shape)
    # 5
    # conv5 13*13*256
    model.add(Conv2D(256, (3,3), padding='same', activation='relu'))
    print("conv_5: ", model.output_shape)
    # max_pool_3 6*6*256
    model.add(MaxPool2D(pool_size=(3,3), strides=2))
    print("max_pool_3", model.output_shape)

    # 6
    # full_1
    model.add(Conv2D(4096, (6,6), activation='relu'))
    print("full_1: ", model.output_shape)
    model.add(Dropout(0.5))
    model.add(Flatten())
    print("flatten: ", model.output_shape)
    # 7
    # full_2
    model.add(Dense(4096, activation='relu'))
    print("full_2: ", model.output_shape)
    model.add(Dropout(0.5))
    # 8
    # full_3
    #model.add(Dense(1000, activation='softmax'))
    # change for apply mnist
    model.add(Dense(1000, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    print("full_3(output): ", model.output_shape)
    model.name = 'alex-net'
    return model


def main():
    model = create_alex_net((227, 227, 3), 10)
    # !使用adam 会导致结果不收敛
    model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()


if __name__ == '__main__':
    main()