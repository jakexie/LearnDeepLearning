# vgg 16
from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPool2D, BatchNormalization, Dropout, Flatten


def create_vgg16_net():
    model = Sequential()
    # 64
    model.add(Conv2D(64, (3,3), activation='relu', padding='same', input_shape=(224,224,3)))
    print("conv_1: ", model.output_shape)
    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    print("conv_2: ", model.output_shape)
    model.add(MaxPool2D())
    print("max_pool_1: ", model.output_shape)

    #128
    model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    print("conv_3: ", model.output_shape)
    model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    print("conv_4: ", model.output_shape)
    model.add(MaxPool2D())
    print("max_pool_2: ", model.output_shape)

    #256
    model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
    print("conv_5: ", model.output_shape)
    model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
    print("conv_6: ", model.output_shape)
    model.add(Conv2D(256, (1,1), activation='relu', padding='same'))
    print("conv_7: ", model.output_shape)
    model.add(MaxPool2D())
    print("max_pool_3: ", model.output_shape)

    #512
    model.add(Conv2D(512, (3,3), activation='relu', padding='same'))
    print("conv_8: ", model.output_shape)
    model.add(Conv2D(512, (3,3), activation='relu', padding='same'))
    print("conv_9: ", model.output_shape)
    model.add(Conv2D(512, (1,1), activation='relu', padding='same'))
    print("conv_10: ", model.output_shape)
    model.add(MaxPool2D())
    print("max_pool_4: ", model.output_shape)

    #512
    model.add(Conv2D(512, (3,3), activation='relu', padding='same'))
    print("conv_11: ", model.output_shape)
    model.add(Conv2D(512, (3,3), activation='relu', padding='same'))
    print("conv_12: ", model.output_shape)
    model.add(Conv2D(512, (1,1), activation='relu', padding='same'))
    print("conv_13: ", model.output_shape)
    model.add(MaxPool2D())
    print("max_pool_5: ", model.output_shape)

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    print("full_1: ", model.output_shape)
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    print("full_2: ", model.output_shape)
    model.add(Dense(1000, activation='relu'))
    model.add(Dropout(0.5))
    print("full_3: ", model.output_shape)
    model.add(Dense(10, activation='softmax'))
    print("output: ", model.output_shape)

    return model


def main():
    model = create_vgg16_net()
    model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())


if __name__ == '__main__':
    main()

