# resnet 2015
import keras
from keras.models import Model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, AveragePooling2D, Dropout
from keras.layers import BatchNormalization, Input, Activation


def conv2d_bn(input, nums_kernal, size, strides=1, padding='same'):
    x = Conv2D(nums_kernal, size, padding=padding, strides=strides)(input)
    x = BatchNormalization()(x)
    return Activation('relu')(x)


# 匹配输入尺度和输出尺度
# 每个convn_x 第一层调用
def conv_block34(input, filter_nums1, filter_nums2, strides=2):
    # branch_normal
    x = conv2d_bn(input, filter_nums1, (3, 3), strides=strides)
    x = Conv2D(filter_nums2, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    # branch_shortcut
    short_cut = Conv2D(filter_nums2, (1, 1), strides=strides)(input)
    short_cut = BatchNormalization()(short_cut)
    # add
    x = keras.layers.add([x, short_cut])
    return Activation('relu')(x)


# identity mapping
# 紧接conv_block34 调用
def identity_block34(input, filter_nums1, filter_nums2):
    # branch_normal
    x = conv2d_bn(input, filter_nums1, (3, 3))
    x = Conv2D(filter_nums2, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    # branch_shortcut
    short_cut = BatchNormalization()(input)

    x = keras.layers.add([x, short_cut])
    return Activation('relu')(x)


# 匹配输入尺度和输出尺度
# 每个convn_x 第一层调用
def conv_block(input, filter_nums1, filter_nums2, filter_nums3, strides=2):
    # branch_normal
    x = conv2d_bn(input, filter_nums1, (1, 1), strides=strides)
    x = conv2d_bn(x, filter_nums2, (3, 3))
    x = Conv2D(filter_nums3, (1, 1))(x)
    x = BatchNormalization()(x)
    # branch_shortcut
    short_cut = Conv2D(filter_nums3, (1, 1), strides=strides)(input)
    short_cut = BatchNormalization()(short_cut)
    # add
    x = keras.layers.add([x, short_cut])
    return Activation('relu')(x)


# identity mapping
# 紧接conv_block 调用
def identity_block(input, filter_nums1, filter_nums2, filter_nums3):
    # branch_normal
    x = conv2d_bn(input, filter_nums1, (1, 1))
    x = conv2d_bn(x, filter_nums2, (3, 3))
    x = Conv2D(filter_nums3, (1, 1))(x)
    x = BatchNormalization()(x)
    # branch_shortcut
    short_cut = BatchNormalization()(input)

    x = keras.layers.add([x, short_cut])
    return Activation('relu')(x)


def full_block(input):
    x = AveragePooling2D(pool_size=(7, 7))(input)
    x = Dropout(0.4)(x)
    x = Flatten()(x)
    x = Dense(1000, activation='relu')(x)
    x = Dropout(0.4)(x)
    output = Dense(10, activation='softmax')(x)
    return output


# resnet 18 34
def create_resnet_shallow(input_shape, depth):
    assert depth in [18, 34]
    conv_nums = []
    if depth == 18:
        conv_nums = [2, 2, 2, 2]
    elif depth == 34:
        conv_nums = [3, 4, 6, 3]

    # conv1 112*112
    input_layer = Input(shape=input_shape)
    x = conv2d_bn(input_layer, 64, size=(7, 7), strides=2)

    # conv2_x 56*56 3subs
    x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)
    x = conv_block34(x, 64, 64, 1)  # 1
    for i in range(conv_nums[0] - 1):
        x = identity_block34(x, 64, 64)  # 2

    # conv3_x 28*28 4subs
    x = conv_block34(x, 128, 128)  # 1
    for i in range(conv_nums[1] - 1):
        x = identity_block34(x, 128, 128)  # 3

    # conv4_x 14*14 6subs
    x = conv_block34(x, 256, 256)  # 1
    for i in range(conv_nums[2] - 1):
        x = identity_block34(x, 256, 256)  # 2

    # conv5_x 7*7 3subs
    x = conv_block34(x, 512, 512)  # 1
    for i in range(conv_nums[3] - 1):
        x = identity_block34(x, 512, 512)  # 2

    # avg_pool 1
    x = full_block(x)

    model = Model(input_layer, x, name="resnet"+str(depth))
    return model


def create_resnet_deep(input_shape, depth=101):
    assert depth in [50, 101, 152]
    conv_nums = [3, 4, 6, 3]
    if depth == 101:
        conv_nums[2] = 23
    elif depth == 152:
        conv_nums[1] = 8
        conv_nums[2] = 36

    # conv1 112*112
    input_layer = Input(shape=input_shape)
    x = conv2d_bn(input_layer, 64, size=(7, 7), strides=2)

    # conv2_x 56*56 3subs
    x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)
    x = conv_block(x, 64, 64, 256, 1)  # 1
    for i in range(conv_nums[0] - 1):
        x = identity_block(x, 64, 64, 256)  # 2

    # conv3_x 28*28 4subs
    x = conv_block(x, 128, 128, 252)  # 1
    for i in range(conv_nums[1] - 1):
        x = identity_block(x, 128, 128, 252)  # 2

    # conv4_x 14*14 6subs
    x = conv_block(x, 256, 256, 1024)  # 1
    for i in range(conv_nums[2] - 1):
        x = identity_block(x, 256, 256, 1024)  # 2

    # conv5_x 7*7 3subs
    x = conv_block(x, 512, 512, 2048)  # 1
    for i in range(conv_nums[3] - 1):
        x = identity_block(x, 512, 512, 2048)  # 2

    # avg_pool 1
    x = full_block(x)

    model = Model(input_layer, x, name="resnet"+str(depth))
    return model


def main():
    model = create_resnet_shallow((224, 224, 3), 34)
    model.compile(optimizer='sgd',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()


print(__name__)
if __name__ == '__main__':
    main()