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


def create_fcn8s(input_size=(512, 512, 3)):
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
    drop_2_n = conv2d_bn(drop_2, 21, (1, 1))

    ratio_1 = (int(pool4.shape[1].value / drop_2_n.shape[1].value), int(pool4.shape[2].value / drop_2_n.shape[2].value))
    print("ratio_1: ", ratio_1)
    bilinear_inter_1 = UpSampling2D(ratio_1)(drop_2_n)
    deconv_1 = conv2d_bn(bilinear_inter_1, 21, (1, 1))

    # merge(+)
    pool4_u = conv2d_bn(pool4, 21, (1, 1))
    merge_1 = keras.layers.add([deconv_1, pool4_u])

    # upsample to merge pool3
    ratio_2 = (int(pool3.shape[1].value / merge_1.shape[1].value), int(pool3.shape[2].value / merge_1.shape[2].value))
    print("ratio_2: ", ratio_2)
    bilinear_inter_2 = UpSampling2D(ratio_2)(merge_1)
    deconv_2 = conv2d_bn(bilinear_inter_2, 21, (1, 1))

    # merge
    pool3_u = conv2d_bn(pool3, 21, (1, 1))
    merge_2 = keras.layers.add([deconv_2, pool3_u])

    # upsample 8
    ratio_3 = (int(input.shape[1].value / merge_2.shape[1].value), int(input.shape[2].value / merge_2.shape[2].value))
    print("ratio_3: ", ratio_3)
    bilinear_inter_3 = UpSampling2D(ratio_3)(merge_2)
    deconv_3 = conv2d_bn(bilinear_inter_3, 21, (1, 1))

    output = Activation('softmax')(deconv_3)

    fcn8s_model = Model(input, output, name='fcn8s_net')
    return fcn8s_model


from dpl.data.pascal.voc_data_generator_v2 import *
from dpl.data.pascal.voc_data_generator import *
import yaml
import keras
import sys


def main(argv):
    config = Config()
    config.batch_size = 1
    config.steps_per_epoch = 500
    config.validation_steps = 100
    config.epochs = 1
    config.image_min_dims = 256
    config.image_max_dims = 256

    optimizer = keras.optimizers.Adam(1e-4)
    model = create_fcn8s((config.image_min_dims, config.image_max_dims, 3))
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())

    if 0:
        train_dataset = PascalDataset(argv[1], is_train=True)
        val_dataset = PascalDataset(argv[1], is_train=False)

        train_generator = data_generator(train_dataset, config)
        val_generator = data_generator(val_dataset, config)
    else:
        with open("../init_args.yml", 'r') as stream:
            try:
                init_args = yaml.load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        target_shape = (config.image_min_dims, config.image_min_dims)
        datagen = PascalVocGenerator(image_shape=[target_shape[0], target_shape[1], 3],
                                     image_resample=True,
                                     pixelwise_center=True,
                                     pixel_mean=[115.85100, 110.50989, 102.16182],
                                     pixelwise_std_normalization=True,
                                     pixel_std=[70.30930, 69.41244, 72.60676])

        train_loader = ImageSetLoader(**init_args['image_set_loader']['train'])
        val_loader = ImageSetLoader(**init_args['image_set_loader']['val'])

        train_generator = datagen.flow_from_imageset(
                class_mode='categorical',
                classes=21,
                batch_size=5,
                shuffle=True,
                image_set_loader=train_loader)
        val_generator = datagen.flow_from_imageset(
                class_mode='categorical',
                classes=21,
                batch_size=5,
                shuffle=True,
                image_set_loader=val_loader)

    workers = 1  # multiprocessing.cpu_count()

    model.fit_generator(train_generator, steps_per_epoch=config.steps_per_epoch, epochs=config.epochs,
                        use_multiprocessing=False, max_queue_size=100, workers=workers
                        )
    #                     validation_data=val_generator, validation_steps=config.validation_steps
    scores = model.evaluate_generator(val_generator, steps=10)
    print("loss: ", scores[0])
    print("acc: ", scores[1])


if __name__ == "__main__":
    main(sys.argv)