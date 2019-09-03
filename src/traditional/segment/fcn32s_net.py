# fcn 32s
# 直接从 pool5 上采样（反卷级）到原尺寸
# 5 layers
from keras.models import Model
from keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, Dropout, UpSampling2D
from keras.layers import Input, ZeroPadding2D, BatchNormalization, Activation


def conv2d_bn(input, nums_kernal, size, strides=1, padding = 'same'):
    x = Conv2D(nums_kernal, size, padding=padding, strides=strides)(input)
    #x = BatchNormalization()(x)
    return Activation('relu')(x)


def create_fcn32(input_size=(512, 512, 3)):
    # model = Sequential()
    # preprocessing
    input = Input(shape=(input_size))
    #x = ZeroPadding2D(100)(input)
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

    # 上采样32倍
    drop_2_n = conv2d_bn(drop_1, 21, (1, 1))

    ratio = (int(input.shape[1].value/drop_2_n.shape[1].value), int(input.shape[2].value/drop_2_n.shape[2].value))
    #print(ratio)
    bilinear_inter_1 = UpSampling2D(ratio)(drop_2_n)
    deconv_1 = conv2d_bn(bilinear_inter_1, 21, (3, 3))

    output=Activation('softmax')(deconv_1)

    fcn32s_model = Model(input, output, name='fcn32s_net')
    return fcn32s_model


from dpl.data.pascal.voc_data_generator_v2 import *
from dpl.data.pascal.voc_data_generator import *
import yaml
import keras
import sys


def main(argv):
    optimizer = keras.optimizers.Adam(1e-4)
    model = create_fcn32((224, 224, 3))
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    config = Config()
    config.batch_size = 1
    config.steps_per_epoch = 500
    config.validation_steps = 100
    config.epochs = 1
    config.image_min_dims = 224
    config.image_max_dims = 224

    if 1:
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
        datagen = PascalVocGenerator(image_shape=[config.image_min_dims, config.image_min_dims, 3],
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