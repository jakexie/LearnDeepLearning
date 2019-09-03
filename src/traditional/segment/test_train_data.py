
from dpl.data.pascal.voc_data_generator_v2 import *
from dpl.data.pascal.voc_data_generator import *
import yaml
import keras


def train_data(model, config, path):

    optimizer = keras.optimizers.Adam(1e-4)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    if 0:
        train_dataset = PascalDataset(path, is_train=True)
        val_dataset = PascalDataset(path, is_train=False)

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
    scores = model.evaluate_generator(val_generator, steps=1000)
    print("loss: ", scores[0])
    print("acc: ", scores[1])

