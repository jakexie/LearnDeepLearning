
from dpl.data.pascal.voc_data_generator_v2 import *
from dpl.data.pascal.voc_data_generator import *
from dpl import utils
from keras.optimizers import SGD
import yaml
import keras
from PIL import Image

def train_data(model, config, path):

    adam = keras.optimizers.Adam(1e-4)
    sgd = SGD(lr=1e-4, momentum=0.9)
    metrics=["accuracy"]#, utils.mean_iou]
    model_name = str(model.name)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=metrics)
    callbacks = [#keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001),
                 keras.callbacks.ModelCheckpoint("../check_points/"+model_name+"_weights.hdf5", save_weights_only=True),
                 keras.callbacks.TensorBoard(log_dir='../logs', update_freq='batch')]

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
            batch_size=config.batch_size,
            shuffle=True,
            image_set_loader=train_loader)
        val_generator = datagen.flow_from_imageset(
            class_mode='categorical',
            classes=21,
            batch_size=config.batch_size,
            shuffle=True,
            image_set_loader=val_loader)
        batch = next(val_generator)

    workers = 1  # multiprocessing.cpu_count()

    model.fit_generator(train_generator, steps_per_epoch=config.steps_per_epoch, epochs=config.epochs,
                        use_multiprocessing=False, max_queue_size=100, workers=workers, callbacks=callbacks,
                        validation_data=val_generator, validation_steps=config.validation_steps
                        )
    #
    scores = model.evaluate_generator(val_generator, steps=100)

    # predict
    predicts = model.predict_on_batch(batch[0])
    imgs = np.argmax(predicts, axis=-1).astype(np.uint8)
    result = Image.fromarray(imgs[0], mode='P')
    result.save("../logs/test_infer.png")

    print(len(scores))
    print("loss: ", scores[0])
    print("acc: ", scores[1])
    #print("mean iou", scores[2])

