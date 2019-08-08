import tensorflow_datasets as tfds
import tensorflow as tf

tf.enable_eager_execution()
print(tfds.list_builders())
ds_train, ds_test = tfds.load(name="mnist", split=["train", "test"])
