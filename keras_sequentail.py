from keras.model import Sequential
from keras.layers import Dense, Activation

model = Sequential([Dense(32, input_shape=(784,)), Activation('relu'), Dense(10), Activation('softmax'),])

# multi classes
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# binary classes
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# mean square root
model.compile(optimizer='rmsprop', loss='mse')

# user defined metrics
def mean_pred(y_true, y_pred)
    return K.mean(y_pred)

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy', mean_pred])

# train
# generate dataset
# binary
import numpy as np
data = np.random.random((1000, 100))
labels = np.random.randint(2, size=(1000, 1))

model = Squential()
model.add(Dense(32, activation='relu', input_dim=100))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss = 'binary_crossentropy', metrics=['accuracy'])
model.fit(data, labels, epochs=10, batch_size=32)

# multiple
labels = np.random.randint(10, size=(1000, 1))
one_hot_labels = keras.utils.to_categorical(label, num_classes=10)
model.fit(data, one_hot_labels, epoches=10, batch_size=32)
