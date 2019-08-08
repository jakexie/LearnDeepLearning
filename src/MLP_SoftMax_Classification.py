import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizer import SGD

# create simulation data
import numpy as np
data_train = np.random.random((1000, 20))
labels_train = keras.utils.to_categorical((np.random.randint(10, size=(1000,1))), num_classes=10) 
data_test = np.random.random((1000, 20))
labels_test = keras.utils.to_categorical((np.random.randint(10, size=(1000,1))), num_classes=10) 

data_2_predict = np.random.random((1000, 20))

# set up cnn
model = Sequential();
model.add(Dense(64, activation='relu', input_dim=20))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
# compile
sgd = SGD(lr=0.01, decay=le-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metics=['accuracy'])
model.fit(data_train, labels_train, epochs=20, batch_size=128)
score = model.evaluate(data_test, labels_test, batch_size=128)
pre_labels = model.predict(data_2_predict)
