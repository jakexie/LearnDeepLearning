import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout

# create simulation data
data_train = np.random.random((1000, 20))
label_train = np.random.randint(2, size(1000,1))
data_test = np.random.random((1000, 20))
label_test = np.random.randint(2, size(1000,1))

# creat MLP
model = Sequential()
model.add(Dense(64, input_dim=20, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# compile
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# train
model.fit(data_train, label_train, epochs=20, batch_size=128)

# evaluate
model.evaluate(data_test, label_test, batch_size=128)

# prediction
predict_labels = model.predict(data_test)

