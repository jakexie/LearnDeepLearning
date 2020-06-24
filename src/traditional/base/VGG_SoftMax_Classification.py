import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import SGD

#create Simulation data
data_train = np.random.random(100, 100, 100, 3)
label_train = keras.utils.to_categorical(np.random.randint(10, size(100, 1)), num_classes=10)
data_train = np.random.random(100, 100, 100, 3)
label_train = keras.utils.to_categorical(np.random.randint(10, size(100, 1)), num_classes=10)

# create VGG
model = Sequentail()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(100,100,3)))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(COnv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=le-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizers=sgd)

# train
model.fit(data_train, label_train, epochs=10, batch_size=32)

# evaluate
model.evaluate(data_test,label_test, batch_size=32)

# predict
label_predicts = model.predict(data_test)



