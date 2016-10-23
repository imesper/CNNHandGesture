from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils

import numpy as np 


X_train = np.array([[1,2], [6,5], [8,2]])

y_train = np.array([2,3,7])
print y_train
y_train = np_utils.to_categorical(y_train)
print y_train
input_dim = X_train.shape[1]

model = Sequential()

model.add(Dense(output_dim=512, input_dim=input_dim))
model.add(Activation("relu"))
model.add(Dense(output_dim=8))
model.add(Activation("softmax"))
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

model.fit(X_train, y_train, nb_epoch=50, batch_size=32)