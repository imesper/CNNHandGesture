'''Train a deep CNN on the Thalmic Labs gestures dataset.

GPU run command:
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python gestureDL.py

'''

from __future__ import print_function

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
import dataUtils
import numpy as np

batch_size = 32
nClasses = 6
nEpoch = 100
kernel = 2
dataAugmentation = False

(X_train, y_train), (X_test, y_test) = dataUtils.loadData()

#print('X_train shape:', X_train.shape)

#print(X_train.shape[0], 'train samples')
#print(X_test.shape[0], 'test samples')

Y_train = np_utils.to_categorical(y_train, nClasses)
Y_test = np_utils.to_categorical(y_test, nClasses)

model = Sequential()

model.add(Convolution2D(6, kernel, kernel,
                        border_mode='same', input_shape=(1, 50, 8)))
model.add(Activation('relu'))

model.summary()

# model.add(Convolution2D(64, kernel, kernel))
# model.add(Activation('tanh'))
# model.add(Dropout(0.25))
# model.summary()
# model.add(Convolution2D(128, kernel, kernel, border_mode='same'))
# model.add(Activation('tanh'))
# model.add(Convolution2D(128,kernel, kernel))
# model.add(Activation('tanh'))
# model.add(Dropout(0.25))
# model.add(Convolution2D(128, kernel, kernel, border_mode='same'))
# model.add(Activation('tanh'))
# model.add(Convolution2D(128,kernel, kernel))
# model.add(Activation('tanh'))
# model.add(Dropout(0.25))

model.summary()
model.add(Flatten())
# model.add(Dense(512))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
model.add(Dense(nClasses))
model.add(Activation('softmax'))
model.summary()

sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
#X_train /= np.max(X_train)
#X_test /= np.max(X_train)

hist = model.fit(X_train, Y_train,
                 verbose=2,
                 batch_size=batch_size,
                 nb_epoch=nEpoch,
                 validation_data=(X_test, Y_test),
                 shuffle=True)

score, acc = model.evaluate(X_test, Y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
scores = model.evaluate(X_test, Y_test)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
