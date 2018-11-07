"""Trains a simple NN on the MNIST dataset.
"""

from __future__ import print_function

import plaidml.keras
plaidml.keras.install_backend()

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

import pickle
import numpy

batch_size = 20
num_classes = 2
epochs = 20

section_size = 3
inputs = section_size**2

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

midpoint = x_train.shape[1]/2
start = int(midpoint-section_size/2)
end = int(start+section_size)
print("Take inner {0}x{0} section (from {1} to {2}) for {3} inputs.".format(section_size,start,end,inputs))

train_filter = numpy.where((y_train < num_classes))
test_filter = numpy.where((y_test < num_classes))

x_train = x_train[:,start:end,start:end]
x_test = x_test[:,start:end,start:end]

x_train, y_train = x_train[train_filter], y_train[train_filter]
x_test, y_test = x_test[test_filter], y_test[test_filter]

x_train = x_train.reshape(x_train.shape[0], inputs)
x_test = x_test.reshape(x_test.shape[0], inputs)


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(4, activation='relu', input_shape=(inputs,)))
# model.add(Dropout(0.2))
#model.add(Dense(8, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=0.001),
              metrics=['accuracy'])

# history = model.fit(x_train, y_train,
#                     batch_size=batch_size,
#                     epochs=epochs,
#                     verbose=1,
#                     validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

i = 0
for layer in model.layers:

    data = {"weights": layer.get_weights()[0], "bias": layer.get_weights()[1]}

    pickle.dump(data, open("verify_{0}x{0}_{1}.p".format(section_size,i), "wb"))

    i = i + 1
