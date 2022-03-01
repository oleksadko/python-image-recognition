# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 20:24:29 2022

@author: oleks
"""

import numpy
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
tf.config.experimental_run_functions_eagerly(True)

# preset random seed so results can be reproduced
seed = 19

# loading in data
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
# change data from 0-255 to 0-1
x_train = x_train.astype("float32")
print(x_train)
x_train = x_train/255
print("\n", x_train)
x_test = x_test.astype("float32")
# one hot encodeing
y_train = keras.utils.to_categorical(y_train)
print(y_train)
y_test = keras.utils.to_categorical(y_test)
# number of classes - different types of images, plane, cat, car, etc.
class_num = y_test.shape[1]
print(class_num)

# creating the cnn model
model = keras.Sequential()
# convolutial layer, runs filters on image
model.add(keras.layers.Conv2D(32,# no. of channels
                              (3, 3),# filter size
                              input_shape=x_train.shape[1:],# shape of image
                              activation = "relu",
                              padding = "same" #image size stays the same
                              ))

# dropout layer, preventing overfitting
model.add(keras.layers.Dropout(0.2))# 20% of connections are dropped

# tbf i dunno what this does, someone on stackoverflow said to use it
# i think its pooling
model.add(keras.layers.BatchNormalization())


#TODO make this neater
model.add(keras.layers.Conv2D(64, 3, activation="relu", padding="same"))
model.add(keras.layers.MaxPooling2D(2))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.BatchNormalization())
    
# model.add(keras.layers.Conv2D(128, 3, activation='relu', padding='same'))
# model.add(keras.layers.Dropout(0.2))
# model.add(keras.layers.BatchNormalization())

# flatten the data
model.add(keras.layers.Flatten()) 
model.add(keras.layers.Dropout(0.2))

# classify images based on feature maps
model.add(keras.layers.Dense(32, activation='relu'))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.BatchNormalization())

# Select the highest activation neuron out of class_num - 10
model.add(keras.layers.Dense(class_num, activation='softmax'))

model.compile(loss="categorical_crossentropy", optimizer="adam", 
              metrics=["accuracy", "categorical_accuracy"])
print(model.summary())

numpy.random.seed(seed)
history = model.fit(x_train, y_train, validation_data=(x_test, y_test),
                    epochs=25, batch_size=64)

scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

pd.DataFrame(history.history).plot()
plt.show()