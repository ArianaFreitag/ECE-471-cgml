#!/bin/python3.6

"""
Ariana Freitag
ECE-471: CGML
Assignment 4: CIFAR10 Classification
"""
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    Flatten,
    MaxPooling2D,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator


LEARNING_RATE = 0.001
LAMBDA = 0.01
EPOCHS = 75
BATCH_SIZE = 200
SPLIT = 0.9
input_shape = (32, 32, 3)
num_classes = 10


class Data(object):
    def __init__(self):

        np.random.seed(31415)

        (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar100.load_data()
        self.index = np.arange(self.x_train.shape[0])
        self.x_train, self.y_train = self.get_batch()
        self.x_train, self.y_train, self.x_val, self.y_val = self.split_data(
            self.x_train.shape[0]
        )

        self.y_train = keras.utils.to_categorical(self.y_train, num_classes)
        self.y_val = keras.utils.to_categorical(self.y_val, num_classes)
        self.y_test = keras.utils.to_categorical(self.y_test, num_classes)

    def split_data(self, size):
        """
        Split the data to get a validation set
        """
        self.x_val = self.x_train[math.ceil(size * SPLIT) : len(self.x_train)]
        self.y_val = self.y_train[math.ceil(size * SPLIT) : len(self.y_train)]

        return (
            self.x_train[0 : math.ceil(size * SPLIT)],
            self.y_train[0 : math.ceil(size * SPLIT)],
            self.x_val,
            self.y_val,
        )

    def get_batch(self):
        """
        Select random subset of examples for training batch
        """
        choices = np.random.choice(self.index, size=self.x_train.shape[0])

        return self.x_train[choices], self.y_train[choices]

    def get_data(self):
        """
        Return all data
        """
        self.x_train = self.x_train.astype("float32")
        self.x_test = self.x_test.astype("float32")
        self.x_val = self.x_val.astype("float32")
        self.x_train /= 255
        self.x_test /= 255
        self.x_val /= 255
        return (
            self.x_train,
            self.y_train,
            self.x_val,
            self.y_val,
            self.x_test,
            self.y_test,
        )


class Model:
    def __init__(self):
        """
        Construct a CNN using L2 regularization, drop-out, max pooling, and batch normalization

        """
        self.datagen = ImageDataGenerator(
            horizontal_flip=True,
            fill_mode="constant",
            width_shift_range=4,
            height_shift_range=4,
        )

        self.model = tf.keras.Sequential()
        self.model.add(BatchNormalization())
        self.model.add(
            Conv2D(
                32,
                kernel_size=(3, 3),
                activation="relu",
                input_shape=input_shape,
                kernel_regularizer=tf.keras.regularizers.l2(l=LAMBDA),
            )
        )
        self.model.add(
            Conv2D(
                64,
                kernel_size=(3, 3),
                activation="relu",
                input_shape=(32, 64, 3),
                kernel_regularizer=tf.keras.regularizers.l2(l=LAMBDA),
            )
        )
        self.model.add(Dropout(0.25))
        self.model.add(
            Conv2D(
                64,
                kernel_size=(3, 3),
                activation="relu",
                input_shape=(64, 64, 3),
                kernel_regularizer=tf.keras.regularizers.l2(l=LAMBDA),
            )
        )
        self.model.add(Dropout(0.25))
        self.model.add(
            Conv2D(
                64,
                kernel_size=(3, 3),
                activation="relu",
                input_shape=(64, 64, 3),
                kernel_regularizer=tf.keras.regularizers.l2(l=LAMBDA),
            )
        )
        self.model.add(BatchNormalization())
        self.model.add(
            Conv2D(
                64,
                kernel_size=(3, 3),
                activation="relu",
                input_shape=(64, 64, 3),
                kernel_regularizer=tf.keras.regularizers.l2(l=LAMBDA),
            )
        )
        self.model.add(
            Conv2D(
                32,
                kernel_size=(3, 3),
                activation="relu",
                input_shape=(64, 32, 3),
                kernel_regularizer=tf.keras.regularizers.l2(l=LAMBDA),
            )
        )
        self.model.add(
            Conv2D(
                32,
                kernel_size=(3, 3),
                activation="relu",
                input_shape=(32, 32, 3),
                kernel_regularizer=tf.keras.regularizers.l2(l=LAMBDA),
            )
        )
        self.model.add(
            Conv2D(
                32,
                kernel_size=(3, 3),
                activation="relu",
                input_shape=(32, 32, 3),
                kernel_regularizer=tf.keras.regularizers.l2(l=LAMBDA),
            )
        )
        self.model.add(MaxPooling2D(pool_size=(3, 3)))
        self.model.add(Flatten())
        self.model.add(Dense(num_classes, activation="softmax"))
        adam = keras.optimizers.Adam(
            learning_rate=LEARNING_RATE, beta_1=0.9, beta_2=0.999, amsgrad=False
        )
        self.model.compile(
            loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"]
        )

    def evaluate(self, x_test, y_test):
        self.model.evaluate(x_test, y_test)

    def __call__(self, feature, label, x_val, y_val):
        self.datagen.fit(feature)
        self.model.fit_generator(
            self.datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
            steps_per_epoch=len(x_train) / BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=(x_val, y_val),
        )

        self.model.save("model.h5")

    if __name__ == "__main__":

        data = Data()
        model = Model()
        x_train, y_train, x_val, y_val, x_test, y_test = data.get_data()

        model(x_train, y_train, x_val, y_val)
        model.evaluate(x_test, y_test)
