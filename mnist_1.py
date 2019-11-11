#!/bin/python3.6

"""
Ariana Freitag
ECE-471: CGML
Assignment 4: CIFAR Classification
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import gzip
import pickle
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout


NUM_BATCHES = 50000
LEARNING_RATE = 0.01
LAMBDA = 0.001
EPOCHS = 10
BATCH_SIZE = 1024
save_file = "mnist.pkl"
input_shape = (28, 28, 1)
num_classes = 10


class Data(object):
    def __init__(self):

        np.random.seed(31415)

        # I serialized a pickle in another program, and then used that pickle when loading the data for each trainging loop
        with open("mnist.pkl", "rb") as f:
            self.dataset = pickle.load(f)

        # clean up the data, reshape it to be a grid
        for key in ("train_img", "test_img"):
            self.dataset[key] = self.dataset[key].astype(np.float32)
            self.dataset[key] /= 255.0
            self.dataset[key] = self.dataset[key].reshape(
                self.dataset[key].shape[0], 28, 28, 1
            )

        # get my val data from the larger training set
        self.dataset["train_img"], self.dataset[
            "train_label"
        ], self.val_img, self.val_label = self.split_data()

        # one hot encode the labels
        self.dataset["train_label"] = self.one_hot(self.dataset["train_label"])
        self.dataset["test_label"] = self.one_hot(self.dataset["test_label"])
        self.val_label = self.one_hot(self.val_label)

        self.train_length = len(self.dataset["train_img"])
        self.index = np.arange(self.train_length)

    def one_hot(self, X):
        """
        Transform the labels into a one hot encoding
        """
        T = np.zeros((X.size, 10))
        for idx, row in enumerate(T):
            row[X[idx]] = 1

        return T

    def get_batch(self):
        """
        Select random subset of examples for training batch
        """
        choices = np.random.choice(self.index, size=self.train_length)

        return self.dataset["train_label"][choices], self.dataset["train_img"][choices]

    def split_data(self):
        """
        Split the data to get a validation set
        """
        self.val_img = self.dataset["train_img"][
            50000 : len(self.dataset["train_label"])
        ]
        self.val_label = self.dataset["train_label"][
            50000 : len(self.dataset["train_label"])
        ]

        return (
            self.dataset["train_img"][0:50000],
            self.dataset["train_label"][0:50000],
            self.val_img,
            self.val_label,
        )

    def get_val(self):
        return self.val_label, self.val_img

    def get_test(self):
        return self.dataset["test_label"], self.dataset["test_img"]


class Model(tf.Module):
    def __init__(self):
        """
        Construct a CNN using L2 regularization, drop-out, and max pooling
        """
        self.model = tf.keras.Sequential()
        self.model.add(
            Conv2D(
                64,
                kernel_size=(3, 3),
                activation="relu",
                input_shape=input_shape,
                kernel_regularizer=tf.keras.regularizers.l2(l=LAMBDA),
            )
        )
        self.model.add(MaxPooling2D(pool_size=(3, 3)))
        self.model.add(Flatten())
        self.model.add(Dense(32, activation="relu"))
        self.model.add(Dropout(0.25))
        self.model.add(Dense(num_classes, activation="softmax"))

        self.model.compile(
            loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
        )

    def evaluate(self, test_img, test_label):
        self.model.evaluate(test_img, test_label)

    def __call__(self, feature, label, val_img, val_label):
        self.model.fit(
            feature,
            label,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=(val_img, val_label),
        )
        self.model.summary()


if __name__ == "__main__":
    data = Data()
    model = Model()
    train_label, train_img = data.get_batch()
    val_label, val_img = data.get_val()
    test_label, test_img = data.get_test()
    model(train_img, train_label, val_img, val_label)
    model.evaluate(test_img, test_label)

    """
    Model Results:
    My model has 132,074 parameters and was able to achieve a 97.5% accuracy rate after 10 epochs.
    """
