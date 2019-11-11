#!/bin/python3.6

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tqdm import trange

"""
Ariana Freitag
ECE-471: CGML
Assignment 1
"""

NUM_FEATURES = 10  # number of basis functions
NUM_SAMP = 50
BATCH_SIZE = 32
NUM_BATCHES = 600
LEARNING_RATE = 0.1


class Data(object):
    def __init__(self, num_features=NUM_FEATURES, num_samp=NUM_SAMP):
        """
        Draw random weights and bias. Project vectors in R^NUM_FEATURES
        onto R with said weights and bias.
        """
        num_samp = NUM_SAMP
        np.random.seed(31415)
        self.e = np.random.normal(0, 0.1, size=(num_samp))

        self.index = np.arange(num_samp)
        self.x = np.random.uniform(size=(num_samp)).astype(np.float32)
        self.y = np.sin(self.x * 2 * np.pi) + self.e

    def get_batch(self, batch_size=BATCH_SIZE):
        """
        Select random subset of examples for training batch
        """
        choices = np.random.choice(self.index, size=batch_size)

        return self.x[choices], self.y[choices].flatten()


class Model(tf.Module):
    def __init__(self, num_features=NUM_FEATURES):
        """
        A plain linear regression model with a bias term
        """
        self.w = tf.Variable(tf.random.normal(shape=[num_features, 1]))
        self.b = tf.Variable(tf.zeros(shape=[1, 1]))
        self.mu = tf.Variable(tf.random.normal(shape=[1, num_features]))
        self.sigma = tf.Variable(tf.random.normal(shape=[1, num_features]))

    def __call__(self, x):
        x = tf.expand_dims(x, -1)
        phi = tf.math.exp(-(x - self.mu) ** 2 / self.sigma ** 2)
        return tf.squeeze(phi @ self.w + self.b)


if __name__ == "__main__":
    data = Data()
    model = Model()
    optimizer = tf.optimizers.SGD(learning_rate=LEARNING_RATE)

    bar = trange(NUM_BATCHES)
    for i in bar:
        with tf.GradientTape() as tape:
            x, y = data.get_batch()
            y_hat = model(x)
            loss = tf.reduce_mean((y_hat - y) ** 2)

        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(zip(grads, model.variables))

        bar.set_description(f"Loss @ {i} => {loss.numpy():0.6f}")
        bar.refresh()

    # plot of predicted sinewave
    x_plot = np.linspace(0, 1, 200)
    plt.plot(x_plot, model(x_plot.astype(np.float32)))
    plt.plot(x_plot, np.sin(x_plot * 2 * np.pi), linestyle="--")
    plt.plot(data.x, data.y, "o")
    plt.suptitle("Fit", fontsize=12)
    plt.show()

    # plot of basis functions
    for i in range(NUM_FEATURES):
        plt.plot(
            np.linspace(-10, 10),
            tf.math.exp(
                -(np.linspace(0, 5).astype(np.float32) - model.mu[0][i]) ** 2
                / model.sigma[0][i] ** 2
            ),
        )
    plt.suptitle("Bases for Fit", fontsize=12)
    plt.show()
