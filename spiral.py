#!/bin/python3.6

"""
Ariana Freitag
ECE-471: CGML
Assignment 2: Binary Classification
"""

import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt

from tqdm import trange

NUM_SAMP = 400
BATCH_SIZE = 64
NUM_BATCHES = 50000
LEARNING_RATE = 0.01
LAMBDA = 0.1

n_in = 2  # x,y
n_classes = 1  # 0 or 1
n_h1 = 64
n_h2 = 32


class Data(object):
    def __init__(self, num_samp=NUM_SAMP):
        self.index = np.arange(2 * num_samp)

        # generate one spiral
        theta = np.sqrt(np.random.rand(NUM_SAMP)) * 3 * math.pi
        r = 3 * -1 * theta + -1 * math.pi
        x_a = np.array([np.cos(theta) * r, np.sin(theta) * r]).T + np.random.normal(
            0, 0.5, size=(NUM_SAMP, 2)
        )
        # generate other spiral
        theta = np.sqrt(np.random.rand(NUM_SAMP)) * 3 * math.pi
        r = 3 * theta + math.pi
        x_b = np.array([np.cos(theta) * r, np.sin(theta) * r]).T + np.random.normal(
            0, 0.5, size=(NUM_SAMP, 2)
        )

        # add labels
        self.x_a_labels = np.append(x_a, np.zeros((num_samp, 1)), axis=1)
        self.x_b_labels = np.append(x_b, np.ones((num_samp, 1)), axis=1)
        # add both spirals together in one array and shuffle
        self.data = np.append(self.x_a_labels, self.x_b_labels, axis=0)
        np.random.shuffle(self.data)

    def get_batch(self, batch_size=BATCH_SIZE):
        """
        Select random subset of examples for training batch
        """
        choices = np.random.choice(self.index, size=batch_size)

        return self.data[choices]

    def get_all(self):
        """
        return all data points with no shuffling
        """
        return self.x_a_labels, self.x_b_labels


class Model(tf.Module):
    def __init__(self):
        self.h1 = tf.Variable(tf.random.normal(shape=[n_in, n_h1]))
        self.h2 = tf.Variable(tf.random.normal(shape=[n_h1, n_h2]))
        self.out = tf.Variable(tf.random.normal(shape=[n_h2, n_classes]))

        self.b1 = tf.Variable(tf.random.normal(shape=[n_h1]))
        self.b2 = tf.Variable(tf.random.normal(shape=[n_h2]))
        self.b_out = tf.Variable(tf.random.normal(shape=[n_classes]))

    def __call__(self, x):
        layer_1 = x @ self.h1 + self.b1
        layer_1 = tf.nn.relu(layer_1)

        layer_2 = layer_1 @ self.h2 + self.b2
        layer_2 = tf.nn.relu(layer_2)

        out_layer = layer_2 @ self.out + self.b_out

        return out_layer


if __name__ == "__main__":
    data = Data()
    model = Model()
    optimizer = tf.optimizers.SGD(learning_rate=LEARNING_RATE)
    train_set = data.get_batch()

    bar = trange(NUM_BATCHES)
    for i in bar:
        with tf.GradientTape() as tape:
            train_set = data.get_batch()
            x = train_set[:, 0:2]
            y = train_set[:, 2].astype(np.float32)
            y_hat = model(x)

            loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.squeeze(y), logits=tf.squeeze(y_hat)
                )
            ) + LAMBDA * tf.nn.l2_loss(model.h1)
            +LAMBDA * tf.nn.l2_loss(model.h2)
            +LAMBDA * tf.nn.l2_loss(model.out)
            +LAMBDA * tf.nn.l2_loss(model.b1)
            +LAMBDA * tf.nn.l2_loss(model.b2)
            +LAMBDA * tf.nn.l2_loss(model.b_out)

        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(zip(grads, model.variables))
        bar.set_description(f"Loss @ {i} => {loss.numpy():0.6f}")
        bar.refresh()

    # Make the plots
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[10, 10])

    # set up grid of points to make the contour plots
    x = y = np.linspace(-40, 40, 300).astype(np.float32)
    xx, yy = np.meshgrid(x, y)
    grid = np.array(list(zip(xx.flatten(), yy.flatten())))
    pred = tf.nn.sigmoid(model(grid))
    plot_data = np.reshape(pred, (300, 300))

    # plot contour from prediction
    plt.contourf(xx, yy, plot_data, [0, 0.5, 1])

    # plot sample data
    a_plot, b_plot = data.get_all()
    plt.scatter(a_plot[:, 0], a_plot[:, 1], label="Spiral 0")
    plt.scatter(b_plot[:, 0], b_plot[:, 1], label="Spiral 1")

    plt.suptitle("Binary Classification for Spirals", fontsize=12)
    ax.set(xlabel="x", ylabel="y")
    plt.legend()
    plt.show()
