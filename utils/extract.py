import keras
import numpy as np
import matplotlib.pyplot as plt
from preprocess import *


def load_dataset():
    print('Loading MNIST dataset')
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    print('Shape of image: ', x_train[0].shape)
    return x_train, y_train, x_test, y_test


def print_images(display_count=36):
    fig = plt.figure(figsize=(20, 5))
    for i in range(display_count):
        ax = fig.add_subplot(3, 12, i + 1, xticks=[], yticks=[])
        ax.imshow(np.squeeze(x_train[i]))


def extract(display_images=False, reshape_input=False):
    x_train, y_train, x_test, y_test = load_dataset()

    #Preprocess features
    x_train = preprocess_features(x_train)
    x_test = preprocess_features(x_test)

    #Preprocess label
    y_train = one_hot_label(y_train, log=True)
    y_test = one_hot_label(y_test)

    # Split training and validation set
    x_train, x_valid = x_train[6000:], x_train[:6000]
    y_train, y_valid = y_train[6000:], y_train[:6000]

    if reshape_input:
        x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
        x_valid = x_valid.reshape((x_valid.shape[0], 28, 28, 1))
        x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    print(x_valid.shape[0], 'validation samples')

    return x_train, y_train, x_valid, y_valid, x_test, y_test
