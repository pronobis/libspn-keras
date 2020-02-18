import numpy as np
from tensorflow import keras


def load_olivetti(test_size=50):
    x = np.loadtxt("olivetti.raw").transpose().reshape(
        400, 64, 64).transpose((0, 2, 1)).astype(np.float32)
    y = np.repeat(np.arange(len(x) // 10), 10)
    train_x = x[:-test_size]
    test_x = x[-test_size:]
    train_y = y[:-test_size]
    test_y = y[-test_size:]
    return train_x, train_y, test_x, test_y


def load_mnist():

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    return x_train, y_train, x_test, y_test


def load_cifar10():

    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    return x_train, y_train, x_test, y_test


def load_data(dataset, spatial):
    if dataset == "mnist":
        x_train, y_train, x_test, y_test = load_mnist()
    elif dataset == 'cifar10':
        x_train, y_train, x_test, y_test = load_cifar10()
    else:
        x_train, y_train, x_test, y_test = load_olivetti()

    if not spatial:
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)
    elif dataset != 'cifar10':
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)

    return x_train.astype(np.float32), y_train, x_test.astype(np.float32), y_test