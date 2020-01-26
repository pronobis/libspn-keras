import numpy as np
from sklearn.datasets import _olivetti_faces as olivetti_faces
from tensorflow import keras


def load_olivetti(test_size=50):
    bunch = olivetti_faces.fetch_olivetti_faces()
    x, y = bunch.images, bunch.target
    train_x = x[:-test_size]
    test_x = x[-test_size:]
    train_y = y[:-test_size]
    test_y = y[:-test_size]
    return train_x * 255, train_y, test_x * 255, test_y


def load_mnist():

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    return x_train, y_train, x_test, y_test


def load_data(dataset, spatial):
    if dataset == "mnist":
        x_train, y_train, x_test, y_test = load_mnist()
    else:
        x_train, y_train, x_test, y_test = load_olivetti()
    if not spatial:
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)
    else:
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)

    return x_train.astype(np.float32), y_train, x_test.astype(np.float32), y_test