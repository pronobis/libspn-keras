from tensorflow import keras
import tensorflow as tf


class NegativeLogLikelihood(keras.losses.Loss):

    def call(self, y_true, y_pred):

        return -tf.reduce_logsumexp(y_pred, axis=1)
