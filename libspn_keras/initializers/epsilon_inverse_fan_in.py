from tensorflow.keras import initializers
import tensorflow as tf


class EpsilonInverseFanIn(initializers.Initializer):

    def __init__(self, axis, epsilon=1e-3, dtype=None):
        self.dtype = dtype
        self.axis = axis
        self.epsilon = epsilon

    def __call__(self, shape, dtype=None, partition_info=None):

        if dtype is None:
            dtype = self.dtype

        n = shape[self.axis]

        return tf.cast(tf.fill(shape, self.epsilon / n), dtype)

    def get_config(self):
        return {
            "dtype": self.dtype.name,
            "epsilon": self.epsilon,
            "axis": self.axis
        }
