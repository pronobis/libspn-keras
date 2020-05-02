import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import initializers
import numpy as np


class PermuteAndPadScopes(keras.layers.Layer):
    """
    Permutes scopes, usually applied after a ``FlatToRegions`` and a ``BaseLeaf`` layer.

    Args:
        num_decomps: Number of decompositions
        permutations: If None, permutations must be specified later
        **kwargs: kwargs to pass on to the keras.Layer superclass.
    """
    def __init__(self, permutations, **kwargs):
        super(PermuteAndPadScopes, self).__init__(**kwargs)
        self._permutation_values = permutations

    def build(self, input_shape):
        num_decomps_in = input_shape[2]
        if num_decomps_in != len(self._permutation_values):
            raise ValueError("Expected decomp size of {}, got {}".format(len(self._permutation_values), num_decomps_in))
        self.permutations = self.add_weight(
            initializer=initializers.Constant(self._permutation_values), trainable=False,
            shape=np.asarray(self._permutation_values).shape, dtype=tf.int32)

    def call(self, x):
        decomps_first = tf.transpose(x, (2, 1, 0, 3))
        decomps_first_padded = tf.pad(decomps_first, [[0, 0], [1, 0], [0, 0], [0, 0]])
        gather_indices = self.permutations + 1
        permuted = tf.gather(decomps_first_padded, gather_indices, axis=1, batch_dims=1)
        return tf.transpose(permuted, (2, 1, 0, 3))

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = dict()
        base_config = super(PermuteAndPadScopes, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
