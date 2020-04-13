import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import initializers


class PermuteAndPadScopes(keras.layers.Layer):
    """
    Permutes scopes, usually applied after a ``FlatToRegions`` and a ``BaseLeaf`` layer.

    Args:
        num_decomps: Number of decompositions
        permutations: If None, permutations must be specified later
        **kwargs: kwargs to pass on to the keras.Layer superclass.
    """
    def __init__(self, num_decomps, permutations=None, **kwargs):
        super(PermuteAndPadScopes, self).__init__(**kwargs)
        self.num_decomps = num_decomps
        self.permutations = None
        if permutations is not None:
            self.set_permutations(permutations)

    def set_permutations(self, permutations):
        self.permutations = self.add_weight(
            initializer=initializers.Constant(permutations), trainable=False,
            shape=permutations.shape, dtype=tf.int32)

    def call(self, x):

        decomps_first = tf.transpose(x, (1, 0, 2, 3))
        decomps_first_padded = tf.pad(decomps_first, [[0, 0], [1, 0], [0, 0], [0, 0]])
        gather_indices = self.permutations + 1

        if self.permutations is None:
            raise ValueError("First need to set permutations")
        permuted = tf.gather(decomps_first_padded, gather_indices, axis=1, batch_dims=1)
        return tf.transpose(permuted, (1, 0, 2, 3))

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = dict(
            num_decomps=self.num_decomps,
        )
        base_config = super(PermuteAndPadScopes, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
