from tensorflow import keras
import tensorflow as tf


class FlatToRegions(keras.layers.Layer):
    """
    Reshapes a flat input of shape ``[batch, num_vars[, var_dimensionality]]``
    to ``[num_vars == scopes, decomp, batch, var_dimensionality]``

    If ``var_dimensionality`` is 1, the shape can also be ``[batch, num_vars]``.

    Args:
        **kwargs: Keyword arguments to pass on the keras.Layer super class
    """
    def __init__(self, num_decomps, **kwargs):
        self.num_decomps = num_decomps
        super(FlatToRegions, self).__init__(**kwargs)

    def call(self, inputs):
        if len(inputs.shape) == 2:
            inputs = tf.expand_dims(inputs, axis=-1)
        transposed = tf.transpose(inputs, (1, 0, 2))
        with_decomps = tf.tile(
            tf.expand_dims(transposed, axis=1), multiples=(1, self.num_decomps, 1, 1))
        return with_decomps

    def compute_output_shape(self, input_shape):

        if len(input_shape) == 2:
            num_batch, num_vars = input_shape
            return [num_vars, self.num_decomps, num_batch, 1]
        else:
            num_batch, num_vars, var_dimensionality = input_shape
            return [num_vars, self.num_decomps, num_batch, var_dimensionality]

    def get_config(self):
        config = dict(
            num_decomps=self.num_decomps
        )
        base_config = super(FlatToRegions, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
