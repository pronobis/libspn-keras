import tensorflow as tf
from tensorflow import keras


class TemporalDenseProduct(keras.layers.Layer):
    """
    Computes 'temporal' dense products to connect an interface stack at :math:`t - 1` of a dynamic
    SPN with a template SPN at :math:`t`. Computes a product of all possible combinations of
    nodes along the last axis of the two incoming layers.

    Args:
        **kwargs: kwargs to pass on to the keras.Layer super class.
    """

    def __init__(self, **kwargs):
        self._num_nodes_out = None
        super(TemporalDenseProduct, self).__init__(**kwargs)

    def build(self, input_shape):
        a_shape, b_shape = input_shape
        if a_shape != b_shape:
            raise ValueError("Shapes of incoming layers must be equal, but got {} and {}".format(
                a_shape, b_shape))

        self._num_nodes_out = a_shape[-1] ** 2
        super(TemporalDenseProduct, self).build(input_shape)

    def call(self, x):
        # Split in list of tensors which will be added up using outer products
        a, b = x
        a_expanded = tf.expand_dims(a, axis=-1)
        b_expanded = tf.expand_dims(b, axis=-2)
        outer_dims = tf.shape(a_expanded)[:-1]
        out_shape = tf.concat([outer_dims, [self._num_nodes_out]], axis=0)
        return tf.reshape(a_expanded + b_expanded, out_shape)

    def compute_output_shape(self, input_shape):
        a_shape, _ = input_shape
        num_scopes_in, num_decomps, num_batch, num_nodes_in = a_shape
        return (
            num_scopes_in,
            num_decomps,
            num_batch,
            num_nodes_in ** 2
        )
