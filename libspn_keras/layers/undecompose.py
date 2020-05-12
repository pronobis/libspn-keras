import tensorflow as tf
from tensorflow import keras


class Undecompose(keras.layers.Layer):
    """
    Undecomposes the incoming tensor by aligning the decomposition axis on the final node axis.
    Can only be done if the number of scopes (at the very first dimension of the input) is 1.

    Args:
        **kwargs: kwargs to pass onto the keras.Layer super class
    """
    def __init__(self, num_decomps=1, **kwargs):
        super(Undecompose, self).__init__(**kwargs)
        self.num_decomps = num_decomps
        self._num_nodes = self._num_scopes = None

    def build(self, input_shape):
        _, self._num_scopes, num_decomps_in, nodes_in = input_shape

        if num_decomps_in % self.num_decomps != 0:
            raise ValueError("Number of decomps in input must be multiple of target number of decomps, got "
                             "{} for input decomps and {} for target decomps.".format(num_decomps_in, self.num_decomps))

        number_of_decomps_to_join = num_decomps_in // self.num_decomps
        self._num_nodes = number_of_decomps_to_join * nodes_in

    def call(self, x):
        shape = [-1, self._num_scopes, self.num_decomps, self._num_nodes]
        return tf.reshape(x, shape)

    def compute_output_shape(self, input_shape):
        num_batch, num_scopes_in, num_decomps_in, nodes_in = input_shape
        number_of_decomps_to_join = num_decomps_in // self.num_decomps
        self._num_nodes = number_of_decomps_to_join * nodes_in
        return (
            num_batch,
            num_scopes_in,
            self.num_decomps,
            nodes_in * number_of_decomps_to_join
        )

    def get_config(self):
        config = dict(
            num_decomps=self.num_decomps,
        )
        base_config = super(Undecompose, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
