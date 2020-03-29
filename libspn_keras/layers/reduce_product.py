import tensorflow as tf
from tensorflow import keras


class ReduceProduct(keras.layers.Layer):
    """
    Computes products per decomposition and scope by reduction. Assumes the
    incoming tensor is of shape ``[num_scopes, num_decomps, num_batch, num_nodes]`` and produces an
    output of ``[num_scopes // num_factors, num_decomps, num_batch, num_nodes]``.

    Args:
        num_factors: Number of factors per product
        **kwargs: kwargs to pass on to the keras.Layer super class.
    """
    def __init__(self, num_factors, **kwargs):
        super(ReduceProduct, self).__init__(**kwargs)
        self.num_factors = num_factors
        self._num_decomps = self._num_scopes = self._num_scopes_in \
            = self._num_products = self._num_nodes_in = None

    def build(self, input_shape):
        self._num_scopes_in, self._num_decomps, _, self._num_nodes_in = input_shape
        if self._num_scopes_in % self.num_factors != 0:
            raise ValueError("Number of input scopes is not divisible by factor")
        self._num_scopes = self._num_scopes_in // self.num_factors
        self._num_products = self._num_nodes_in
        super(ReduceProduct, self).build(input_shape)

    def call(self, x):
        # Split in list of tensors which will be added up using outer products
        shape = [self._num_scopes, self.num_factors, self._num_decomps, -1, self._num_nodes_in]
        return tf.reduce_sum(tf.reshape(x, shape=shape), axis=1)

    def compute_output_shape(self, input_shape):
        num_scopes_in, num_decomps, num_batch, num_nodes_in = input_shape
        return (
            num_scopes_in // self.num_factors,
            self._num_decomps,
            num_batch,
            num_nodes_in
        )

    def get_config(self):
        config = dict(
            num_factors=self.num_factors,
        )
        base_config = super(ReduceProduct, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
