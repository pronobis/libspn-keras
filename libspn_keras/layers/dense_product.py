import tensorflow as tf
from tensorflow import keras
import operator
import functools


class DenseProduct(keras.layers.Layer):

    def __init__(self, num_factors):
        super(DenseProduct, self).__init__()
        self.num_factors = num_factors
        self._num_decomps = self._num_scopes = self._num_scopes_in \
            = self._num_products = self._num_nodes_in = None

    def build(self, input_shape):
        self._num_scopes_in, self._num_decomps, _, self._num_nodes_in = input_shape
        if self._num_scopes_in % self.num_factors != 0:
            raise ValueError("NUmber of input scopes is not divisible by factor")
        self._num_scopes = self._num_scopes_in // self.num_factors
        self._num_products = self._num_nodes_in ** self.num_factors
        super(DenseProduct, self).build(input_shape)

    def call(self, x):
        # Split in list of tensors which will be added up using outer products
        shape = [self._num_scopes, self.num_factors, self._num_decomps, -1, self._num_nodes_in]

        log_prob_per_in_scope = tf.split(
            tf.reshape(x, shape=shape), axis=1, num_or_size_splits=self.num_factors)

        # Reshape to [scopes, decomps, batch, 1, ..., child.dim_nodes, ..., 1] where
        # child.dim_nodes is inserted at the i-th index within the trailing 1s, where i corresponds
        # to the index of the log prob
        log_prob_per_in_scope = [
            tf.reshape(
                log_prob,
                [self._num_scopes, self._num_decomps, -1] +
                [1 if j != i else self._num_nodes_in for j in range(self.num_factors)]
            )
            for i, log_prob in enumerate(log_prob_per_in_scope)
        ]
        # Add up everything (effectively computing an outer product) and flatten
        return tf.reshape(
            functools.reduce(operator.add, log_prob_per_in_scope),
            [self._num_scopes, self._num_decomps, -1, self._num_products]
        )

    def compute_output_shape(self, input_shape):
        num_scopes_in, num_decomps, _, num_nodes_in = input_shape
        return (
            num_scopes_in // self.num_factors,
            self._num_decomps,
            None,
            num_nodes_in ** self.num_factors
        )

    def get_config(self):
        config = dict(
            num_factors=self.num_factors,
        )
        base_config = super(DenseProduct, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
