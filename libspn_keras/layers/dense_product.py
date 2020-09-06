import functools
import operator
from typing import Optional, Tuple

import tensorflow as tf
from tensorflow import keras


class DenseProduct(keras.layers.Layer):
    """
    Computes products per decomposition and scope by an 'n-order' outer product.

    Assumes the incoming tensor is of shape ``[num_scopes, num_decomps, num_batch, num_nodes]`` and
    produces an output of ``[num_scopes // num_factors, num_decomps, num_batch, num_nodes ** num_factors]``.
    It can be considered a *dense* product as it computes all possible products given the scopes it has
    to merge.

    Args:
        num_factors (int): Number of factors per product
        **kwargs: kwargs to pass on to the keras.Layer super class
    """

    def __init__(self, num_factors: int, **kwargs):
        super(DenseProduct, self).__init__(**kwargs)
        self.num_factors = num_factors

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the internal components for this layer.

        Args:
            input_shape: Shape of the input Tensor.

        Raises:
            ValueError: When shape could not be determined.
        """
        _, self._num_scopes_in, self._num_decomps, self._num_nodes_in = input_shape
        if self._num_scopes_in is None:
            raise ValueError("Cannot build with unknown number of input scopes")
        if self._num_nodes_in is None:
            raise ValueError("Cannot build with unknown number of input nodes")
        if self._num_scopes_in % self.num_factors != 0:
            raise ValueError("Number of input scopes is not divisible by factor")
        self._num_scopes_out = self._num_scopes_in // self.num_factors
        self._num_products = self._num_nodes_in ** self.num_factors
        super(DenseProduct, self).build(input_shape)

    def call(self, x: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Compute a dense product layer by using all possible combinations of children.

        Args:
            x: Region Tensor.
            kwargs: Remaining keyword arguments.

        Returns:
            A Tensor with dense products.
        """
        # Split in list of tensors which will be added up using outer products
        shape = [
            -1,
            self._num_scopes_out,
            self.num_factors,
            self._num_decomps,
            self._num_nodes_in,
        ]
        with tf.name_scope("LogProbPerFactor"):
            log_prob_per_factor = tf.split(
                tf.reshape(x, shape=shape), axis=2, num_or_size_splits=self.num_factors
            )

        # Reshape to [scopes, decomps, batch, 1, ..., child.dim_nodes, ..., 1] where
        # child.dim_nodes is inserted at the i-th index within the trailing 1s, where i corresponds
        # to the index of the log prob
        with tf.name_scope("ToBroadcastable"):
            log_prob_per_factor_broadcastable = [
                tf.reshape(
                    log_prob,
                    [-1, self._num_scopes_out, self._num_decomps]
                    + [
                        1 if j != i else self._num_nodes_in
                        for j in range(self.num_factors)
                    ],
                )
                for i, log_prob in enumerate(log_prob_per_factor)
            ]
        # Add up everything (effectively computing an outer product) and flatten the last
        # num_factors dimensions.
        with tf.name_scope("NOrderOuterProduct"):
            outer_product = functools.reduce(
                operator.add, log_prob_per_factor_broadcastable
            )
            return tf.reshape(
                outer_product,
                [-1, self._num_scopes_out, self._num_decomps, self._num_products],
            )

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute output shape of the layer.

        Args:
            input_shape: Input shape of the layer.

        Returns:
            Tuple of ints holding the output shape of the layer.

        Raises:
            ValueError: When shape cannot be determined.
        """
        num_batch, num_scopes_in, num_decomps, num_nodes_in = input_shape
        if num_scopes_in is None:
            raise ValueError(
                "Cannot compute output shape with unknown number of input scopes"
            )
        if num_nodes_in is None:
            raise ValueError(
                "Cannot compute output shape with unknown number of input nodes"
            )
        return (
            num_batch,
            num_scopes_in // self.num_factors,
            self._num_decomps,
            int(num_nodes_in ** self.num_factors),
        )

    def get_config(self) -> dict:
        """
        Obtain a key-value representation of the layer config.

        Returns:
            A dict holding the configuration of the layer.
        """
        config = dict(num_factors=self.num_factors,)
        base_config = super(DenseProduct, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
