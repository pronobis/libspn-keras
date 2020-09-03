from typing import Optional, Tuple

import tensorflow as tf
from tensorflow import keras


class ReduceProduct(keras.layers.Layer):
    """
    Computes products per decomposition and scope by reduction.

    Assumes the incoming tensor is of shape ``[num_batch, num_scopes, num_decomps, num_nodes]``
    and produces an output of ``[num_batch, num_scopes // num_factors, num_decomps, num_nodes]``.

    Args:
        num_factors: Number of factors per product
        **kwargs: kwargs to pass on to the keras.Layer super class.
    """

    def __init__(self, num_factors: int, **kwargs):
        super(ReduceProduct, self).__init__(**kwargs)
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
            raise ValueError("Cannot build product with unknown number of input scopes")
        if self._num_scopes_in % self.num_factors != 0:
            raise ValueError("Number of input scopes is not divisible by factor")
        self._num_scopes = self._num_scopes_in // self.num_factors
        self._num_products = self._num_nodes_in
        super(ReduceProduct, self).build(input_shape)

    def call(self, x: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Compute product by reduction over scope subset.

        Args:
            x: Region input tensor.
            kwargs: Remaining keyword arguments.

        Returns:
            Products where by reducing over neighboring scopes.
        """
        # Split in list of tensors which will be added up using outer products
        shape = [
            -1,
            self._num_scopes,
            self.num_factors,
            self._num_decomps,
            self._num_nodes_in,
        ]
        return tf.reduce_sum(tf.reshape(x, shape=shape), axis=2)

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
            ValueError: If shape cannot be determined.
        """
        num_scopes_in, num_decomps, num_batch, num_nodes_in = input_shape
        if num_scopes_in is None:
            raise ValueError("Cannot compute shape with unknown number of input scopes")
        return (
            num_batch,
            num_scopes_in // self.num_factors,
            self._num_decomps,
            num_nodes_in,
        )

    def get_config(self) -> dict:
        """
        Obtain a key-value representation of the layer config.

        Returns:
            A dict holding the configuration of the layer.
        """
        config = dict(num_factors=self.num_factors,)
        base_config = super(ReduceProduct, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
