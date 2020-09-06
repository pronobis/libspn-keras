from typing import Optional, Tuple

import tensorflow as tf
from tensorflow import keras


class FlatToRegions(keras.layers.Layer):
    """
    Flat representation to a dense representation.

    Reshapes a flat input of shape ``[batch, num_vars[, var_dimensionality]]``
    to ``[batch, num_vars == scopes, decomp, var_dimensionality]``

    If ``var_dimensionality`` is 1, the shape can also be ``[batch, num_vars]``.

    Args:
        **kwargs: Keyword arguments to pass on the keras.Layer super class
    """

    def __init__(self, num_decomps: int, **kwargs):
        self.num_decomps = num_decomps
        super(FlatToRegions, self).__init__(**kwargs)

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Reshape to a region representation.

        Args:
            inputs: Flat representation of raw input to the SPN.
            kwargs: Remaining keyword arguments.

        Returns:
            A Tensor with axes ``[batch, scopes, decomps, nodes]``.
        """
        if len(inputs.shape) == 2:
            inputs = tf.expand_dims(inputs, axis=-1)
        with_decomps = tf.tile(
            tf.expand_dims(inputs, axis=2), multiples=(1, 1, self.num_decomps, 1)
        )
        return with_decomps

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute output shape of the layer.

        Args:
            input_shape: Input shape of the layer.

        Returns:
            Tuple of ints holding the output shape of the layer.
        """
        num_batch, num_vars, var_dimensionality = input_shape
        return num_batch, num_vars, self.num_decomps, var_dimensionality

    def get_config(self) -> dict:
        """
        Obtain a key-value representation of the layer config.

        Returns:
            A dict holding the configuration of the layer.
        """
        config = dict(num_decomps=self.num_decomps)
        base_config = super(FlatToRegions, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
