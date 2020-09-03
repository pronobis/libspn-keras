from typing import Optional, Tuple

import tensorflow as tf
from tensorflow import keras


class Undecompose(keras.layers.Layer):
    """
    Undecomposes the incoming tensor by aligning the decomposition axis on the final node axis.

    Can only be done if the number of scopes (at the very first dimension of the input) is 1.

    Args:
        **kwargs: kwargs to pass onto the keras.Layer super class
    """

    def __init__(self, num_decomps: int = 1, **kwargs):
        super(Undecompose, self).__init__(**kwargs)
        self.num_decomps = num_decomps

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the internal components for this layer.

        Args:
            input_shape: Shape of the input Tensor.

        Raises:
            ValueError: If shape cannot be determined.
        """
        _, self._num_scopes, num_decomps_in, nodes_in = input_shape

        if num_decomps_in is None:
            raise ValueError("Cannot build Undecompose with unknown number of decomps.")

        if nodes_in is None:
            raise ValueError(
                "Cannot build Undecompose with unknown number of input nodes."
            )

        if num_decomps_in % self.num_decomps != 0:
            raise ValueError(
                "Number of decomps in input must be multiple of target number of decomps, got "
                "{} for input decomps and {} for target decomps.".format(
                    num_decomps_in, self.num_decomps
                )
            )

        number_of_decomps_to_join = num_decomps_in // self.num_decomps
        self._num_nodes = number_of_decomps_to_join * nodes_in

    def call(self, x: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Undecompose scopes by aligning neighboring decompositions along the node axis.

        Args:
            x: Decomposed Region input tensor.
            kwargs: Remaining keyword arguments.

        Returns:
            Undecomposed tensor.
        """
        shape = [-1, self._num_scopes, self.num_decomps, self._num_nodes]
        return tf.reshape(x, shape)

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
        num_batch, num_scopes_in, num_decomps_in, num_nodes_in = input_shape
        if num_decomps_in is None:
            raise ValueError(
                "Cannot compute output shape with unknown number of input decomps"
            )
        if num_nodes_in is None:
            raise ValueError(
                "Cannot compute output shape with unknown number of input nodes"
            )
        number_of_decomps_to_join = num_decomps_in // self.num_decomps
        return (
            num_batch,
            num_scopes_in,
            self.num_decomps,
            num_nodes_in * number_of_decomps_to_join,
        )

    def get_config(self) -> dict:
        """
        Obtain a key-value representation of the layer config.

        Returns:
            A dict holding the configuration of the layer.
        """
        config = dict(num_decomps=self.num_decomps,)
        base_config = super(Undecompose, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
