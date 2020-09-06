from typing import Optional, Tuple

import tensorflow as tf
from tensorflow import keras


class SpatialToRegions(keras.layers.Layer):
    """
    Reshapes spatial SPN layer to a dense layer.

    The dense output has leading dimensions for scopes and decomps (which will be ``[1, 1]``).
    """

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the internal components for this layer.

        Args:
            input_shape: Shape of the input Tensor.

        Raises:
            ValueError: If dimensions are unknown.
        """
        _, num_cells_vertical, num_cells_horizontal, num_nodes = input_shape
        if num_cells_horizontal is None:
            raise ValueError(
                "Cannot compute shape with unknown number of horizontal cells"
            )
        if num_cells_vertical is None:
            raise ValueError(
                "Cannot compute shape with unknown number of vertical cells"
            )
        if num_nodes is None:
            raise ValueError("Cannot compute shape with unknown number of nodes")
        self._out_num_nodes = num_cells_vertical * num_cells_horizontal * num_nodes

    def call(self, x: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Compute region representation from spatial tensor.

        Assumes that all nodes have the same scope along the spatial axes.

        Args:
            x: Spatial input.
            kwargs: Remaining keyword arguments.

        Returns:
            A region representation of the input.
        """
        shape = tf.shape(x)
        return tf.reshape(x, [shape[0], 1, 1, self._out_num_nodes])

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
            ValueError: When shape cannot be computed.
        """
        num_batch, num_cells_vertical, num_cells_horizontal, num_nodes = input_shape
        if num_cells_horizontal is None:
            raise ValueError(
                "Cannot compute shape with unknown number of horizontal cells"
            )
        if num_cells_vertical is None:
            raise ValueError(
                "Cannot compute shape with unknown number of vertical cells"
            )
        if num_nodes is None:
            raise ValueError("Cannot compute shape with unknown number of nodes")
        return num_batch, 1, 1, num_cells_vertical * num_cells_horizontal * num_nodes
