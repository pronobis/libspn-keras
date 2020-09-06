from typing import List, Optional, Tuple

import tensorflow as tf
from tensorflow import keras


class TemporalDenseProduct(keras.layers.Layer):
    r"""
    Computes 'temporal' dense products.

    This is used to connect an interface stack at :math:`t - 1` of a dynamic
    SPN with a template SPN at :math:`t`. Computes a product of all possible combinations of
    nodes along the last axis of the two incoming layers.

    Args:
        **kwargs: kwargs to pass on to the keras.Layer super class.
    """

    def __init__(self, **kwargs):
        self._num_nodes_out = None
        super(TemporalDenseProduct, self).__init__(**kwargs)

    def build(self, input_shape: List[Tuple[Optional[int], ...]]) -> None:
        """
        Build the internal components for this leaf layer.

        Args:
            input_shape: Shape of the input Tensor.

        Raises:
            ValueError: If shape cannot be determined.
        """
        a_shape, b_shape = input_shape
        if a_shape[-1] is None:
            raise ValueError("Unknown number of nodes in first tensor")
        if b_shape[-1] is None:
            raise ValueError("Unknown number of nodes in second tensor")
        self._num_nodes_out = a_shape[-1] * b_shape[-1]
        super(TemporalDenseProduct, self).build(input_shape)

    def call(self, x: List[tf.Tensor], **kwargs) -> tf.Tensor:
        """
        Compute temporal dense product.

        Multiplies every node at timestep t-1 with every node at timestep t, across all scopes
        and decompositions.

        Args:
            x: Spatial or Region tensor
            kwargs: Remaining keyword arguments.

        Returns:
            A Tensor with a dense temporal product.
        """
        # Split in list of tensors which will be added up using outer products
        a, b = x
        a_expanded = tf.expand_dims(a, axis=-1)
        b_expanded = tf.expand_dims(b, axis=-2)
        outer_dims = tf.shape(a_expanded)[:-2]
        out_shape = tf.concat([outer_dims, [self._num_nodes_out]], axis=0)
        return tf.reshape(a_expanded + b_expanded, out_shape)

    def compute_output_shape(
        self, input_shape: List[Tuple[Optional[int], ...]]
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
        a_shape, b_shape = input_shape
        num_batch, num_scopes_in, num_decomps, num_nodes_in_a = a_shape

        if num_nodes_in_a is None:
            raise ValueError(
                "Cannot compute output shape without knowing num nodes in first input"
            )

        if b_shape[-1] is None:
            raise ValueError(
                "Cannot compute output shape without knowing num nodes in second input"
            )

        return num_batch, num_scopes_in, num_decomps, int(num_nodes_in_a ** b_shape[-1])
