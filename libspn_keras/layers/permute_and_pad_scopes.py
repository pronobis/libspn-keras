from typing import List, Optional, Tuple, Union

import numpy
import tensorflow as tf
from tensorflow import keras


class PermuteAndPadScopes(keras.layers.Layer):
    """
    Permutes and pads scopes, usually applied after a ``FlatToRegions`` and a ``BaseLeaf`` layer.

    Padding can be used to ensure uniform dimension sizes across scopes, decomps and nodes in the
    layer stack. Padding is achieved by using ``-1`` for scope indices in ``permutations``.

    Args:
        num_decomps: Number of decompositions
        permutations: If None, permutations must be specified later
        **kwargs: kwargs to pass on to the keras.Layer superclass.
    """

    def __init__(
        self,
        permutations: Optional[Union[List[List[int]], numpy.ndarray]] = None,
        **kwargs
    ):
        super(PermuteAndPadScopes, self).__init__(**kwargs)
        self.permutations = permutations

    def call(self, x: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Permute and pad scopes.

        Padding is achieved by using ``-1`` for scope indices in ``permutations``/.

        Args:
            x: A Region input tensor
            kwargs: Remaining keyword arguments.

        Returns:
            Tensor with scopes permuted and padded.
        """
        decomps_first = tf.transpose(x, (2, 1, 0, 3))
        decomps_first_padded = tf.pad(decomps_first, [[0, 0], [1, 0], [0, 0], [0, 0]])
        gather_indices = tf.convert_to_tensor(self.permutations) + 1
        permuted = tf.gather(decomps_first_padded, gather_indices, axis=1, batch_dims=1)
        return tf.transpose(permuted, (2, 1, 0, 3))

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
        return input_shape

    def get_config(self) -> dict:
        """
        Obtain a key-value representation of the layer config.

        Returns:
            A dict holding the configuration of the layer.
        """
        config = dict(permutations=self.permutations)
        base_config = super(PermuteAndPadScopes, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
