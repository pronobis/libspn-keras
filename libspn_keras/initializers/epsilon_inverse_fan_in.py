from typing import Optional, Tuple

import tensorflow as tf
from tensorflow.keras import initializers


class EpsilonInverseFanIn(initializers.Initializer):
    r"""
    Initializes all values in a tensor with :math:`\epsilon K^{-1}`.

    Where :math:`K` is the dimension at ``axis``.

    This is particularly useful for (unweighted) hard EM learning and should generally be avoided
    otherwise.

    Args:
        axis: The axis for input nodes so that :math:`K^{-1}` is the inverse fan in. Usually,
            this is ``-2``.
        epsilon: A small non-zero constant
    """

    def __init__(
        self, axis: int = -2, epsilon: float = 1e-4,
    ):
        self.axis = axis
        self.epsilon = epsilon

    def __call__(
        self, shape: Tuple[int, ...], dtype: Optional[tf.DType] = None
    ) -> tf.Tensor:
        """
        Compute initializations along the given axis.

        Args:
            shape: Shape of Tensor to initialize.
            dtype: DType of Tensor to initialize.

        Returns:
            Initial value.
        """
        n = shape[self.axis]
        return tf.cast(tf.fill(shape, self.epsilon / n), dtype)

    def get_config(self) -> dict:
        """
        Obtain the config.

        Returns:
            Key-value mapping with the configuration of this initializer.
        """
        return {"epsilon": self.epsilon, "axis": self.axis}
