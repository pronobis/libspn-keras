from typing import Optional, Tuple

import tensorflow as tf
from tensorflow.keras import initializers


class Equidistant(initializers.Initializer):
    """
    Initializer that generates tensors where the last axis is initialized with 'equidistant' values.

    Args:
      minval: A python scalar or a scalar tensor. Lower bound of the range
        of random values to generate.
      maxval: A python scalar or a scalar tensor. Upper bound of the range
        of random values to generate.  Defaults to 1 for float types.
    """

    def __init__(self, minval: float = 0.0, maxval: float = 1.0):
        self.minval = minval
        self.maxval = maxval

    def __call__(
        self, shape: Tuple[Optional[int], ...], dtype: Optional[tf.DType] = None
    ):
        """
        Compute equidistant initializations on the last axis.

        Args:
            shape: Shape of Tensor to initialize.
            dtype: DType of Tensor to initialize.

        Returns:
            Initial value.

        Raises:
            ValueError: If shape cannot be determined.
        """
        rank = len(shape)
        last_dim = shape[-1]
        if last_dim is None:
            raise ValueError("Cannot compute Equidistant with unknown last dimension")
        linspace = tf.reshape(
            tf.linspace(self.minval, self.maxval, num=last_dim),
            [1] * (rank - 1) + [last_dim],
        )
        return tf.cast(
            tf.tile(linspace, tf.concat([shape[:-1], [1]], axis=0)), dtype=dtype
        )

    def get_config(self) -> dict:
        """
        Obtain config.

        Returns:
            Key-value mapping of the configuration.
        """
        return {"minval": self.minval, "maxval": self.maxval}
