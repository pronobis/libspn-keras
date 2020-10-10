from typing import Optional, Tuple

import tensorflow as tf
from tensorflow.keras import initializers
from tensorflow_probability import distributions


class Dirichlet(initializers.Initializer):
    r"""
    Initializes all values in a tensor with :math:`Dir(\alpha)`.

    Args:
        axis: The axis over which to sample from a :math:`Dir(\alpha)`.
        alpha: The :math:`\alpha` parameter of the Dirichlet distribution.
            If a scalar, this is broadcast along the given axis.
    """

    def __init__(
        self, axis: int = -2, alpha: float = 0.1,
    ):
        self.axis = axis
        self.alpha = alpha

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
        axis = (len(shape) + self.axis) % len(shape)

        alpha_as_tensor = tf.convert_to_tensor(self.alpha)
        alpha = (
            tf.tile([alpha_as_tensor], [shape[axis]])
            if tf.size(alpha_as_tensor) == 1
            else alpha_as_tensor
        )
        dirichlet_sample = distributions.Dirichlet(concentration=alpha).sample(
            [dim for i, dim in enumerate(shape) if i != axis]
        )

        perm = [
            i if i < axis else len(shape) - 1 if i == axis else i - 1
            for i in range(len(shape))
        ]
        return tf.cast(tf.transpose(dirichlet_sample, perm), dtype)

    def get_config(self) -> dict:
        """
        Obtain the config.

        Returns:
            Key-value mapping with the configuration of this initializer.
        """
        return {"alpha": self.alpha, "axis": self.axis}
