from typing import Optional

import tensorflow as tf
from tensorflow.keras.constraints import Constraint


class LogNormalize(Constraint):
    """
    Normalizes log-space weights.

    Args:
        axis (int): Axis along whichto normalize
    """

    def __init__(self, axis: Optional[int] = None):
        self.axis = -2 if axis is None else axis

    def __call__(self, w: tf.Tensor) -> tf.Tensor:
        """
        Normalize weight in log-space.

        Args:
            w: Unnormalized weight Tensor.

        Returns:
            Log-normalized weight Tensor.
        """
        return tf.nn.log_softmax(w, axis=self.axis)

    def get_config(self) -> dict:
        """
        Obtain config.

        Returns:
            Key-value mapping with configuration of this constraint.
        """
        return dict(axis=self.axis)
