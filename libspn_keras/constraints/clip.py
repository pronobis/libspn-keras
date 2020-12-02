from typing import Optional

import tensorflow as tf
from tensorflow.keras.constraints import Constraint


class Clip(Constraint):
    """
    Constraints the weights to be between ``min`` and ``max``.

    Args:
        min: Minimum clip value
        max: Maximum clip value
    """

    def __init__(self, min: float, max: Optional[float] = None):
        self.min = min
        self.max = max

    def __call__(self, w: tf.Tensor) -> tf.Tensor:
        """
        Clip input tensor to ``[min, max]``.

        Args:
            w: Weight Tensor.

        Returns:
            Clipped weight Tensor.
        """
        return tf.clip_by_value(w, clip_value_min=self.min, clip_value_max=self.max)

    def get_config(self) -> dict:
        """
        Obtain config.

        Returns:
            Key-value mapping with configuration of this constraint.
        """
        return dict(min=self.min, max=self.max)
