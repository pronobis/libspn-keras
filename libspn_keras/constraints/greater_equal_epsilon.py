import tensorflow as tf
from tensorflow.keras.constraints import Constraint


class GreaterEqualEpsilon(Constraint):
    """
    Constraints the weight to be greater than or equal to epsilon.

    Args:
        epsilon: Constant, usually small non-zero
    """

    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = epsilon

    def __call__(self, w: tf.Tensor) -> tf.Tensor:
        """
        Lower-clip input tensor to ``epsilon``.

        Args:
            w: Weight Tensor.

        Returns:
            Clipped weight Tensor.
        """
        return tf.maximum(w, self.epsilon)

    def get_config(self) -> dict:
        """
        Obtain config.

        Returns:
            Key-value mapping with configuration of this constraint.
        """
        return dict(epsilon=self.epsilon)
