import tensorflow as tf
from tensorflow.keras.constraints import Constraint


class GreaterEqualEpsilonNormalized(Constraint):
    """
    Constraints the weight to be greater than or equal to epsilon and then normalizes.

    Args:
        epsilon: Constant, usually small non-zero
    """

    def __init__(self, epsilon: float = 1e-10, axis: int = -2):
        self.epsilon = epsilon
        self.axis = axis

    def __call__(self, w: tf.Tensor) -> tf.Tensor:
        """
        Lower-clip input tensor to ``epsilon`` and then normalize.

        Args:
            w: Weight Tensor.

        Returns:
            Clipped weight Tensor.
        """
        clipped = tf.maximum(
            w, self.epsilon / tf.cast(tf.shape(w)[self.axis], tf.float32)
        )
        return clipped / tf.reduce_sum(clipped, axis=self.axis, keepdims=True)

    def get_config(self) -> dict:
        """
        Obtain config.

        Returns:
            Key-value mapping with configuration of this constraint.
        """
        return dict(epsilon=self.epsilon, axis=self.axis)
