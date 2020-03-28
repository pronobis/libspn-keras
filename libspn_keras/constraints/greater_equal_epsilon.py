from tensorflow.keras.constraints import Constraint
import tensorflow as tf


class GreaterEqualEpsilon(Constraint):
    """
    Constraints the weight to be greater than or equal to epsilon.

    Args:
        epsilon: Constant, usually small non-zero
    """
    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = epsilon

    def __call__(self, w):
        return tf.maximum(w, self.epsilon)

    def get_config(self):
        return dict(epsilon=self.epsilon)
