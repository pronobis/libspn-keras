from tensorflow.keras.constraints import Constraint
import tensorflow as tf


class GreaterThanEpsilon(Constraint):

    def __init__(self, epsilon=1e-8):
        """
        Constraints the weight to be greater than some small non zero epsilon

        Args:
            epsilon: Small non-zero constant
        """
        self.epsilon = epsilon

    def __call__(self, w):
        return tf.maximum(w, self.epsilon)

    def get_config(self):
        return dict(epsilon=self.epsilon)
