from tensorflow.keras.constraints import Constraint
import tensorflow as tf


class GreaterThanEpsilon(Constraint):
    """Constraints the weight to be greater than some small non zero epsilon"""
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def __call__(self, w):
        return tf.maximum(w, self.epsilon)

    def get_config(self):
        return dict(epsilon=self.epsilon)
