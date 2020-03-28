from tensorflow.keras import initializers
import tensorflow as tf


class EpsilonInverseFanIn(initializers.Initializer):
    """
    Initializes all values in a tensor with
    :math:`\epsilon K^{-1}`
    where :math:`K` is the dimension at ``axis``.

    This is particularly useful for (unweighted) hard EM learning and should generally be avoided
    otherwise.

    Args:
        axis: The axis for input nodes so that :math:`K^{-1}` is the inverse fan in. Usually,
            this is ``-2``.
        epsilon: A small non-zero constant
        dtype: dtype of initial values
    """

    def __init__(self, axis: int, epsilon: float = 1e-3, dtype=None):
        self.dtype = dtype
        self.axis = axis
        self.epsilon = epsilon

    def __call__(self, shape, dtype=None, partition_info=None):

        if dtype is None:
            dtype = self.dtype

        n = shape[self.axis]

        return tf.cast(tf.fill(shape, self.epsilon / n), dtype)

    def get_config(self):
        return {
            "dtype": self.dtype.name,
            "epsilon": self.epsilon,
            "axis": self.axis
        }
