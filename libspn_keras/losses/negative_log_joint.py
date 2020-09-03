import tensorflow as tf
from tensorflow import keras


class NegativeLogJoint(keras.losses.Loss):
    r"""
    Compute :math:`-\log(p(X,Y))`.

    Assumes that its input is :math:`\log(p(X|Y))` where Y is indexed on the second axis. This can
    be used for supervised generative learning with gradient-based optimizers or
    (hard) expectation maximization.
    """

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Compute generative loss for optimizing :math:`p(X,Y)`.

        Args:
            y_true: Sparse labels
            y_pred: Predicted logits

        Returns:
            A `Tensor` that describes  a generative supervised loss.
        """
        return -tf.gather(y_pred, tf.cast(y_true, tf.int32), axis=1)
