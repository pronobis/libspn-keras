from tensorflow import keras
import tensorflow as tf


class NegativeLogJoint(keras.losses.Loss):
    """
    Computes :math:`-\log(p(X,Y))` assuming that its input is :math:`-\log(p(X|Y))` where Y is indexed
    on the second axis. This can be used for supervised generative learning with gradient-based
    optimizers or (hard) expectation maximization.
    """

    def call(self, y_true, y_pred):
        """
        Computes generative loss for optimizing :math:`p(X,Y)`

        Args:
            y_true: Sparse labels
            y_pred: Predicted logits

        Returns:
            A `Tensor` that describes  a generative supervised loss.
        """

        return -tf.gather(y_pred, tf.cast(y_true, tf.int32), axis=1)
