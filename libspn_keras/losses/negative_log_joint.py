from tensorflow import keras
import tensorflow as tf


class NegativeLogJoint(keras.losses.Loss):

    def call(self, y_true, y_pred):
        """

        Args:
            y_true: True sparse labels
            y_pred: Predicted logits

        Returns:
            A `Tensor` that describes a generative supervised loss.
        """

        return -tf.gather(y_pred, tf.cast(y_true, tf.int32), axis=1)
