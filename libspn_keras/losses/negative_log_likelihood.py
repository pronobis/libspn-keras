from tensorflow import keras
import tensorflow as tf


class NegativeLogMarginal(keras.losses.Loss):
    """
    Marginalizes over last dimension
    """

    def call(self, _, y_pred):
        """

        Args:
            y_true: True labels, ignored, but still provided to preserve Keras compatibility
            y_pred: Predicted logits (or already marginalized root value)

        Returns:
            A `Tensor` that describes a generative unsupervised loss.
        """

        return -tf.reduce_logsumexp(y_pred, axis=1)
