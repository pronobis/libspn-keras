from tensorflow import keras
import tensorflow as tf


class NegativeLogMarginal(keras.losses.Loss):
    """
    Marginalizes logits over last dimension so that it computes :math:`-\log(p(X))`. This can be
    used for unsupervised generative learning.
    """

    def call(self, _, y_pred):
        """
        Computes generative loss for optimizing :math:`\log(p(X))`

        Args:
            _: True labels, ignored, but still provided to preserve Keras compatibility
            y_pred: Predicted logits (or already marginalized region_graph_root value)

        Returns:
            A `Tensor` that describes a generative unsupervised loss.
        """

        return -tf.reduce_logsumexp(y_pred, axis=1)
