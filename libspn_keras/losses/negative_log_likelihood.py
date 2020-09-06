import tensorflow as tf
from tensorflow import keras


class NegativeLogLikelihood(keras.losses.Loss):
    r"""
    Marginalize logits over last dimension so that it computes :math:`-\log(p(X))`.

    This can be used for unsupervised generative learning.
    """

    def call(self, _: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        r"""
        Compute generative loss for optimizing :math:`\log(p(X))`.

        Args:
            _: True labels, ignored, but still provided to preserve Keras compatibility
            y_pred: Predicted logits (or already marginalized region_graph_root value)

        Returns:
            A `Tensor` that describes a generative unsupervised loss.
        """
        return -tf.reduce_logsumexp(y_pred, axis=1)
