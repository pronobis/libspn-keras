from tensorflow import keras
import tensorflow as tf


class LogMarginalLikelihood(keras.metrics.Mean):
    """
    Computes log marginal :math:`1/N \sum \log(p(X))` assuming that the last layer of the SPN
    is a ``RootSum``. It ignores the ``y_true`` argument, as a target for :math:`Y` is absent in
    generative learning.
    """

    def __init__(self, name='log_marginal', **kwargs):
        super(LogMarginalLikelihood, self).__init__(name=name, **kwargs)

    def update_state(self, _, y_pred, sample_weight=None):
        values = tf.reduce_logsumexp(y_pred, axis=-1)
        return super(LogMarginalLikelihood, self).update_state(values, sample_weight=sample_weight)
