from tensorflow import keras
import tensorflow as tf


class LogMarginal(keras.metrics.Mean):

    def __init__(self, name='log_likelihood', **kwargs):
        super(LogMarginal, self).__init__(name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        values = tf.reduce_logsumexp(y_pred, axis=-1)
        return super(LogMarginal, self).update_state(values, sample_weight=sample_weight)
