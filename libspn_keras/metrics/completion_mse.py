from tensorflow import keras
import tensorflow as tf


class CompletionMSE(keras.metrics.Mean):

    def __init__(self, name='completion_mse', reduction_axes=(1, 2, 3), **kwargs):
        super(CompletionMSE, self).__init__(name=name, **kwargs)
        self.reduction_axes = reduction_axes

    def update_state(self, y_target, y_pred, sample_weight=None):
        squared_diff = tf.math.squared_difference(y_target, y_pred)
        mse = tf.reduce_mean(squared_diff, axis=self.reduction_axes)
        return super(CompletionMSE, self).update_state(mse, sample_weight=sample_weight)
