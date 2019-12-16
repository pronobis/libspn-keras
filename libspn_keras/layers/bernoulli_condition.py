from tensorflow import keras
import tensorflow as tf
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras import backend as K


class BernoulliCondition(keras.layers.Layer):

    def __init__(self, rate, seed=None, **kwargs):
        super(BernoulliCondition, self).__init__(**kwargs)
        self.rate = min(1., max(0., rate))
        self.seed = seed
        self.supports_masking = True

    def call(self, inputs, training=None):
        if_val, else_val = inputs
        if training is None:
            training = K.learning_phase()

        if_shape = tf.shape(if_val)

        def cond_out():
            cond = tf.greater(self.rate, tf.random.uniform(shape=if_shape))
            return tf.reshape(tf.where(cond, if_val, else_val), shape=if_shape)

        output = tf_utils.smart_cond(
            training, cond_out, lambda: tf.identity(if_val))
        return output

    def compute_output_shape(self, input_shape):
        return input_shape
