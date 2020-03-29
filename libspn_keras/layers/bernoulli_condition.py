from tensorflow import keras
import tensorflow as tf
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras import backend as K


class BernoulliCondition(keras.layers.Layer):
    """
    Applies ``tf.cond`` element-wise where its condition argument is a boolean drawn from a
    Bernoulli distribution with :math:`p=` ``rate``. This can be useful to compute probabilistic
    dropout at input layers. This layer expects two input tensors.

    Args:
        rate (float): Rate at which to take elements of the first input tensor.
        seed: Random seed
        **kwargs: Parameters passed on to ``tf.keras.Layer``
    """

    def __init__(self, rate, seed=None, **kwargs):
        super(BernoulliCondition, self).__init__(**kwargs)
        self.rate = min(1., max(0., rate))
        self.seed = seed

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

    def get_config(self):
        config = dict(
            seed=self.seed,
            rate=self.rate,
        )
        base_config = super(BernoulliCondition, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
