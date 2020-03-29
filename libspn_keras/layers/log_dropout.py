from tensorflow import keras
import tensorflow as tf
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras import backend as K


class LogDropout(keras.layers.Layer):
    """
    Log dropout layer. Applies dropout in log-space. Should not precede product layers in an
    SPN, since their scope probability then potentially becomes -inf, resulting in NaN-values
    during training.

    Args:
        rate: Rate at which to randomly dropout inputs.
        noise_shape: Shape of dropout noise tensor
        seed: Random seed
        **kwargs: kwargs to pass on to the keras.Layer super class
    """
    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):

        super(LogDropout, self).__init__(**kwargs)
        self.rate = min(1., max(0., rate))
        self.noise_shape = noise_shape
        self.seed = seed
        self.supports_masking = True

    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape

        symbolic_shape = K.shape(inputs)
        noise_shape = [symbolic_shape[axis] if shape is None else shape
                       for axis, shape in enumerate(self.noise_shape)]
        return tuple(noise_shape)

    def call(self, inputs, training=None):
        if training is None:
            training = K.learning_phase()

        def dropped_inputs():
            keep_tensor = tf.greater(
                tf.random.uniform(shape=tf.shape(inputs), seed=self.seed), self.rate)
            return tf.reshape(tf.where(keep_tensor, inputs, float('-inf')), shape=tf.shape(inputs))

        if self.rate == 0.0:
            return tf.identity(inputs)

        output = tf_utils.smart_cond(
            training, dropped_inputs, lambda: tf.identity(inputs))
        return output

    def get_config(self):
        config = {'rate': self.rate,
                  'noise_shape': self.noise_shape,
                  'seed': self.seed}
        base_config = super(LogDropout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape