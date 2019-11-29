from tensorflow import keras
import tensorflow as tf

from libspn_keras.logspace import logspace_wrapper_initializer
from libspn_keras.math.logmatmul import logmatmul
from tensorflow import initializers


class RootSum(keras.layers.Layer):

    def __init__(
        self, return_weighted_child_logits=False, logspace_accumulators=False, hard_em_grad=False,
        initializer=initializers.Constant(1.0)
    ):
        super(RootSum, self).__init__()
        self._return_weighted_child_logits = return_weighted_child_logits
        self._accumulators = self._num_nodes_in = None
        self._accumulators_logspace = logspace_accumulators
        self._hard_em_grad = hard_em_grad
        self._initializer = initializer

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        num_scopes_in, num_decomps_in, _, num_nodes_in = input_shape

        self._num_nodes_in = num_nodes_in

        if num_scopes_in != 1 or num_decomps_in != 1:
            raise ValueError("Number of scopes and decomps must both be 1")

        initializer = self._initializer
        if self._accumulators_logspace:
            initializer = logspace_wrapper_initializer(initializer)

        self._accumulators = self.add_weight(
            name='weights', shape=(num_nodes_in,), initializer=initializer)

    def call(self, x):

        x_squeezed = tf.reshape(x, (-1, self._num_nodes_in))
        log_weights_unnormalized = self._accumulators
        if not self._accumulators_logspace:
            log_weights_unnormalized = tf.math.log(log_weights_unnormalized)
        log_weights_normalized = tf.nn.log_softmax(log_weights_unnormalized, axis=0)

        if self._return_weighted_child_logits:
            return tf.expand_dims(log_weights_normalized, axis=0) + x_squeezed
        else:
            return logmatmul(x_squeezed, tf.expand_dims(log_weights_normalized, axis=1))

    def compute_output_shape(self, input_shape):
        *_, num_nodes_in = input_shape
        if self._return_weighted_child_logits:
            return [None, num_nodes_in]
        else:
            return [None, 1]


