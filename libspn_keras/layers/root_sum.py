from tensorflow import keras
import tensorflow as tf

from libspn_keras.logspace import logspace_wrapper_initializer
from libspn_keras.math.logmatmul import logmatmul, logmatmul_hard_em_through_grads_from_accumulators
from tensorflow import initializers


class RootSum(keras.layers.Layer):

    def __init__(
        self, return_weighted_child_logits=False, logspace_accumulators=False, hard_em_backward=False,
        initializer=initializers.Constant(1.0)
    ):
        super(RootSum, self).__init__()
        self._return_weighted_child_logits = return_weighted_child_logits
        self._accumulators = self._num_nodes_in = None
        self._logspace_accumulators = logspace_accumulators
        self._hard_em_backward = hard_em_backward
        self._initializer = initializer

        if hard_em_backward and logspace_accumulators:
            raise NotImplementedError("Cannot use both Hard EM gradients and logspace accumulators")

        if hard_em_backward and return_weighted_child_logits:
            raise NotImplementedError("Cannot use both Hard EM gradients and weighted child logits")

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        num_scopes_in, num_decomps_in, _, num_nodes_in = input_shape

        self._num_nodes_in = num_nodes_in

        if num_scopes_in != 1 or num_decomps_in != 1:
            raise ValueError("Number of scopes and decomps must both be 1")

        initializer = self._initializer
        if self._logspace_accumulators:
            initializer = logspace_wrapper_initializer(initializer)

        self._accumulators = self.add_weight(
            name='weights', shape=(num_nodes_in,), initializer=initializer)

    def call(self, x):

        log_weights_unnormalized = self._accumulators
        if not self._logspace_accumulators:
            if self._hard_em_backward:
                logmatmul_out = logmatmul_hard_em_through_grads_from_accumulators(
                    x, tf.reshape(self._accumulators, (1, 1, self._num_nodes_in, 1)))

                return tf.reshape(logmatmul_out, (-1, 1))

            log_weights_unnormalized = tf.math.log(log_weights_unnormalized)

        x_squeezed = tf.reshape(x, (-1, self._num_nodes_in))

        log_weights_normalized = tf.nn.log_softmax(log_weights_unnormalized, axis=0)

        if self._return_weighted_child_logits:
            return tf.expand_dims(log_weights_normalized, axis=0) + x_squeezed
        else:
            return logmatmul(
                x_squeezed, tf.expand_dims(log_weights_normalized, axis=1))

    def compute_output_shape(self, input_shape):
        num_batch, num_nodes_in = input_shape
        if self._return_weighted_child_logits:
            return [num_batch, num_nodes_in]
        else:
            return [num_batch, 1]


