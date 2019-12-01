from tensorflow import keras
import tensorflow as tf

from libspn_keras.logspace import logspace_wrapper_initializer
from libspn_keras.math.logmatmul import logmatmul
from libspn_keras.math.hard_em_grads import logmatmul_hard_em_through_grads_from_accumulators, logmultiply_hard_em
from tensorflow import initializers

from libspn_keras.math.soft_em_grads import log_softmax_with_soft_em_grad


class RootSum(keras.layers.Layer):

    def __init__(
        self, return_weighted_child_logits=False, logspace_accumulators=False, hard_em_backward=False,
        initializer=initializers.Constant(1.0), soft_em_backward=False
    ):
        super(RootSum, self).__init__()
        self._return_weighted_child_logits = return_weighted_child_logits
        self._accumulators = self._num_nodes_in = None
        self._logspace_accumulators = logspace_accumulators
        self._hard_em_backward = hard_em_backward
        self._soft_em_backward = soft_em_backward
        self._initializer = initializer

        if hard_em_backward and logspace_accumulators:
            raise NotImplementedError("Cannot use both Hard EM gradients and logspace accumulators")

        if soft_em_backward and logspace_accumulators:
            raise NotImplementedError("Cannot use both Soft EM gradients and logspace accumulators")

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
                if self._return_weighted_child_logits:
                    return logmultiply_hard_em(tf.reshape(x, (-1, self._num_nodes_in)), self._accumulators)

                logmatmul_out = logmatmul_hard_em_through_grads_from_accumulators(
                    x, tf.reshape(self._accumulators, (1, 1, self._num_nodes_in, 1)))

                return tf.reshape(logmatmul_out, (-1, 1))

            log_weights_unnormalized = tf.math.log(log_weights_unnormalized)

        x_squeezed = tf.reshape(x, (-1, self._num_nodes_in))

        if self._soft_em_backward:
            log_weights_normalized = log_softmax_with_soft_em_grad(self._accumulators, axis=0)
        else:
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


