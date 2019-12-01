from libspn_keras.logspace import logspace_wrapper_initializer
from libspn_keras.math.logmatmul import logmatmul
from libspn_keras.math.hard_em_grads import logmatmul_hard_em_through_grads_from_accumulators
from tensorflow import keras
from tensorflow import initializers
import tensorflow as tf

from libspn_keras.math.soft_em_grads import log_softmax_with_soft_em_grad


class ScopeWiseSum(keras.layers.Layer):

    def __init__(
        self, num_sums, logspace_accumulators=False,
        initializer=initializers.Constant(1), hard_em_backward=False, soft_em_backward=False
    ):
        super(ScopeWiseSum, self).__init__()
        self._num_sums = num_sums
        self._logspace_accumulators = logspace_accumulators
        self._num_decomps = self._num_scopes = self._accumulators = None
        self._initializer = initializer
        self._hard_em_backward = hard_em_backward
        self._soft_em_backward = soft_em_backward

        if soft_em_backward and hard_em_backward:
            raise ValueError("Cannot have both soft and hard em backward passes enabled at the same time")

    @property
    def num_sums(self):
        return self._num_sums

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self._num_scopes, self._num_decomps, _, num_nodes_in = input_shape

        weights_shape = (self._num_scopes, self._num_decomps, num_nodes_in, self._num_sums)

        initializer = self._initializer
        if self._logspace_accumulators:
            initializer = logspace_wrapper_initializer(self._initializer)

        self._accumulators = self.add_weight(
            name='sum_weights', shape=weights_shape, initializer=initializer)
        super(ScopeWiseSum, self).build(input_shape)

    def call(self, x):

        # dlogR / dW == dlogR / dlogW * dlogW / dW vs. dlogR / dlogW == dlogR / dlogW
        # dlogR / dW == dlogR / dlogW * 1 / W
        # dlogR / dW == dlogR / dS * w * C
        # dlogR / dW == dlogR / dlogS * w * C

        # dlogR / dW == dlogR / dR * dR / dS * C * w
        #            == dlogR / dW * w
        #            == dlogR / dlogW * dlogW / dW * w
        #            == dlogR / dlogW
        log_weights_unnormalized = self._accumulators

        if not self._logspace_accumulators and self._hard_em_backward:
            return logmatmul_hard_em_through_grads_from_accumulators(x, self._accumulators)\

        if not self._logspace_accumulators and self._soft_em_backward:
            log_weights_normalized = log_softmax_with_soft_em_grad(self._accumulators, axis=2)
        elif not self._logspace_accumulators:
            log_weights_normalized = tf.nn.log_softmax(tf.math.log(log_weights_unnormalized), axis=2)
        else:
            log_weights_normalized = tf.nn.log_softmax(log_weights_unnormalized, axis=2)

        return logmatmul(x, log_weights_normalized)

    def compute_output_shape(self, input_shape):
        num_scopes, num_decomps, num_batch, _ = input_shape
        return num_scopes, num_decomps, num_batch, self._num_sums

