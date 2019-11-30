from libspn_keras.logspace import logspace_wrapper_initializer
from libspn_keras.math.logmatmul import logmatmul, logmatmul_hard_em_through_grads_from_accumulators
from tensorflow import keras
from tensorflow import initializers
import tensorflow as tf


class ParallelScopeSum(keras.layers.Layer):

    def __init__(
        self, num_sums, logspace_accumulators=False,
        initializer=initializers.Constant(1), hard_em_backward=False
    ):
        super(ParallelScopeSum, self).__init__()
        self._num_sums = num_sums
        self._logspace_accumulators = logspace_accumulators
        self._num_decomps = self._num_scopes = self._accumulators = None
        self._initializer = initializer
        self._hard_em_backward = hard_em_backward

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
        super(ParallelScopeSum, self).build(input_shape)

    def call(self, x):

        log_weights_unnormalized = self._accumulators

        if not self._logspace_accumulators:
            if self._hard_em_backward:
                return logmatmul_hard_em_through_grads_from_accumulators(x, self._accumulators)

            log_weights_unnormalized = tf.math.log(log_weights_unnormalized)

        log_weights_normalized = tf.nn.log_softmax(log_weights_unnormalized, axis=2)

        return logmatmul(x, log_weights_normalized)

    def compute_output_shape(self, input_shape):
        num_scopes, num_decomps, num_batch, _ = input_shape
        return num_scopes, num_decomps, num_batch, self._num_sums

