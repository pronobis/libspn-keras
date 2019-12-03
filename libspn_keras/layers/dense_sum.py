from libspn_keras.backprop_mode import BackpropMode
from libspn_keras.logspace import logspace_wrapper_initializer
from libspn_keras.math.logmatmul import logmatmul
from libspn_keras.math.hard_em_grads import logmatmul_hard_em_through_grads_from_accumulators
from tensorflow import keras
from tensorflow import initializers
import tensorflow as tf

from libspn_keras.math.soft_em_grads import log_softmax_from_accumulators_with_em_grad


class DenseSum(keras.layers.Layer):

    def __init__(
        self, num_sums, logspace_accumulators=False,
        accumulator_initializer=initializers.Constant(1),
        backprop_mode=BackpropMode.GRADIENT,
    ):
        super(DenseSum, self).__init__()
        self.num_sums = num_sums
        self.logspace_accumulators = logspace_accumulators
        self.accumulator_initializer = accumulator_initializer
        self.backprop_mode = backprop_mode
        self._num_decomps = self._num_scopes = self._accumulators = None

        if backprop_mode != BackpropMode.GRADIENT and logspace_accumulators:
            raise ValueError("Logspace accumulators are only supported for gradient backprop mode")

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self._num_scopes, self._num_decomps, _, num_nodes_in = input_shape

        weights_shape = (self._num_scopes, self._num_decomps, num_nodes_in, self.num_sums)

        initializer = self.accumulator_initializer
        if self.logspace_accumulators:
            initializer = logspace_wrapper_initializer(self.accumulator_initializer)

        self._accumulators = self.add_weight(
            name='sum_weights', shape=weights_shape, initializer=initializer)
        super(DenseSum, self).build(input_shape)

    def call(self, x):

        # TODO make the control flow more readable e.g. implement _call_backprop_gradient, _call_backprop_em
        # dlogR / dW == dlogR / dlogW * dlogW / dW vs. dlogR / dlogW == dlogR / dlogW
        # dlogR / dW == dlogR / dlogW * 1 / W
        # dlogR / dW == dlogR / dS * w * C
        # dlogR / dW == dlogR / dlogS * w * C

        # dlogR / dW == dlogR / dR * dR / dS * C * w
        #            == dlogR / dW * w
        #            == dlogR / dlogW * dlogW / dW * w
        #            == dlogR / dlogW
        log_weights_unnormalized = self._accumulators

        if not self.logspace_accumulators and self.backprop_mode == BackpropMode.HARD_EM:
            return logmatmul_hard_em_through_grads_from_accumulators(x, self._accumulators)

        if not self.logspace_accumulators and self.backprop_mode == BackpropMode.EM:
            log_weights_normalized = log_softmax_from_accumulators_with_em_grad(self._accumulators, axis=2)
        elif not self.logspace_accumulators:
            log_weights_normalized = tf.nn.log_softmax(tf.math.log(log_weights_unnormalized), axis=2)
        else:
            log_weights_normalized = tf.nn.log_softmax(log_weights_unnormalized, axis=2)

        return logmatmul(x, log_weights_normalized)

    def compute_output_shape(self, input_shape):
        num_scopes, num_decomps, num_batch, _ = input_shape
        return num_scopes, num_decomps, num_batch, self.num_sums

    def get_config(self):
        config = dict(
            num_sums=self.num_sums,
            accumulator_initializer=initializers.serialize(self.accumulator_initializer),
            logspace_accumulators=self.logspace_accumulators,
            backprop_mode=self.backprop_mode
        )
        base_config = super(DenseSum, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
