from tensorflow import keras
import tensorflow as tf

from libspn_keras.backprop_mode import BackpropMode
from libspn_keras.contraints.greater_than_epsilon import GreaterThanEpsilon
from libspn_keras.dimension_permutation import DimensionPermutation, infer_dimension_permutation
from libspn_keras.logspace import logspace_wrapper_initializer
from libspn_keras.math.logmatmul import logmatmul
from libspn_keras.math.hard_em_grads import \
    logmatmul_hard_em_through_grads_from_accumulators, logmultiply_hard_em
from tensorflow import initializers

from libspn_keras.math.soft_em_grads import log_softmax_from_accumulators_with_em_grad
import numpy as np


class RootSum(keras.layers.Layer):

    def __init__(
        self, return_weighted_child_logits=True, logspace_accumulators=False,
        accumulator_initializer=None, backprop_mode=BackpropMode.GRADIENT,
        dimension_permutation=DimensionPermutation.AUTO, accumulator_regularizer=None,
        accumulator_constraint=GreaterThanEpsilon(1e-10), **kwargs
    ):
        super(RootSum, self).__init__(**kwargs)
        self.return_weighted_child_logits = return_weighted_child_logits
        self.accumulator_initializer = accumulator_initializer or initializers.Constant(1.0)
        self.logspace_accumulators = logspace_accumulators
        self.backprop_mode = backprop_mode
        self.dimension_permutation = dimension_permutation
        self.accumulator_regularizer = accumulator_regularizer
        self.accumulator_constraint = accumulator_constraint
        self.accumulators = self._num_nodes_in = self._inferred_dimension_permutation = None

        if backprop_mode != BackpropMode.GRADIENT and logspace_accumulators:
            raise NotImplementedError(
                "Logspace accumulators can only be used with BackpropMode Gradient")

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self._inferred_dimension_permutation = infer_dimension_permutation(input_shape) \
            if self.dimension_permutation == DimensionPermutation.AUTO \
            else self.dimension_permutation

        if self._inferred_dimension_permutation == DimensionPermutation.BATCH_FIRST:
            self._num_nodes_in = num_nodes_in = np.prod(input_shape[1:])
        else:
            num_scopes_in, num_decomps_in, _, num_nodes_in = input_shape

            self._num_nodes_in = num_nodes_in

            if num_scopes_in != 1 or num_decomps_in != 1:
                raise ValueError("Number of scopes and decomps must both be 1")

        initializer = self.accumulator_initializer
        accumulator_constraint = self.accumulator_constraint
        if self.logspace_accumulators:
            initializer = logspace_wrapper_initializer(initializer)
            accumulator_constraint = None

        self.accumulators = self.add_weight(
            name='weights', shape=(num_nodes_in,), initializer=initializer,
            regularizer=self.accumulator_regularizer, constraint=accumulator_constraint
        )

    def call(self, x):
        log_weights_unnormalized = self.accumulators
        x_squeezed = tf.reshape(x, (-1, self._num_nodes_in))
        if not self.logspace_accumulators:

            if self.backprop_mode in [BackpropMode.HARD_EM, BackpropMode.HARD_EM_UNWEIGHTED]:
                if self.return_weighted_child_logits:
                    return logmultiply_hard_em(x_squeezed, self.accumulators)

                logmatmul_out = logmatmul_hard_em_through_grads_from_accumulators(
                    tf.reshape(x, (1, 1, -1, self._num_nodes_in)),
                    tf.reshape(self.accumulators, (1, 1, self._num_nodes_in, 1)),
                    unweighted=self.backprop_mode == BackpropMode.HARD_EM_UNWEIGHTED
                )
                return tf.reshape(logmatmul_out, (-1, 1))

            log_weights_unnormalized = tf.math.log(log_weights_unnormalized)

        if self.backprop_mode == BackpropMode.EM:
            log_weights_normalized = log_softmax_from_accumulators_with_em_grad(
                self.accumulators, axis=0)
        else:
            log_weights_normalized = tf.nn.log_softmax(log_weights_unnormalized, axis=0)

        if self.return_weighted_child_logits:
            return tf.expand_dims(log_weights_normalized, axis=0) + x_squeezed
        else:
            return logmatmul(
                x_squeezed, tf.expand_dims(log_weights_normalized, axis=1))

    def compute_output_shape(self, input_shape):
        num_batch, num_nodes_in = input_shape
        if self.return_weighted_child_logits:
            return [num_batch, num_nodes_in]
        else:
            return [num_batch, 1]

    def get_config(self):
        config = dict(
            accumulator_initializer=initializers.serialize(self.accumulator_initializer),
            logspace_accumulators=self.logspace_accumulators,
            return_weighted_child_logits=self.return_weighted_child_logits,
            backprop_mode=self.backprop_mode
        )
        base_config = super(RootSum, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
