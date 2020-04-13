from tensorflow import keras
import tensorflow as tf

from libspn_keras.backprop_mode import BackpropMode, infer_logspace_accumulators
from libspn_keras.constraints.greater_equal_epsilon import GreaterEqualEpsilon
from libspn_keras.dimension_permutation import DimensionPermutation, infer_dimension_permutation
from libspn_keras.logspace import logspace_wrapper_initializer
from libspn_keras.math.logmatmul import logmatmul
from libspn_keras.math.hard_em_grads import \
    logmatmul_hard_em_through_grads_from_accumulators, logmultiply_hard_em
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras import constraints

from libspn_keras.math.soft_em_grads import log_softmax_from_accumulators_with_em_grad
import numpy as np


class RootSum(keras.layers.Layer):
    """
    Final sum of an SPN. Expects input to be in log-space and produces log-space output.

    Args:
        return_weighted_child_logits: If True, returns a weighted child log probability, which
            can be used for e.g. (Sparse)CategoricalCrossEntropy losses. If False, computes
            the weighted sum of the input, which effectively is the log probability of the
            distribution defined by the SPN.
        logspace_accumulators: If ``True``, accumulators will be represented in log-space which
            is typically used with ``BackpropMode.GRADIENT``. If ``False``, accumulators will be
            represented in linear space. Weights are computed by normalizing the accumulators
            per sum, so that we always end up with a normalized SPN. If ``None`` (default) it
            will be set to ``True`` for ``BackpropMode.GRADIENT`` and ``False`` otherwise.
        accumulator_initializer: Initializer for accumulator. If None, defaults to
            initializers.Constant(1.0)
        backprop_mode: Backpropagation mode. Can be either BackpropMode.GRADIENT,
            BackpropMode.HARD_EM, BackpropMode.SOFT_EM or BackpropMode.HARD_EM_UNWEIGHTED
        dimension_permutation: Dimension permutation. If DimensionPermutation.AUTO, the layer
            will try to infer it from the input tensors during the graph build phase. If needed,
            it can be changed to DimensionPermutation.SPATIAL for e.g. spatial SPNs or
            DimensionPermutation.REGIONS for dense SPNs.
        accumulator_regularizer: Regularizer for accumulator.
        linear_accumulator_constraint: Constraint for linear accumulators. Defaults to a
            constraint that ensures a minimum of a small positive constant. If
            logspace_accumulators is set to True, this constraint wil be ignored
        **kwargs: kwargs to pass on to the keras.Layer super class
    """
    def __init__(
        self, return_weighted_child_logits=True, logspace_accumulators=None,
        accumulator_initializer=None, backprop_mode=BackpropMode.GRADIENT,
        dimension_permutation=DimensionPermutation.AUTO, accumulator_regularizer=None,
        linear_accumulator_constraint=None, **kwargs
    ):
        super(RootSum, self).__init__(**kwargs)
        self.return_weighted_child_logits = return_weighted_child_logits
        self.accumulator_initializer = accumulator_initializer or initializers.Constant(1.0)
        self.logspace_accumulators = infer_logspace_accumulators(backprop_mode) \
            if logspace_accumulators is None else logspace_accumulators
        self.backprop_mode = backprop_mode
        self.dimension_permutation = dimension_permutation
        self.accumulator_regularizer = accumulator_regularizer
        self.linear_accumulator_constraint = \
            linear_accumulator_constraint or GreaterEqualEpsilon(1e-10)
        self.accumulators = self._num_nodes_in = self._inferred_dimension_permutation = None

        if backprop_mode != BackpropMode.GRADIENT and logspace_accumulators:
            raise NotImplementedError(
                "Logspace accumulators can only be used with BackpropMode.GRADIENT")

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self._inferred_dimension_permutation = infer_dimension_permutation(input_shape) \
            if self.dimension_permutation == DimensionPermutation.AUTO \
            else self.dimension_permutation

        if self._inferred_dimension_permutation == DimensionPermutation.SPATIAL:
            self._num_nodes_in = num_nodes_in = np.prod(input_shape[1:])
        else:
            num_scopes_in, num_decomps_in, _, num_nodes_in = input_shape

            self._num_nodes_in = num_nodes_in

            if num_scopes_in != 1 or num_decomps_in != 1:
                raise ValueError("Number of scopes and decomps must both be 1")

        initializer = self.accumulator_initializer
        accumulator_constraint = self.linear_accumulator_constraint
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
        if len(input_shape) == 2:
            num_batch, num_nodes_in = input_shape
        else:
            inferred_dimension_permutation = infer_dimension_permutation(input_shape) \
                if self.dimension_permutation == DimensionPermutation.AUTO \
                else self.dimension_permutation
            if inferred_dimension_permutation == DimensionPermutation.SPATIAL:
                num_batch, _, _, num_nodes_in = input_shape
            else:
                _, _, num_batch, num_nodes_in = input_shape
        if self.return_weighted_child_logits:
            return [num_batch, num_nodes_in]
        else:
            return [num_batch, 1]

    def get_config(self):
        config = dict(
            accumulator_initializer=initializers.serialize(self.accumulator_initializer),
            logspace_accumulators=self.logspace_accumulators,
            return_weighted_child_logits=self.return_weighted_child_logits,
            backprop_mode=self.backprop_mode,
            accumulator_regularizer=regularizers.serialize(self.accumulator_regularizer),
            linear_accumulator_constraint=constraints.serialize(self.linear_accumulator_constraint)
        )
        base_config = super(RootSum, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
