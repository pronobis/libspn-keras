from libspn_keras.backprop_mode import BackpropMode, infer_logspace_accumulators
from libspn_keras.constraints.greater_equal_epsilon import GreaterEqualEpsilon
from libspn_keras.logspace import logspace_wrapper_initializer
from libspn_keras.math.logmatmul import logmatmul
from libspn_keras.math.hard_em_grads import logmatmul_hard_em_through_grads_from_accumulators
from libspn_keras.math.soft_em_grads import log_softmax_from_accumulators_with_em_grad
from tensorflow import keras
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras import constraints
import tensorflow as tf


class DenseSum(keras.layers.Layer):
    """
    Computes densely connected sums per scope and decomposition. Expects incoming ``Tensor`` to be of
    shape [num_scopes, num_decomps, num_batch, num_nodes]. If your input is passed through a
    ``FlatToRegions`` layer this is already taken care of.

    Args:
        num_sums: Number of sums per scope
        logspace_accumulators: If ``True``, accumulators will be represented in log-space which
            is typically used with ``BackpropMode.GRADIENT``. If ``False``, accumulators will be
            represented in linear space. Weights are computed by normalizing the accumulators
            per sum, so that we always end up with a normalized SPN. If ``None`` (default) it
            will be set to ``True`` for ``BackpropMode.GRADIENT`` and ``False`` otherwise.
        accumulator_initializer: Initializer for accumulator. Will automatically be converted
            to log-space values if ``logspace_accumulators`` is enabled.
        backprop_mode: Backpropagation mode can be BackpropMode.GRADIENT, BackpropMode.HARD_EM,
            BackpropMode.HARD_EM_UNWEIGHTED or BackpropMode.SOFT_EM.
        accumulator_regularizer: Regularizer for accumulator (experimental)
        linear_accumulator_constraint: Constraint for accumulator defaults to constraint that
            ensures small positive constant at minimum. Will be ignored if logspace_accumulators
            is set to True.
        **kwargs: kwargs to pass on to keras.Layer super class
    """
    def __init__(
        self, num_sums, logspace_accumulators=None, accumulator_initializer=None,
        backprop_mode=BackpropMode.GRADIENT, accumulator_regularizer=None,
        linear_accumulator_constraint=None, **kwargs
    ):
        super(DenseSum, self).__init__(**kwargs)
        self.num_sums = num_sums
        self.logspace_accumulators = infer_logspace_accumulators(backprop_mode) \
            if logspace_accumulators is None else logspace_accumulators
        self.accumulator_initializer = accumulator_initializer or initializers.Constant(1)
        self.backprop_mode = backprop_mode
        self.accumulator_regularizer = accumulator_regularizer
        self.linear_accumulator_constraint = \
            linear_accumulator_constraint or GreaterEqualEpsilon(1e-10)
        self._num_decomps = self._num_scopes = self._accumulators = None

        if backprop_mode != BackpropMode.GRADIENT and logspace_accumulators:
            raise ValueError("Logspace accumulators are only supported for gradient backprop mode")

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self._num_scopes, self._num_decomps, _, num_nodes_in = input_shape

        weights_shape = (self._num_scopes, self._num_decomps, num_nodes_in, self.num_sums)

        initializer = self.accumulator_initializer
        accumulator_constraint = self.linear_accumulator_constraint
        if self.logspace_accumulators:
            initializer = logspace_wrapper_initializer(self.accumulator_initializer)
            accumulator_constraint = None

        self._accumulators = self.add_weight(
            name='sum_weights', shape=weights_shape, initializer=initializer,
            regularizer=self.accumulator_regularizer, constraint=accumulator_constraint
        )
        super(DenseSum, self).build(input_shape)

    def call(self, x):
        log_weights_unnormalized = self._accumulators

        if not self.logspace_accumulators and \
                self.backprop_mode in [BackpropMode.HARD_EM, BackpropMode.HARD_EM_UNWEIGHTED]:
            return logmatmul_hard_em_through_grads_from_accumulators(
                x, self._accumulators,
                unweighted=self.backprop_mode == BackpropMode.HARD_EM_UNWEIGHTED
            )

        if not self.logspace_accumulators and self.backprop_mode == BackpropMode.EM:
            log_weights_normalized = log_softmax_from_accumulators_with_em_grad(
                self._accumulators, axis=2)
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
            backprop_mode=self.backprop_mode,
            accumulator_regularizer=regularizers.serialize(self.accumulator_regularizer),
            linear_accumulator_constraint=constraints.serialize(self.linear_accumulator_constraint)
        )
        base_config = super(DenseSum, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
