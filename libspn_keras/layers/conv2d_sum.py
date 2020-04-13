from libspn_keras.backprop_mode import BackpropMode, infer_logspace_accumulators
from libspn_keras.constraints.greater_equal_epsilon import GreaterEqualEpsilon
from libspn_keras.logspace import logspace_wrapper_initializer
from libspn_keras.math.hard_em_grads import logmatmul_hard_em_through_grads_from_accumulators, \
    logconv1x1_hard_em_through_grads_from_accumulators
from libspn_keras.math.logconv import logconv1x1_2d
from libspn_keras.math.logmatmul import logmatmul
from tensorflow import keras
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras import constraints
import tensorflow as tf

from libspn_keras.math.soft_em_grads import log_softmax_from_accumulators_with_em_grad


class Conv2DSum(keras.layers.Layer):
    """
    Computes a convolutional sum, i.e. weights are shared across the spatial axes.

    Args:
        num_sums: Number of sums per spatial cell. Corresponds to the number of channels in
            the output
        logspace_accumulators: If ``True``, accumulators will be represented in log-space which
            is typically used with ``BackpropMode.GRADIENT``. If ``False``, accumulators will be
            represented in linear space. Weights are computed by normalizing the accumulators
            per sum, so that we always end up with a normalized SPN. If ``None`` (default) it
            will be set to ``True`` for ``BackpropMode.GRADIENT`` and ``False`` otherwise.
        accumulator_initializer: Initializer for accumulator
        backprop_mode: Backpropagation mode. Can be either GRADIENT, HARD_EM, SOFT_EM or
            HARD_EM_UNWEIGHTED
        accumulator_regularizer: Regularizer for accumulators
        linear_accumulator_constraint: Constraint for accumulators (only applied if
            log_space_accumulators==False)
        **kwargs: kwargs to pass on to the keras.Layer super class
    """

    def __init__(
        self, num_sums, logspace_accumulators=None, accumulator_initializer=None,
        backprop_mode=BackpropMode.GRADIENT, accumulator_regularizer=None,
        linear_accumulator_constraint=GreaterEqualEpsilon(1e-10), **kwargs
    ):
        # TODO make docstrings more consistent across different sum instances

        # TODO automatically infer value of logspace_accumulator from the backprop mode

        # TODO verify compatibility of backprop mode and logspace_accumulator

        # TODO consider renaming 'accumulator' to 'child_counts'
        super(Conv2DSum, self).__init__(**kwargs)
        self.num_sums = num_sums
        self.logspace_accumulators = infer_logspace_accumulators(backprop_mode) \
            if logspace_accumulators is None else logspace_accumulators
        self.accumulator_initializer = accumulator_initializer or initializers.Constant(1)
        self.backprop_mode = backprop_mode
        self.accumulator_regularizer = accumulator_regularizer
        self.linear_accumulator_constraint = linear_accumulator_constraint
        self.accumulators = None

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        _, num_scopes_vertical, num_scopes_horizontal, num_channels_in = input_shape

        weights_shape = (1, 1, num_channels_in, self.num_sums)

        initializer = self.accumulator_initializer
        accumulator_contraint = self.linear_accumulator_constraint
        if self.logspace_accumulators:
            initializer = logspace_wrapper_initializer(initializer)
            accumulator_contraint = None

        self.accumulators = self.add_weight(
            name='sum_weights', shape=weights_shape, initializer=initializer,
            regularizer=self.accumulator_regularizer, constraint=accumulator_contraint
        )
        super(Conv2DSum, self).build(input_shape)

    def call(self, x):

        log_weights_unnormalized = self.accumulators

        if not self.logspace_accumulators \
                and self.backprop_mode in [BackpropMode.HARD_EM, BackpropMode.HARD_EM_UNWEIGHTED]:
            out = logconv1x1_hard_em_through_grads_from_accumulators(
                x, self.accumulators,
                unweighted=self.backprop_mode == BackpropMode.HARD_EM_UNWEIGHTED
            )
            return out

        if not self.logspace_accumulators and self.backprop_mode == BackpropMode.EM:
            log_weights_normalized = log_softmax_from_accumulators_with_em_grad(
                self.accumulators, axis=2)
        elif not self.logspace_accumulators:
            log_weights_normalized = tf.nn.log_softmax(
                tf.math.log(log_weights_unnormalized), axis=2)
        else:
            log_weights_normalized = tf.nn.log_softmax(log_weights_unnormalized, axis=2)

        out = logconv1x1_2d(x, log_weights_normalized)

        return out

    def compute_output_shape(self, input_shape):
        num_batch, num_scopes_vertical, num_scopes_horizontal, _ = input_shape
        return num_batch, num_scopes_vertical, num_scopes_horizontal, self.num_sums

    def get_config(self):
        config = dict(
            num_sums=self.num_sums,
            accumulator_initializer=initializers.serialize(self.accumulator_initializer),
            logspace_accumulators=self.logspace_accumulators,
            backprop_mode=self.backprop_mode,
            accumulator_regularizer=regularizers.serialize(self.accumulator_regularizer),
            linear_accumulator_constraint=constraints.serialize(self.linear_accumulator_constraint)
        )
        base_config = super(Conv2DSum, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
