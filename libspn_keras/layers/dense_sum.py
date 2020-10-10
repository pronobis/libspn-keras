from typing import Optional, Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import constraints
from tensorflow.keras import initializers
from tensorflow.keras import regularizers

from libspn_keras.config.accumulator_initializer import (
    get_default_accumulator_initializer,
)
from libspn_keras.config.linear_accumulator_constraint import (
    get_default_linear_accumulators_constraint,
)
from libspn_keras.config.logspace_accumulator_constraint import (
    get_default_logspace_accumulators_constraint,
)
from libspn_keras.config.sum_op import get_default_sum_op
from libspn_keras.constraints import LogNormalized
from libspn_keras.constraints.greater_equal_epsilon_normalized import (
    GreaterEqualEpsilonNormalized,
)
from libspn_keras.logspace import logspace_wrapper_initializer
from libspn_keras.sum_ops import SumOpBase


class DenseSum(keras.layers.Layer):
    """
    Computes densely connected sums per scope and decomposition.

    Expects incoming ``Tensor`` to be of shape [num_scopes, num_decomps, num_batch, num_nodes]. If your
    input is passed through a ``FlatToRegions`` layer this is already taken care of.

    Args:
        num_sums: Number of sums per scope
        logspace_accumulators: If ``True``, accumulators will be represented in log-space which
            is typically used with ``BackpropMode.GRADIENT``. If ``False``, accumulators will be
            represented in linear space. Weights are computed by normalizing the accumulators
            per sum, so that we always end up with a normalized SPN. If ``None`` (default) it
            will be set to ``True`` for ``BackpropMode.GRADIENT`` and ``False`` otherwise.
        accumulator_initializer: Initializer for accumulator. Will automatically be converted
            to log-space values if ``logspace_accumulators`` is enabled.
        accumulator_regularizer: Regularizer for accumulator (experimental)
        linear_accumulator_constraint: Constraint for accumulator defaults to constraint that
            ensures small positive constant at minimum. Will be ignored if logspace_accumulators
            is set to True.
        sum_op (SumOpBase): SumOpBase instance which determines how to compute the forward and
            backward pass of the weighted sums
        **kwargs: kwargs to pass on to keras.Layer super class
    """

    def __init__(
        self,
        num_sums: int,
        logspace_accumulators: Optional[bool] = None,
        accumulator_initializer: Optional[keras.initializers.Initializer] = None,
        sum_op: Optional[SumOpBase] = None,
        accumulator_regularizer: Optional[keras.regularizers.Regularizer] = None,
        logspace_accumulator_constraint: Optional[keras.constraints.Constraint] = None,
        linear_accumulator_constraint: Optional[keras.constraints.Constraint] = None,
        **kwargs
    ):
        super(DenseSum, self).__init__(**kwargs)
        self.num_sums = num_sums
        self.sum_op = sum_op or get_default_sum_op()
        self.logspace_accumulators = (
            self.sum_op.default_logspace_accumulators()
            if logspace_accumulators is None
            else logspace_accumulators
        )
        self.accumulator_initializer = (
            accumulator_initializer or get_default_accumulator_initializer()
        )
        self.accumulator_regularizer = accumulator_regularizer
        self.linear_accumulator_constraint = (
            linear_accumulator_constraint
            or get_default_linear_accumulators_constraint()
        )
        self.logspace_accumulator_constraint = (
            logspace_accumulator_constraint
            or get_default_logspace_accumulators_constraint()
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the internal components for this layer.

        Args:
            input_shape: Shape of the input Tensor.
        """
        # Create a trainable weight variable for this layer.
        _, self._num_scopes, self._num_decomps, self._num_nodes_in = input_shape

        weights_shape = (
            self._num_scopes,
            self._num_decomps,
            self._num_nodes_in,
            self.num_sums,
        )

        initializer = self.accumulator_initializer
        accumulator_constraint = self.linear_accumulator_constraint
        if self.logspace_accumulators:
            initializer = logspace_wrapper_initializer(self.accumulator_initializer)
            accumulator_constraint = self.logspace_accumulator_constraint

        self._accumulators = self.add_weight(
            name="sum_weights",
            shape=weights_shape,
            initializer=initializer,
            regularizer=self.accumulator_regularizer,
            constraint=accumulator_constraint,
        )
        if accumulator_constraint is not None:
            self._accumulators.assign(accumulator_constraint(self._accumulators))
        self._forward_normalize = not isinstance(
            accumulator_constraint, (GreaterEqualEpsilonNormalized, LogNormalized)
        )
        super(DenseSum, self).build(input_shape)

    def call(self, x: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Compute the probability of the leaf nodes.

        Args:
            x: Spatial or region Tensor with raw input values.
            kwargs: Remaining keyword arguments.

        Returns:
            A Tensor with the probabilities per component.
        """
        return self.sum_op.weighted_sum(
            x, self._accumulators, self.logspace_accumulators, self._forward_normalize
        )

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute output shape of the layer.

        Args:
            input_shape: Input shape of the layer.

        Returns:
            Tuple of ints holding the output shape of the layer.
        """
        num_scopes, num_decomps, num_batch, _ = input_shape
        return num_batch, num_scopes, num_decomps, self.num_sums

    def get_config(self) -> dict:
        """
        Obtain a key-value representation of the layer config.

        Returns:
            A dict holding the configuration of the layer.
        """
        config = dict(
            num_sums=self.num_sums,
            accumulator_initializer=initializers.serialize(
                self.accumulator_initializer
            ),
            logspace_accumulators=self.logspace_accumulators,
            accumulator_regularizer=regularizers.serialize(
                self.accumulator_regularizer
            ),
            linear_accumulator_constraint=constraints.serialize(
                self.linear_accumulator_constraint
            ),
        )
        base_config = super(DenseSum, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
