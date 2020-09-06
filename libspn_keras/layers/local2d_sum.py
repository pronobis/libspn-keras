from typing import Optional, Tuple

from libspn_keras.layers.dense_sum import DenseSum
from libspn_keras.logspace import logspace_wrapper_initializer


class Local2DSum(DenseSum):
    """
    Computes a spatial local sum, i.e. all cells will have unique weights.

    In other words, there is no weight sharing across the spatial axes.

    Args:
        num_sums: Number of sums per spatial cell. Corresponds to the number of channels in
            the output
        logspace_accumulators: If ``True``, accumulators will be represented in log-space which
            is typically used with ``BackpropMode.GRADIENT``. If ``False``, accumulators will be
            represented in linear space. Weights are computed by normalizing the accumulators
            per sum, so that we always end up with a normalized SPN. If ``None`` (default) it
            will be set to ``True`` for ``BackpropMode.GRADIENT`` and ``False`` otherwise.
        accumulator_initializer: Initializer for accumulator
        accumulator_regularizer: Regularizer for accumulators
        linear_accumulator_constraint: Constraint for accumulators (only applied if
            log_space_accumulators==False)
        sum_op (SumOpBase): SumOpBase instance which determines how to compute the forward and
            backward pass of the weighted sums
        **kwargs: kwargs to pass on to the keras.Layer super class
    """

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the internal components for this layer.

        Args:
            input_shape: Shape of the input Tensor.
        """
        # Create a trainable weight variable for this layer.
        _, num_scopes_vertical, num_scopes_horizontal, num_channels_in = input_shape

        weights_shape = (
            num_scopes_vertical,
            num_scopes_horizontal,
            num_channels_in,
            self.num_sums,
        )

        initializer = self.accumulator_initializer
        accumulator_contraint = self.linear_accumulator_constraint
        if self.logspace_accumulators:
            initializer = logspace_wrapper_initializer(initializer)
            accumulator_contraint = self.logspace_accumulator_constraint

        self._accumulators = self.add_weight(
            name="sum_weights",
            shape=weights_shape,
            initializer=initializer,
            regularizer=self.accumulator_regularizer,
            constraint=accumulator_contraint,
        )
        super(Local2DSum, self).build(input_shape)
