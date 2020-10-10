from typing import Optional, Tuple

import tensorflow as tf

from libspn_keras.layers.dense_sum import DenseSum
from libspn_keras.logspace import logspace_wrapper_initializer


class Conv2DSum(DenseSum):
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
        sum_op (SumOpBase): SumOpBase instance which determines how to compute the forward and
            backward pass of the weighted sums
        accumulator_regularizer: Regularizer for accumulators
        linear_accumulator_constraint: Constraint for accumulators (only applied if
            log_space_accumulators==False)
        **kwargs: kwargs to pass on to the keras.Layer super class
    """

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the internal components for this leaf layer.

        Args:
            input_shape: Shape of the input Tensor.
        """
        # Create a trainable weight variable for this layer.
        _, num_scopes_vertical, num_scopes_horizontal, num_channels_in = input_shape

        weights_shape = (1, 1, num_channels_in, self.num_sums)

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
        super(DenseSum, self).build(input_shape)

    def call(self, x: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Compute a convolutional sum, using 1x1 convolutions.

        Args:
            x: Spatial Tensor.
            kwargs: Remaining keyword arguments.

        Returns:
            A Tensor with the same spatial dimensions and a number of channels determined
            by the number of channels set at the layer's instantiation.
        """
        return self.sum_op.weighted_conv(
            x,
            accumulators=self._accumulators,
            logspace_accumulators=self.logspace_accumulators,
            normalize_in_forward_pass=self._forward_normalize,
        )
