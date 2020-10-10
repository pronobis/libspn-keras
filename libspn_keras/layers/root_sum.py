from typing import Optional, Tuple

import tensorflow as tf

from libspn_keras.layers.dense_sum import DenseSum
from libspn_keras.sum_ops import SumOpBase


class RootSum(DenseSum):
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
        accumulator_regularizer: Regularizer for accumulator.
        linear_accumulator_constraint: Constraint for linear accumulators. Defaults to a
            constraint that ensures a minimum of a small positive constant. If
            logspace_accumulators is set to True, this constraint wil be ignored
        sum_op (SumOpBase): SumOpBase instance which determines how to compute the forward and
            backward pass of the weighted sums
        **kwargs: kwargs to pass on to the keras.Layer super class
    """

    def __init__(
        self,
        return_weighted_child_logits: bool = True,
        logspace_accumulators: Optional[bool] = None,
        accumulator_initializer: Optional[tf.keras.initializers.Initializer] = None,
        trainable: bool = True,
        logspace_accumulator_constraint: Optional[
            tf.keras.constraints.Constraint
        ] = None,
        accumulator_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
        linear_accumulator_constraint: Optional[tf.keras.constraints.Constraint] = None,
        sum_op: Optional[SumOpBase] = None,
        **kwargs,
    ):
        super(RootSum, self).__init__(
            logspace_accumulators=logspace_accumulators,
            accumulator_initializer=accumulator_initializer,
            trainable=trainable,
            logspace_accumulator_constraint=logspace_accumulator_constraint,
            accumulator_regularizer=accumulator_regularizer,
            linear_accumulator_constraint=linear_accumulator_constraint,
            sum_op=sum_op,
            num_sums=1,
            **kwargs,
        )
        self.return_weighted_child_logits = return_weighted_child_logits

    def call(self, x: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Compute weighted sum at the root.

        If ``return_weighted_child_logits`` is set to ``True``, this will return P(X,Y_i) for all i
        rather than P(X).

        Args:
            x: Input, must have only 1 decomposition and 1 scope.
            kwargs: Remaining keyword arguments.

        Returns:
            Sum or weighted children.
        """
        x_reshaped = tf.reshape(x, (-1, 1, 1, self._num_nodes_in))

        if self.return_weighted_child_logits:
            out = self.sum_op.weighted_children(
                x_reshaped,
                accumulators=self._accumulators,
                logspace_accumulators=self.logspace_accumulators,
                normalize_in_forward_pass=self._forward_normalize,
            )
            num_out = self._accumulators.shape[2]
        else:
            out = super(RootSum, self).call(x_reshaped, **kwargs)
            num_out = 1
        return tf.reshape(out, [-1, num_out])

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the internal components for this layer.

        Args:
            input_shape: Shape of the input Tensor.

        Raises:
            ValueError: Raised when the number of input scopes does not equal 1, or
                if number of decompositions or nodes in the input is unknown.
        """
        num_batch, num_scopes, num_decomps, num_nodes = input_shape
        if num_scopes != 1:
            raise ValueError(
                f"Expected to have 1 scope at the input of a RootSum, got {num_scopes}"
            )
        if num_decomps is None:
            raise ValueError("Must have known number of decompositions")
        if num_nodes is None:
            raise ValueError("Must have known number of nodes")
        super(RootSum, self).build((num_batch, num_scopes, 1, num_decomps * num_nodes))

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute output shape of the layer.

        Args:
            input_shape: Input shape of the layer.

        Returns:
            Tuple of ints holding the output shape of the layer.


        Raises:
            ValueError: Raised when the number of decompositions
                or nodes in the input is unknown.
        """
        num_batch, _, num_decomps, num_nodes_in = input_shape
        if num_decomps is None:
            raise ValueError("Must have known number of decompositions")
        if num_nodes_in is None:
            raise ValueError("Must have known number of nodes")
        return (
            num_batch,
            num_decomps * num_nodes_in
            if self.return_weighted_child_logits
            else num_batch,
        )

    def get_config(self) -> dict:
        """
        Obtain a key-value representation of the layer config.

        Returns:
            A dict holding the configuration of the layer.
        """
        config = dict(return_weighted_child_logits=self.return_weighted_child_logits)
        base_config = super(RootSum, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
