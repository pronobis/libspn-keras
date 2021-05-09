import tensorflow as tf

from libspn_keras.math.logconv import logconv1x1_2d
from libspn_keras.math.logmatmul import logmatmul
from libspn_keras.sum_ops.base import SumOpBase
from libspn_keras.sum_ops.batch_scope_transpose import batch_scope_transpose


class SumOpEMBackprop(SumOpBase):
    """
    Sum op primitive with EM signals in backpropagation.

    These are dense EM signals as opposed to the other EM based instances of
    :class:`~libspn_keras.sum_ops.SumOpBase`
    """

    @batch_scope_transpose
    def weighted_sum(
        self,
        x: tf.Tensor,
        accumulators: tf.Tensor,
        logspace_accumulators: bool,
        normalize_in_forward_pass: bool,
    ) -> tf.Tensor:
        """
        Compute a weighted sum.

        Args:
            x: Input Tensor
            accumulators: Accumulators, can be seen as unnormalized representations of weights.
            logspace_accumulators: Whether or not accumulators are represented in logspace.
            normalize_in_forward_pass: Whether weights should be normalized during forward inference.

        Returns:
            A Tensor with the weighted sums.

        Raises:
            NotImplementedError: When called with ``losgpace_accumulators == True``.
        """
        if logspace_accumulators:
            raise NotImplementedError(
                "EM is only implemented for linear space accumulators"
            )
        w = self._to_logspace_override_grad(accumulators, normalize_in_forward_pass)
        return logmatmul(x, w)

    def weighted_children(
        self,
        x: tf.Tensor,
        accumulators: tf.Tensor,
        logspace_accumulators: bool,
        normalize_in_forward_pass: bool,
    ) -> tf.Tensor:
        """
        Compute weighted children, without summing over the final axis.

        This is used for a RootSum to compute :math:`P(X,Y_i)` for any :math:`i`

        Args:
            x: Input Tensor
            accumulators: Accumulators, can be seen as unnormalized representations of weights.
            logspace_accumulators: Whether or not accumulators are represented in logspace.
            normalize_in_forward_pass: Whether weights should be normalized during forward inference.

        Returns:
            A Tensor with the weighted sums.

        Raises:
            NotImplementedError: When called with ``losgpace_accumulators == True``.
        """
        if logspace_accumulators:
            raise NotImplementedError(
                "EM is only implemented for linear space accumulators"
            )
        w = self._to_logspace_override_grad(accumulators, normalize_in_forward_pass)
        with tf.name_scope("PairwiseLogMultiply"):
            return x + tf.linalg.matrix_transpose(w)

    def weighted_conv(
        self,
        x: tf.Tensor,
        accumulators: tf.Tensor,
        logspace_accumulators: bool,
        normalize_in_forward_pass: bool,
    ) -> tf.Tensor:
        """
        Compute weighted convolutions.

        This is used for a Conv2DSum.

        Args:
            x: Input Tensor
            accumulators: Accumulators, can be seen as unnormalized representations of weights.
            logspace_accumulators: Whether or not accumulators are represented in logspace.
            normalize_in_forward_pass: Whether weights should be normalized during forward inference.

        Returns:
            A Tensor with the weighted convolutions.

        Raises:
            NotImplementedError: When called with ``losgpace_accumulators == True``.
        """
        if logspace_accumulators:
            raise NotImplementedError(
                "EM is only implemented for linear space accumulators"
            )
        w = self._to_logspace_override_grad(accumulators, normalize_in_forward_pass)
        return logconv1x1_2d(x, w)

    def default_logspace_accumulators(self) -> bool:
        """
        Whether or not accumulators should be represented in log-space by default.

        Returns:
            True if the default representation is in logspace and False otherwise.
        """
        return False
