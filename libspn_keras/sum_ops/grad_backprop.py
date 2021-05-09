from typing import Optional

import tensorflow as tf

from libspn_keras.math.logconv import logconv1x1_2d
from libspn_keras.math.logmatmul import logmatmul
from libspn_keras.sum_ops.base import SumOpBase
from libspn_keras.sum_ops.batch_scope_transpose import batch_scope_transpose


class SumOpGradBackprop(SumOpBase):
    """
    Sum op primitive with gradient in backpropagation when computed through TensorFlow's autograd engine.

    Internally, weighted sums are computed with default gradients for all ops being used.

    Args:
            logspace_accumulators: If provided overrides default log-space choice. For a
                :class:`~libspn_keras.SumOpGradBackprop` the default is ``True``
    """

    def __init__(
        self, logspace_accumulators: Optional[bool] = None,
    ):
        self._logspace_accumulators = logspace_accumulators

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
        """
        w = self._weights_in_logspace(
            accumulators, logspace_accumulators, normalize_in_forward_pass
        )
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
        """
        w = self._weights_in_logspace(
            accumulators, logspace_accumulators, normalize_in_forward_pass
        )
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
        """
        w = self._weights_in_logspace(
            accumulators, logspace_accumulators, normalize_in_forward_pass
        )
        return logconv1x1_2d(x, w)

    def default_logspace_accumulators(self) -> bool:
        """
        Whether or not accumulators should be represented in log-space by default.

        Returns:
            True if the default representation is in logspace and False otherwise.
        """
        return (
            self._logspace_accumulators
            if self._logspace_accumulators is not None
            else True
        )
