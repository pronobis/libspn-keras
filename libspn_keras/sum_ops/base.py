import abc
from typing import Callable
from typing import Tuple

import tensorflow as tf


class SumOpBase(abc.ABC):
    """
    The base sum op primitive.

    Descendants define weighted sum implementations to override gradients if necessary.
    """

    @abc.abstractmethod
    def weighted_sum(
        self,
        x: tf.Tensor,
        accumulators: tf.Tensor,
        logspace_accumulators: bool,
        normalize_in_forward_pass: bool,
    ) -> tf.Tensor:
        """
        Implement sum operation on log inputs X and accumulators w.

        Args:
            x: Input Tensor.
            accumulators: Unnormalized accumulators.
            logspace_accumulators: Whether accumulators are in logspace.
            normalize_in_forward_pass: Whether weights should be normalized during forward inference.
        """

    @abc.abstractmethod
    def weighted_children(
        self,
        x: tf.Tensor,
        accumulators: tf.Tensor,
        logspace_accumulators: bool,
        normalize_in_forward_pass: bool,
    ) -> tf.Tensor:
        """
        Compute weighted children (used in RootSum).

        Args:
            x: Input Tensor.
            accumulators: Unnormalized accumulators.
            logspace_accumulators: Whether accumulators are in logspace.
            normalize_in_forward_pass: Whether weights should be normalized during forward inference.
        """

    @abc.abstractmethod
    def weighted_conv(
        self,
        x: tf.Tensor,
        accumulators: tf.Tensor,
        logspace_accumulators: bool,
        normalize_in_forward_pass: bool,
    ) -> tf.Tensor:
        """
        Compute weighted convolution (used in ConvSum).

        Args:
            x: Input Tensor.
            accumulators: Unnormalized accumulators.
            logspace_accumulators: Whether accumulators are in logspace.
            normalize_in_forward_pass: Whether weights should be normalized during forward inference.
        """

    @abc.abstractmethod
    def default_logspace_accumulators(self) -> bool:
        """Whether default config is to have accumulators in logspace."""

    @staticmethod
    def _to_log_weights(x: tf.Tensor) -> tf.Tensor:
        with tf.name_scope("ToLogWeights"):
            return SumOpBase._log_normalize(tf.math.log(x))

    @staticmethod
    def _log_normalize(x: tf.Tensor) -> tf.Tensor:
        with tf.name_scope("LogNormalize"):
            return tf.nn.log_softmax(x, axis=-2)

    def _to_logspace_override_grad(
        self, accumulators: tf.Tensor, normalize_in_forward_pass: bool
    ) -> Tuple[tf.Tensor, Callable[[tf.Tensor], tf.Tensor]]:
        @tf.custom_gradient
        def _inner(
            accumulators: tf.Tensor,
        ) -> Tuple[tf.Tensor, Callable[[tf.Tensor], tf.Tensor]]:
            if normalize_in_forward_pass:
                return (
                    self._to_log_weights(accumulators),
                    lambda dy: tf.math.divide_no_nan(
                        dy, tf.abs(tf.reduce_sum(dy, axis=2, keepdims=True))
                    ),
                )
            else:
                return (
                    tf.math.log(accumulators),
                    lambda dy: tf.math.divide_no_nan(
                        dy, tf.abs(tf.reduce_sum(dy, axis=2, keepdims=True))
                    ),
                )

        return _inner(accumulators)

    def _weights_in_logspace(
        self,
        accumulators: tf.Tensor,
        logspace_accumulators: bool,
        normalize_in_forward_pass: bool,
    ) -> tf.Tensor:
        if logspace_accumulators:
            return (
                self._log_normalize(accumulators)
                if normalize_in_forward_pass
                else accumulators
            )
        else:
            return (
                self._to_log_weights(accumulators)
                if normalize_in_forward_pass
                else tf.math.log(accumulators)
            )
