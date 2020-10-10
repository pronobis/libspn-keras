import abc
import functools
from typing import Callable, Optional, Tuple, Union

import tensorflow as tf

from libspn_keras.math.logconv import logconv1x1_2d
from libspn_keras.math.logmatmul import logmatmul


def _batch_scope_tranpose(f):  # type: ignore  # noqa: ANN001,ANN202
    @functools.wraps(f)  # type: ignore  # noqa: ANN202
    def impl(self: SumOpBase, x: tf.Tensor, *args, **kwargs) -> tf.Tensor:  # type: ignore
        with tf.name_scope("ScopesAndDecompsFirst"):
            scopes_decomps_first = tf.transpose(x, (1, 2, 0, 3))
        result = f(self, scopes_decomps_first, *args, **kwargs)
        with tf.name_scope("BatchFirst"):
            return tf.transpose(result, (2, 0, 1, 3))

    return impl


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
                    lambda dy: dy
                    / (tf.abs(tf.reduce_sum(dy, axis=2, keepdims=True)) + 1e-20),
                )
            else:
                return (
                    tf.math.log(accumulators),
                    lambda dy: dy
                    / (tf.abs(tf.reduce_sum(dy, axis=2, keepdims=True)) + 1e-20),
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


class SumOpGradBackprop(SumOpBase):
    """
    Sum op primitive with gradient in backpropagation when computed through TensorFlow's autograd engine.

    Internally, weighted sums are computed with default gradients for all ops being used.

    Args:
            logspace_accumulators: If provided overrides default log-space choice. For a
                ``SumOpGradBackprop`` the default is ``True``
    """

    def __init__(
        self, logspace_accumulators: Optional[bool] = None,
    ):
        self._logspace_accumulators = logspace_accumulators

    @_batch_scope_tranpose
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


class SumOpEMBackprop(SumOpBase):
    """
    Sum op primitive with EM signals in backpropagation.

    These are dense EM signals as opposed to the other EM based instances of ``SumOpBase``.
    """

    @_batch_scope_tranpose
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


class SumOpHardEMBackprop(SumOpBase):
    """
    Sum op with hard EM signals in backpropagation when computed through TensorFlow's autograd engine.

    Args:
        sample_prob: Sampling probability in the range of [0, 1]. Sampling logits are taken from
            the normalized log probability of the children of each sum.
    """

    def __init__(self, sample_prob: Optional[Union[float, tf.Tensor]] = None):
        self.sample_prob = sample_prob

    @_batch_scope_tranpose
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
                "Hard EM is only implemented for linear space accumulators"
            )

        @tf.custom_gradient
        def _inner_fn(
            x: tf.Tensor, accumulators: tf.Tensor
        ) -> Tuple[tf.Tensor, Callable[[tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]]:
            with tf.name_scope("HardEMForwardPass"):
                w = (
                    self._to_log_weights(accumulators)
                    if normalize_in_forward_pass
                    else tf.math.log(accumulators)
                )
                # Pairwise product in forward pass
                x = tf.expand_dims(x, axis=3)
                w = tf.expand_dims(tf.linalg.matrix_transpose(w), axis=2)

                # Max per sum for determining winning child + choosing the constant for numerical
                # stability
                weighted_children = x + w
                max_weighted_child = tf.stop_gradient(
                    tf.reduce_max(weighted_children, axis=-1, keepdims=True)
                )

                # Perform log(sum(exp(...))) with the numerical stability trick
                out = tf.math.log(
                    tf.reduce_sum(
                        tf.exp(weighted_children - max_weighted_child), axis=-1
                    )
                ) + tf.squeeze(max_weighted_child, axis=-1)

            def grad(dy: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
                # Determine winning child
                equal_to_max: tf.Tensor = tf.cast(
                    tf.equal(weighted_children, max_weighted_child), tf.float32
                )

                # Holds the index of the winning child per sum
                if self.sample_prob is not None:
                    equal_to_max = (
                        tf.math.log(
                            self.sample_prob * tf.exp(x - max_weighted_child)
                            + tf.convert_to_tensor(1.0 - self.sample_prob)
                            * equal_to_max
                        )
                        + max_weighted_child
                    )
                else:
                    equal_to_max = tf.math.log(equal_to_max)
                num_in = tf.shape(x)[-1]
                equal_to_max_flat_outer = tf.reshape(
                    equal_to_max, tf.concat([[-1], [num_in]], axis=0)
                )

                # Holds the index of the winning child per sum
                winning_child_per_sum = tf.reshape(
                    tf.random.categorical(
                        tf.math.log(equal_to_max_flat_outer), num_samples=1
                    ),
                    tf.shape(out),
                )

                # Pass on the counts to the edges between child and parent
                per_sample_weight_counts = tf.expand_dims(dy, -1) * tf.one_hot(
                    winning_child_per_sum, depth=num_in
                )

                child_counts = tf.reduce_sum(per_sample_weight_counts, axis=3)
                weight_counts = tf.reduce_sum(per_sample_weight_counts, axis=2)

                return child_counts, tf.transpose(weight_counts, (0, 1, 3, 2))

            return out, grad

        return _inner_fn(x, accumulators)

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

    @_batch_scope_tranpose
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
                "Hard EM is only implemented for linear space accumulators"
            )

        @tf.custom_gradient
        def _inner_fn(
            x: tf.Tensor, accumulators: tf.Tensor
        ) -> Tuple[tf.Tensor, Callable[[tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]]:
            with tf.name_scope("HardEMForwardPass"):
                w = (
                    self._to_log_weights(accumulators)
                    if normalize_in_forward_pass
                    else tf.math.log(normalize_in_forward_pass)
                )
                # Pairwise product in forward pass
                x = tf.expand_dims(x, axis=3)
                w = tf.expand_dims(tf.linalg.matrix_transpose(w), axis=2)

                # Max per sum for determining winning child + choosing the constant for numerical
                # stability
                weighted_children = x + w
                max_weighted_child = tf.stop_gradient(
                    tf.reduce_max(weighted_children, axis=-1, keepdims=True)
                )

                # Perform log(sum(exp(...))) with the numerical stability trick
                out = tf.math.log(
                    tf.reduce_sum(
                        tf.exp(weighted_children - max_weighted_child), axis=-1
                    )
                ) + tf.squeeze(max_weighted_child, axis=-1)

            def grad(dy: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
                # Determine winning child
                equal_to_max = tf.cast(
                    tf.equal(weighted_children, max_weighted_child), tf.float32
                )

                # Holds the index of the winning child per sum
                if self.sample_prob is not None:
                    equal_to_max = (
                        tf.math.log(
                            self.sample_prob * tf.exp(x - max_weighted_child)
                            + (1.0 - self.sample_prob) * equal_to_max
                        )
                        + max_weighted_child
                    )
                else:
                    equal_to_max = tf.math.log(equal_to_max)
                num_in = tf.shape(x)[-1]
                equal_to_max_flat_outer = tf.reshape(
                    equal_to_max, tf.concat([[-1], [num_in]], axis=0)
                )

                # Holds the index of the winning child per sum
                winning_child_per_sum = tf.reshape(
                    tf.random.categorical(
                        tf.math.log(equal_to_max_flat_outer), num_samples=1
                    ),
                    tf.shape(out),
                )

                # Pass on the counts to the edges between child and parent
                per_sample_weight_counts = tf.expand_dims(dy, -1) * tf.one_hot(
                    winning_child_per_sum, depth=num_in
                )

                child_counts = tf.reduce_sum(per_sample_weight_counts, axis=3)
                weight_counts = tf.reduce_sum(per_sample_weight_counts, axis=2)

                # Sum over spatial axes
                weight_counts = tf.reduce_sum(weight_counts, axis=[0, 1], keepdims=True)

                return child_counts, tf.linalg.matrix_transpose(weight_counts)

            return out, grad

        return _inner_fn(x, accumulators)

    def default_logspace_accumulators(self) -> bool:
        """
        Whether or not accumulators should be represented in log-space by default.

        Returns:
            True if the default representation is in logspace and False otherwise.
        """
        return False


class SumOpUnweightedHardEMBackprop(SumOpBase):
    """
    Sum op with hard EM signals in backpropagation when computed through TensorFlow's autograd engine.

    Instead of using weighted sum inputs to select the maximum child, it relies on unweighted child
    inputs, which has the advantage of alleviating a self-amplifying chain of hard EM signals in
    deep SPNs.

    Args:
        sample_prob: Sampling probability in the range of [0, 1]. Sampling logits are taken from
            the normalized log probability of the children of each sum.
    """

    def __init__(self, sample_prob: Optional[Union[float, tf.Tensor]] = None):
        self.sample_prob = sample_prob

    @_batch_scope_tranpose
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
                "Hard EM is only implemented for linear space accumulators"
            )

        @tf.custom_gradient
        def _inner_fn(
            x: tf.Tensor, accumulators: tf.Tensor
        ) -> Tuple[tf.Tensor, Callable[[tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]]:

            # Normalized
            weights = (
                self._to_log_weights(accumulators)
                if normalize_in_forward_pass
                else tf.math.log(accumulators)
            )

            out = logmatmul(x, weights)

            def grad(parent_counts: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
                # Determine winning child
                max_child = tf.reduce_max(x, axis=-1, keepdims=True)
                equal_to_max = tf.cast(tf.equal(max_child, x), tf.float32)

                # Holds the index of the winning child per sum
                if self.sample_prob is not None:
                    equal_to_max = (
                        tf.math.log(
                            self.sample_prob * tf.exp(x - max_child)
                            + (1.0 - self.sample_prob) * equal_to_max
                        )
                        + max_child
                    )
                else:
                    equal_to_max = tf.math.log(equal_to_max)

                num_in = tf.shape(x)[-1]
                equal_to_max_flat_outer = tf.reshape(
                    equal_to_max, tf.concat([[-1], [num_in]], axis=0)
                )
                winning_child_per_scope = tf.reshape(
                    tf.random.categorical(equal_to_max_flat_outer, num_samples=1),
                    tf.shape(x)[:-1],
                )

                sum_parent_counts = tf.reduce_sum(parent_counts, axis=-1, keepdims=True)

                winning_child_per_scope_one_hot = tf.one_hot(
                    winning_child_per_scope, depth=num_in, axis=-1
                )
                child_counts = winning_child_per_scope_one_hot * sum_parent_counts

                weight_counts = tf.matmul(
                    winning_child_per_scope_one_hot, parent_counts, transpose_a=True
                )
                return child_counts, weight_counts

            return out, grad

        return _inner_fn(x, accumulators)

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
                "Hard EM is only implemented for linear space accumulators"
            )

        @tf.custom_gradient
        def _inner_fn(
            x: tf.Tensor, accumulators: tf.Tensor
        ) -> Tuple[tf.Tensor, Callable[[tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]]:

            # Normalized
            weights = (
                self._to_log_weights(accumulators)
                if normalize_in_forward_pass
                else tf.math.log(accumulators)
            )

            out = logmatmul(x, weights)

            def grad(parent_counts: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
                # Determine winning child
                max_child = tf.reduce_max(x, axis=-1, keepdims=True)
                equal_to_max = tf.cast(tf.equal(max_child, x), tf.float32)

                # Holds the index of the winning child per sum
                if self.sample_prob is not None:
                    equal_to_max = (
                        tf.math.log(
                            self.sample_prob * tf.exp(x - max_child)
                            + (1.0 - self.sample_prob) * equal_to_max
                        )
                        + max_child
                    )
                else:
                    equal_to_max = tf.math.log(equal_to_max)

                num_in = tf.shape(x)[-1]
                equal_to_max_flat_outer = tf.reshape(
                    equal_to_max, tf.concat([[-1], [num_in]], axis=0)
                )
                winning_child_per_scope = tf.reshape(
                    tf.random.categorical(equal_to_max_flat_outer, num_samples=1),
                    tf.shape(x)[:-1],
                )

                sum_parent_counts = tf.reduce_sum(parent_counts, axis=-1, keepdims=True)

                winning_child_per_scope_one_hot = tf.one_hot(
                    winning_child_per_scope, depth=num_in, axis=-1
                )
                child_counts = winning_child_per_scope_one_hot * sum_parent_counts

                weight_counts = tf.matmul(
                    winning_child_per_scope_one_hot, parent_counts, transpose_a=True
                )
                weight_counts = tf.reduce_sum(weight_counts, axis=[0, 1], keepdims=True)
                return child_counts, weight_counts

            return out, grad

        return _inner_fn(x, accumulators)

    def default_logspace_accumulators(self) -> bool:
        """
        Whether or not accumulators should be represented in log-space by default.

        Returns:
            True if the default representation is in logspace and False otherwise.
        """
        return False
