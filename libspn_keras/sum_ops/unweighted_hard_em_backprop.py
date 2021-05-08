from typing import Callable
from typing import Optional
from typing import Tuple
from typing import Union

import tensorflow as tf

from libspn_keras.math.logmatmul import logmatmul
from libspn_keras.sum_ops.base import SumOpBase
from libspn_keras.sum_ops.batch_scope_transpose import batch_scope_transpose


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
