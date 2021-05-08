from typing import Callable
from typing import Optional
from typing import Tuple
from typing import Union

import tensorflow as tf

from libspn_keras.sum_ops.base import SumOpBase
from libspn_keras.sum_ops.batch_scope_transpose import batch_scope_transpose


class SumOpHardEMBackprop(SumOpBase):
    """
    Sum op with hard EM signals in backpropagation when computed through TensorFlow's autograd engine.

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
                    tf.random.categorical(equal_to_max_flat_outer, num_samples=1),
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

    @batch_scope_transpose
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
                    tf.random.categorical(equal_to_max_flat_outer, num_samples=1),
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
