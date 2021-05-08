from typing import Callable
from typing import Tuple

import tensorflow as tf

from libspn_keras.sum_ops.base import SumOpBase
from libspn_keras.sum_ops.batch_scope_transpose import batch_scope_transpose


class SumOpSampleBackprop(SumOpBase):
    """
    Sum op with hard EM signals in backpropagation when computed through TensorFlow's autograd engine.

    Args:
        sample_prob: Sampling probability in the range of [0, 1]. Sampling logits are taken from
            the normalized log probability of the children of each sum.
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
                "Hard EM is only implemented for linear space accumulators"
            )

        with tf.name_scope("HardEMForwardPass"):
            weights = (
                self._to_log_weights(accumulators)
                if normalize_in_forward_pass
                else tf.math.log(accumulators)
            )
            # Pairwise product in forward pass
            # [scope, decomp, batch, nodes_in] -> [scope, decomp, batch, 1, nodes_in]
            x = tf.expand_dims(x, axis=3)
            # [scope, decomp, nodes_in, nodes_out] -> [scope, decomp, 1, nodes_out, nodes_in]
            weights = tf.expand_dims(tf.linalg.matrix_transpose(weights), axis=2)

            # Max per sum for determining winning child + choosing the constant for numerical
            # stability
            # [scope, decomp, batch, nodes_out, nodes_in]
            weighted_children = x + weights
            max_weighted_child = tf.stop_gradient(
                tf.reduce_max(weighted_children, axis=-1, keepdims=True)
            )

            # Perform log(sum(exp(...))) with the numerical stability trick
            # [scope, decomp, batch, nodes_out]
            out = tf.math.log(
                tf.reduce_sum(tf.exp(weighted_children - max_weighted_child), axis=-1)
            ) + tf.squeeze(max_weighted_child, axis=-1)

        @tf.custom_gradient
        def _inner_fn(
            x: tf.Tensor, accumulators: tf.Tensor
        ) -> Tuple[tf.Tensor, Callable[[tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]]:
            def grad(dy: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
                # Determine winning child
                num_in = tf.shape(x)[-1]
                num_scopes = tf.shape(weights)[0]
                num_decomps = tf.shape(weights)[1]
                num_out = tf.shape(weights)[-2]
                num_batch = tf.shape(x)[2]
                xw_flat_outer = tf.reshape(
                    weighted_children,
                    [num_scopes * num_decomps * num_batch * num_out, num_in],
                )
                # Holds the index of the winning child per sum
                samples = tf.random.categorical(xw_flat_outer, num_samples=1)
                winning_child_per_sum = tf.reshape(
                    samples, [num_scopes, num_decomps, num_batch, num_out]
                )
                # Pass on the counts to the edges between child and parent
                per_sample_weight_counts = dy[..., tf.newaxis] * tf.one_hot(
                    winning_child_per_sum, depth=num_in
                )

                child_counts = tf.reduce_sum(per_sample_weight_counts, axis=3)
                weight_counts = tf.reduce_sum(per_sample_weight_counts, axis=2)

                return child_counts, tf.linalg.matrix_transpose(weight_counts)

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

        Raises:
            NotImplementedError: Not implemented for SumOpSampleBackprop.
        """
        raise NotImplementedError(
            "Weighted children is not implemented for SumOpSampleBackprop"
        )

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

        Raises:
            NotImplementedError: When called with ``losgpace_accumulators == True``.
        """
        raise NotImplementedError(
            "EM is only implemented for linear space accumulators"
        )

    def default_logspace_accumulators(self) -> bool:
        """
        Whether or not accumulators should be represented in log-space by default.

        Returns:
            True if the default representation is in logspace and False otherwise.
        """
        return False
