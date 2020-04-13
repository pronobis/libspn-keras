import tensorflow as tf

from libspn_keras.math.logconv import logconv1x1_2d
from libspn_keras.math.logmatmul import logmatmul


def logmultiply_hard_em(child_log_prob, linear_accumulators):
    """
    Log multiplication (i.e. addition) for a region_graph_root sum node where we pass on
    Args:
        child_log_prob: Log probability of child
        linear_accumulators: accumulator in linear space.

    Returns:
        A tensor with log(child_log_prob * log_normalize(linear_accumulators))
    """

    @tf.custom_gradient
    def _inner_fn(child_log_prob, linear_accumulators):

        out = child_log_prob + tf.expand_dims(
            tf.nn.log_softmax(tf.log(linear_accumulators)), axis=0)

        def grad(dy):
            return dy, tf.reduce_sum(tf.reshape(dy, (-1, tf.size(linear_accumulators))), axis=0)

        return out, grad

    return _inner_fn(child_log_prob, linear_accumulators)


def logconv1x1_hard_em_through_grads_from_accumulators(
        child_log_prob, linear_accumulators, unweighted=False):
    """
    Hard EM grads by passing the path linear_accumulators down to the max weighted child using
    tf.custom_gradient. By doing so, we can conveniently use the graph built by
    tf.gradients(y, [xs], ...) to compute the updates needed for hard EM learning.

    Args:
        unweighted: A `bool` that indicates whether or not to use unweighted sum inputs for
            selecting the winning child.
        child_log_prob: A `Tensor` with log probabilities of the child node,
            shape is [..., batch, num_in]
        linear_accumulators: A `Tensor` with linear linear_accumulators of the sum node,
            shape is [..., num_in, num_out]
    """

    @tf.custom_gradient
    def _inner_fn(child_log_prob, linear_accumulators):

        log_accumulators = tf.math.log(linear_accumulators)

        # Normalized
        weights = tf.nn.log_softmax(log_accumulators, axis=2)

        if unweighted:
            pairwise_product_backprop = tf.expand_dims(child_log_prob, axis=3)
            out = logconv1x1_2d(child_log_prob, weights)
        else:
            # Pairwise product in forward pass
            # [scopes, decomps, batch, 1, num_in]
            child_log_prob = tf.expand_dims(child_log_prob, axis=3)
            # [scopes, decomps, 1, num_out, num_in]
            weights = tf.expand_dims(tf.transpose(weights, (0, 1, 3, 2)), axis=2)

            pairwise_product = child_log_prob + weights

            # Max per sum for determining winning child + choosing the constant for numerical
            # stability
            max_per_sum = tf.stop_gradient(tf.reduce_max(pairwise_product, axis=-1, keepdims=True))
            pairwise_product_backprop = child_log_prob + weights

            # Perform log(sum(exp(...))) with the numerical stability trick
            out = tf.math.log(tf.reduce_sum(tf.exp(
                pairwise_product - max_per_sum), axis=-1)) + tf.squeeze(max_per_sum, axis=-1)

        def grad(dy):
            # Determine winning child
            if unweighted:
                max_per_sum_backprop = tf.reduce_max(
                    pairwise_product_backprop, axis=-1, keepdims=True)
                equal_to_max = tf.cast(
                    tf.equal(pairwise_product_backprop, max_per_sum_backprop), tf.float32)
            else:
                equal_to_max = tf.cast(tf.equal(pairwise_product_backprop, max_per_sum), tf.float32)

            num_in = tf.shape(child_log_prob)[-1]
            num_out = tf.shape(out)[-1]
            equal_to_max_flat_outer = tf.reshape(
                equal_to_max,
                tf.concat([[-1], [num_in]], axis=0)
            )

            # Holds the index of the winning child per sum
            num_samples = num_out if unweighted else 1
            winning_child_per_sum = tf.reshape(
                tf.random.categorical(
                    tf.math.log(equal_to_max_flat_outer), num_samples=num_samples),
                tf.shape(out)
            )

            # Pass on the counts to the edges between child and parent
            edge_counts = tf.expand_dims(dy, -1) * tf.one_hot(winning_child_per_sum, depth=num_in)

            # Sum over parents to get counts per child
            child_counts = tf.reduce_sum(edge_counts, axis=3)

            # Sum over batch to get counts per weight
            weight_counts = tf.reduce_sum(edge_counts, axis=2)

            # Sum over spatial axes
            weight_counts = tf.reduce_sum(weight_counts, axis=[0, 1], keepdims=True)

            return child_counts, tf.transpose(weight_counts, (0, 1, 3, 2))

        return out, grad

    return _inner_fn(child_log_prob, linear_accumulators)


def logmatmul_hard_em_through_grads_from_accumulators(
        child_log_prob, linear_accumulators, unweighted=False):
    """
    Hard EM grads by passing the path linear_accumulators down to the max weighted child using
    tf.custom_gradient. By doing so, we can conveniently use the graph built by
    tf.gradients(y, [xs], ...) to compute the updates needed for hard EM learning.

    Args:
        unweighted: A `bool` that indicates whether or not to use unweighted sum inputs for
            selecting the winning child.
        child_log_prob: A `Tensor` with log probabilities of the child node,
            shape is [..., batch, num_in]
        linear_accumulators: A `Tensor` with linear linear_accumulators of the sum node,
            shape is [..., num_in, num_out]
    """

    @tf.custom_gradient
    def _inner_fn(child_log_prob, linear_accumulators):

        log_accumulators = tf.math.log(linear_accumulators)

        # Normalized
        weights = tf.nn.log_softmax(log_accumulators, axis=2)

        if unweighted:
            pairwise_product_backprop = tf.expand_dims(child_log_prob, axis=3)
            out = logmatmul(child_log_prob, weights)
        else:
            # Pairwise product in forward pass
            # [scopes, decomps, batch, 1, num_in]
            child_log_prob = tf.expand_dims(child_log_prob, axis=3)
            # [scopes, decomps, 1, num_out, num_in]
            weights = tf.expand_dims(tf.transpose(weights, (0, 1, 3, 2)), axis=2)

            pairwise_product = child_log_prob + weights

            # Max per sum for determining winning child + choosing the constant for numerical
            # stability
            max_per_sum = tf.stop_gradient(tf.reduce_max(pairwise_product, axis=-1, keepdims=True))
            pairwise_product_backprop = child_log_prob + weights

            # Perform log(sum(exp(...))) with the numerical stability trick
            out = tf.math.log(tf.reduce_sum(tf.exp(
                pairwise_product - max_per_sum), axis=-1)) + tf.squeeze(max_per_sum, axis=-1)

        def grad(dy):
            # Determine winning child
            if unweighted:
                max_per_sum_backprop = tf.reduce_max(
                    pairwise_product_backprop, axis=-1, keepdims=True)
                equal_to_max = tf.cast(
                    tf.equal(pairwise_product_backprop, max_per_sum_backprop), tf.float32)
            else:
                equal_to_max = tf.cast(tf.equal(pairwise_product_backprop, max_per_sum), tf.float32)

            num_in = tf.shape(child_log_prob)[-1]
            num_out = tf.shape(out)[-1]
            equal_to_max_flat_outer = tf.reshape(
                equal_to_max,
                tf.concat([[-1], [num_in]], axis=0)
            )

            # Holds the index of the winning child per sum
            num_samples = num_out if unweighted else 1
            winning_child_per_sum = tf.reshape(
                tf.random.categorical(
                    tf.math.log(equal_to_max_flat_outer), num_samples=num_samples),
                tf.shape(out)
            )

            # Pass on the counts to the edges between child and parent
            edge_counts = tf.expand_dims(dy, -1) * tf.one_hot(winning_child_per_sum, depth=num_in)

            # Sum over parents to get counts per child
            child_counts = tf.reduce_sum(edge_counts, axis=3)

            # Sum over batch to get counts per weight
            weight_counts = tf.reduce_sum(edge_counts, axis=2)

            return child_counts, tf.transpose(weight_counts, (0, 1, 3, 2))

        return out, grad

    return _inner_fn(child_log_prob, linear_accumulators)
