import tensorflow as tf


def replace_infs_with_zeros(x):
    """
    Replaces infinite values with zeroes
    Args:
        x: A `Tensor`.

    Returns:
        A `Tensor` with the same shape an dtype as `x` where infinite values are replaced
        with zeroes.
    """
    return tf.where(tf.math.is_inf(x), tf.zeros_like(x), x)


def logmatmul_hard_em_through_grads_from_accumulators(child_log_prob, accumulators, accumulators_in_logspace=False):
    """
    Hard EM grads by passing the path accumulators down to the max weighted child. By doing so, we can
    conveniently use the graph built by tf.gradients(y, x, ...) to compute the updates needed for hard EM learning.

    Args:
        child_log_prob: A `Tensor` with log probabilities of the child node, shape is [..., batch, num_in]
        accumulators: A `Tensor` with accumulators of the sum node, shape is [..., num_in, num_out]
        accumulators_in_logspace: A `bool` that indicates whether or not the accumulators are in log space
    """

    @tf.custom_gradient
    def _inner_fn(child_log_prob, accumulators):

        if accumulators_in_logspace:
            log_accumulators = accumulators
        else:
            log_accumulators = tf.math.log(accumulators)

        # Normalized
        weights = tf.nn.log_softmax(log_accumulators, axis=2)

        # Pairwise product in forward pass
        child_log_prob = tf.expand_dims(child_log_prob, axis=3)
        weights = tf.expand_dims(tf.transpose(weights, (0, 1, 3, 2)), axis=2)

        pairwise_product = child_log_prob + weights

        # Max per sum for determining winning child + choosing the constant for numerical stability
        max_per_sum = tf.stop_gradient(tf.reduce_max(pairwise_product, axis=-1, keepdims=True))

        def grad(dy):
            # Figure out which sum is the winning one
            equal_to_max = tf.cast(tf.equal(pairwise_product, max_per_sum), tf.float32)
            num_sums = tf.shape(equal_to_max)[-1]
            equal_to_max_flat_outer = tf.reshape(equal_to_max, tf.concat([[-1], [num_sums]], axis=0))

            # Holds the index of the winning child per sum
            winning_child_per_sum = tf.reshape(
                tf.random.categorical(tf.math.log(equal_to_max_flat_outer), num_samples=1), tf.shape(equal_to_max)[:-1])

            # Pass on the counts to the edges between child and parent
            # TODO OR
            # pairwise_mult_counts = tf.scatter_nd(dy, winning_child_per_sum, tf.shape(equal_to_max))
            pairwise_mult_counts = tf.expand_dims(dy, axis=-1) * tf.one_hot(winning_child_per_sum, depth=num_sums)

            # Sum over parents to get counts per child
            child_counts = tf.reduce_sum(pairwise_mult_counts, axis=3)

            # Sum over batch to get counts per weight
            weight_counts = tf.reduce_sum(pairwise_mult_counts, axis=2)

            return child_counts, tf.transpose(weight_counts, (0, 1, 3, 2))

        # Perform log(sum(exp(...))) with the numerical stability trick
        out = tf.math.log(tf.reduce_sum(tf.exp(
            pairwise_product - max_per_sum), axis=-1)) + tf.squeeze(max_per_sum, axis=-1)

        return out, grad

    return _inner_fn(child_log_prob, accumulators)


def logmatmul(a, b):
    # Compute max for each tensor for numerical stability
    max_a = replace_infs_with_zeros(
        tf.stop_gradient(tf.reduce_max(a, axis=-1, keepdims=True)))
    max_b = replace_infs_with_zeros(
        tf.stop_gradient(tf.reduce_max(b, axis=-2, keepdims=True)))

    # Compute logsumexp using matrix mutiplication
    return tf.math.log(tf.matmul(tf.exp(a - max_a), tf.exp(b - max_b))) + max_a + max_b
