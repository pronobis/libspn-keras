import tensorflow as tf


def replace_infs_with_zeros(x):
    return tf.where(tf.math.is_inf(x), tf.zeros_like(x), x)


def logmatmul(a, b):
    # Compute max for each tensor for numerical stability
    max_a = replace_infs_with_zeros(
        tf.stop_gradient(tf.reduce_max(a, axis=-1, keepdims=True)))
    max_b = replace_infs_with_zeros(
        tf.stop_gradient(tf.reduce_max(b, axis=-2, keepdims=True)))

    # Compute logsumexp using matrix mutiplication
    return tf.math.log(tf.matmul(tf.exp(a - max_a), tf.exp(b - max_b))) + max_a + max_b
