import tensorflow as tf

from libspn_keras.math.logutils import replace_infs_with_zeros


def logmatmul(log_a: tf.Tensor, log_b: tf.Tensor) -> tf.Tensor:
    """
    Matrix multiplication in log-space.

    Args:
        log_a: log(a) of shape [..., batch, num_in]
        log_b: log(b) of shape [..., num_in, num_out]

    Returns:
        A matrix log(c) where log(c) = log(a @ b)
    """
    with tf.name_scope("LogMatmul"):
        # Compute max for each tensor for numerical stability
        max_a = replace_infs_with_zeros(
            tf.stop_gradient(tf.reduce_max(log_a, axis=-1, keepdims=True))
        )
        max_b = replace_infs_with_zeros(
            tf.stop_gradient(tf.reduce_max(log_b, axis=-2, keepdims=True))
        )

        # Compute logsumexp using matrix multiplication
        return (
            tf.math.log(tf.matmul(tf.exp(log_a - max_a), tf.exp(log_b - max_b)))
            + max_a
            + max_b
        )
