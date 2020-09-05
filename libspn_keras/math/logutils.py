import tensorflow as tf


def replace_infs_with_zeros(x: tf.Tensor) -> tf.Tensor:
    """
    Replace infinite values with zeroes.

    Args:
        x: A `Tensor`.

    Returns:
        A `Tensor` with the same shape an dtype as `x` where infinite values are replaced
        with zeroes.
    """
    with tf.name_scope("ReplaceInfsWithZeroes"):
        return tf.where(tf.math.is_inf(x), tf.zeros_like(x), x)
