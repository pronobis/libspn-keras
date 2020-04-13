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