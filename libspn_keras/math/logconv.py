import tensorflow as tf

from libspn_keras.math.logutils import replace_infs_with_zeros


def logconv1x1_2d(input: tf.Tensor, filter: tf.Tensor) -> tf.Tensor:
    r"""
    Convolution in logspace with 1x1 filters.

    Computes :math:`\log(\text{conv}(\mathtt{input},\mathtt{filter}))` from
    :math:`\log(\mathtt{input})` and :math:`\log(\mathtt{filter})`

    Args:
        input: Input in logspace
        filter: Filter in logspace

    Returns:
        Convolution of input and filter in logspace
    """
    with tf.name_scope("LogConv1x1"):
        filter_max = replace_infs_with_zeros(
            tf.stop_gradient(tf.reduce_max(filter, axis=-2, keepdims=True))
        )
        input_max = replace_infs_with_zeros(
            tf.stop_gradient(tf.reduce_max(input, axis=-1, keepdims=True))
        )

        filter -= filter_max
        input -= input_max

        out = tf.math.log(
            tf.nn.convolution(
                input=tf.exp(input), filters=tf.exp(filter), padding="SAME"
            )
        )
        out += filter_max + input_max

        return out
