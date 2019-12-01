import tensorflow as tf


def log_softmax_from_accumulators_with_em_grad(accumulator, axis=None):
    """
    Implements custom gradient to pass on the weight grads to an accumulator

    Args:
        accumulator: A `Tensor` holding the weight accumulator in linear space
        axis: Axis to reduce log softmax

    Returns:
        A `Tensor` containing normalized logspace weights
    """

    @tf.custom_gradient
    def _inner(accumulator):

        def grad(dy):
            return dy

        out = tf.nn.log_softmax(tf.math.log(accumulator), axis=axis)

        return out, grad

    return _inner(accumulator)
