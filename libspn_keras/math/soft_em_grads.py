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


class LocationScaleEMGradWrapper:

    def __init__(self, location_scale_distribution, accumulator, data_accumulator):
        self.location_scale_distribution = location_scale_distribution
        self.accumulator = accumulator
        self.data_accumulator = data_accumulator

    def __getattr__(self, attr):
        # see if this object has attr
        # NOTE do not use hasattr, it goes into
        # infinite recursion
        if attr in self.__dict__:
            # this object has it
            return getattr(self, attr)

        # proxy to the wrapped object
        return getattr(self.location_scale_distribution, attr)

    def log_prob(self, x):

        @tf.custom_gradient
        def _inner(accumulator, data_accumulator):

            def grad(dy):
                return tf.reduce_sum(dy, axis=0, keepdims=True), \
                       tf.reduce_sum(x * dy, axis=0, keepdims=True)

            out = self.location_scale_distribution.log_prob(x)

            return out, grad

        return _inner(self.accumulator, self.data_accumulator)

