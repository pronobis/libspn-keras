from tensorflow import initializers


def logspace_wrapper_initializer(accumulator):

    if not isinstance(accumulator, initializers.Initializer):
        raise ValueError(
            "Accumulator must be of type {}".format(initializers.Initializer.__class__))

    def wrap_fn(shape, dtype=None):
        return accumulator(shape=shape, dtype=dtype)

    return wrap_fn
