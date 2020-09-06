from tensorflow.keras.initializers import Initializer, TruncatedNormal

_DEFAULT_ACCUMULATOR_INITIALIZER = TruncatedNormal(stddev=0.5, mean=1.0)


def set_default_accumulator_initializer(initializer: Initializer) -> None:
    """
    Configure the default accumulator that will be used for sum accumulators.

    Args:
        initializer: The initializer which will be used by default for sum accumulators.
    """
    global _DEFAULT_ACCUMULATOR_INITIALIZER
    _DEFAULT_ACCUMULATOR_INITIALIZER = initializer


def get_default_accumulator_initializer() -> Initializer:
    """
    Obtain default accumulator initializer.

    Returns:
        The default accumulator initializer that will be use in sum accumulators, unless specified
        explicitly at initialization.
    """
    global _DEFAULT_ACCUMULATOR_INITIALIZER
    return _DEFAULT_ACCUMULATOR_INITIALIZER
