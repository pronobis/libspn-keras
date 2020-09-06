from typing import Callable, Optional, Tuple

import tensorflow as tf
from tensorflow import initializers


def logspace_wrapper_initializer(
    initializer: initializers.Initializer,
) -> Callable[[Tuple[Optional[int], ...], tf.dtypes.DType], tf.Tensor]:
    """
    Wrap an initializer so that its values are projected in log-space.

    Args:
        initializer: The initializer to convert to logspace

    Returns:
        A initialization callable that produces the log-space representation of `initializer`
    """

    def _wrap_fn(
        shape: Tuple[Optional[int], ...], dtype: tf.dtypes.DType = None
    ) -> tf.Tensor:
        return tf.math.log(initializer(shape=shape, dtype=dtype))

    return _wrap_fn
