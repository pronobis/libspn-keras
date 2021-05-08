import functools
from typing import Callable

import tensorflow as tf

from libspn_keras.sum_ops.base import SumOpBase


def batch_scope_transpose(f: Callable) -> Callable:  # type: ignore  # noqa: ANN001,ANN202
    """
    Transpose batch and scope dimension.

    Args:
        f: function to decorate

    Returns:
        Decorated function that transposes batch and scope dimension and retransposes output
    """

    @functools.wraps(f)  # type: ignore  # noqa: ANN202
    def impl(self: SumOpBase, x: tf.Tensor, *args, **kwargs) -> tf.Tensor:  # type: ignore
        with tf.name_scope("ScopesAndDecompsFirst"):
            scopes_decomps_first = tf.transpose(x, (1, 2, 0, 3))
        result = f(self, scopes_decomps_first, *args, **kwargs)
        with tf.name_scope("BatchFirst"):
            return tf.transpose(result, (2, 0, 1, 3))

    return impl
