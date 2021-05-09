import abc
from typing import Optional
from typing import Tuple

import tensorflow as tf
from tensorflow.keras import initializers


class PoonDomingosQuantileSplitBase(abc.ABC, initializers.Initializer):
    """
    Initializes the data according to the algorithm described in (Poon and Domingos, 2011).

    The data is divided over :math:`K` quantiles where :math:`K` is the number of nodes along
    the last axis of the tensor to be initialized. The quantiles are computed over all samples
    in the provided ``data``.

    Args:
        data (numpy.ndarray): Data to compute quantiles over

    References:
        Sum-Product Networks, a New Deep Architecture
        `Poon and Domingos, 2011 <https://arxiv.org/abs/1202.3732>`_
    """

    def __init__(self, data: Optional[tf.data.Dataset] = None):
        self._data_sorted = None
        if data is not None:
            self.adapt(data)

    def adapt(self, data: tf.data.Dataset) -> None:
        """
        Adapt to given data.

        Internally sorts the data so that we can determine the splist easily later on.

        Args:
            data: Adapt this initializer to the data.
        """
        data_as_tensor = tf.concat([batch for batch in data], axis=0)
        self._data_sorted = tf.sort(data_as_tensor, axis=0)

    def __call__(
        self, shape: Tuple[Optional[int], ...], dtype: tf.dtypes.DType = None, **kwargs
    ) -> tf.Tensor:
        """
        Compute initializations along the last axis.

        Args:
            shape: Shape of Tensor to initialize.
            dtype: DType of Tensor to initialize.
            kwargs: Remaining kwargs.

        Returns:
            Initial value.

        Raises:
            RuntimeError: If data was not provided first.
        """
        if self._data_sorted is None:
            raise RuntimeError(
                "Data was not set so cannot determine statistics for Poon & Domingos initialization"
            )

        num_quantiles = shape[-2]

        batch_size = tf.shape(self._data_sorted)[0]
        default_split_size = tf.cast(tf.round(batch_size / num_quantiles), tf.int32)
        num_default_splits = (
            tf.cast(tf.math.ceil(batch_size / default_split_size), tf.int32) - 1
        )
        size_splits = tf.concat(
            [
                tf.repeat(default_split_size, num_default_splits),
                [batch_size - num_default_splits * default_split_size],
            ],
            axis=0,
        )
        values_per_quantile = tf.split(
            self._data_sorted, num_or_size_splits=size_splits, axis=0
        )
        reduced_splits = [self._reduce_split(v) for v in values_per_quantile]
        return tf.cast(tf.stack(reduced_splits, axis=-2), dtype=dtype)

    @abc.abstractmethod
    def _reduce_split(self, split_data: tf.Tensor):
        """
        Reduce data for a split.

        Args:
            split_data: Data for a particular split which should be reduced.
        """


class PoonDomingosMeanOfQuantileSplit(PoonDomingosQuantileSplitBase):
    """
    Initializes the data according to the algorithm described in (Poon and Domingos, 2011).

    The data is divided over :math:`K` quantiles where :math:`K` is the number of nodes along
    the last axis of the tensor to be initialized. The quantiles are computed over all samples
    in the provided ``data``. Then, the mean per quantile is taken as the value for
    initialization.

    Args:
        data (numpy.ndarray): Data to compute quantiles over

    References:
        Sum-Product Networks, a New Deep Architecture
        `Poon and Domingos, 2011 <https://arxiv.org/abs/1202.3732>`_
    """

    def _reduce_split(self, split_data: tf.Tensor):
        return tf.reduce_mean(split_data, axis=0, keepdims=True)


class PoonDomingosStddevOfQuantileSplit(PoonDomingosQuantileSplitBase):
    """
    Initializes the data according to the algorithm described in (Poon and Domingos, 2011).

    The data is divided over :math:`K` quantiles where :math:`K` is the number of nodes along
    the last axis of the tensor to be initialized. The quantiles are computed over all samples
    in the provided ``data``. Then, the stddev per quantile is taken as the value for
    initialization.

    Args:
        data (numpy.ndarray): Data to compute quantiles over

    References:
        Sum-Product Networks, a New Deep Architecture
        `Poon and Domingos, 2011 <https://arxiv.org/abs/1202.3732>`_
    """

    def _reduce_split(self, split_data: tf.Tensor):
        return tf.math.reduce_std(split_data, axis=0, keepdims=True)
