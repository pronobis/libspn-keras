from tensorflow.keras import initializers
import numpy as np
import tensorflow as tf


class PoonDomingosMeanOfQuantileSplit(initializers.Initializer):
    """
    Initializes the data according to the algorithm described in (Poon and Domingos, 2011).

    The data is divided over :math:`K` quantiles where :math:`K` is the number of nodes along
    the last axis of the tensor to be initialized. The quantiles are computed over all samples
    in the provided ``data``. Then, the mean per quantile is taken as the value for
    initialization.

    Args:
        data (numpy.ndarray): Data to compute quantiles over
        samplewise_normalization: Whether to 'Z-score normalize' the data sample-wise before
            computing the quantiles and means.
        normalization_epsilon: Non-zero constant to account for near-zero standard deviations in
            normalizations.

    References:
        Sum-Product Networks, a New Deep Architecture
        `Poon and Domingos, 2011 <https://arxiv.org/abs/1202.3732>`_
    """

    def __init__(self, data=None, samplewise_normalization=True, normalization_epsilon=1e-2):
        self._data = data
        self.samplewise_normalization = samplewise_normalization
        self.normalization_epsilon = normalization_epsilon

    def __call__(self, shape, dtype=None, partition_info=None):

        num_quantiles = shape[-2]

        if self.samplewise_normalization:
            axes = tuple(range(1, len(self._data.shape)))
            data = (self._data - np.mean(self._data, axis=axes, keepdims=True)) \
                   / (np.std(self._data, axis=axes, keepdims=True) + self.normalization_epsilon)
        else:
            data = self._data

        batch_size = data.shape[0]
        quantile_sections = np.arange(
            batch_size // num_quantiles, batch_size, int(np.ceil(batch_size / num_quantiles)))
        sorted_features = np.sort(data, axis=0)
        values_per_quantile = np.split(
            sorted_features, indices_or_sections=quantile_sections, axis=0)
        means_per_quantile = [np.mean(v, axis=0, keepdims=True) for v in values_per_quantile]
        return tf.expand_dims(tf.cast(np.stack(means_per_quantile, axis=-1), dtype=dtype), axis=-1)

    def get_config(self):
        return {
            "samplewise_normalization": self.samplewise_normalization,
            "normalization_epsilon": self.normalization_epsilon
        }
