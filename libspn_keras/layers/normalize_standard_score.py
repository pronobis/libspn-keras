from typing import Optional, Tuple, Union

import tensorflow as tf
from tensorflow import keras


class NormalizeStandardScore(keras.layers.Layer):
    """
    Normalizes samples to a standard score.

    In other words, the output is the input minus its mean and divided by the standard deviation.
    This can be used to achieve the same kind of normalization as used in (Poon and Domingos, 2011).

    Args:
        normalization_epsilon (float): Small positive constant to prevent division by zero,
            but could also be used a 'smoothing' factor.
        **kwargs: kwargs to pass on to the keras.Layer super class

    References:
        Sum-Product Networks, a New Deep Architecture
        `Poon and Domingos, 2011 <https://arxiv.org/abs/1202.3732>`_
    """

    def __init__(self, normalization_epsilon: float = 1e-8, **kwargs):
        super(NormalizeStandardScore, self).__init__(**kwargs)
        self.normalization_epsilon = normalization_epsilon

    def call(
        self, x: tf.Tensor, return_stats: bool = False, **kwargs
    ) -> Union[Tuple[tf.Tensor, ...], tf.Tensor]:
        """
        Normalize raw input by subtracting mean and dividing by the standard deviation per sample.

        Args:
            x: Raw input tensor.
            return_stats: Whether to return mean and standard edviation
            kwargs: Remaining keyword arguments.

        Returns:
            A normalized tensor or a tuple of the normalized tensor, the mean and the standard deviation.
        """
        data_input = x
        normalization_axes_indices = tf.range(1, tf.rank(data_input))
        mean = tf.reduce_mean(
            data_input, axis=normalization_axes_indices, keepdims=True
        )
        stddev = tf.math.reduce_std(
            data_input, axis=normalization_axes_indices, keepdims=True
        )
        normalized_input = (data_input - mean) / (stddev + self.normalization_epsilon)
        if return_stats:
            return normalized_input, mean, stddev
        return normalized_input

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute output shape of the layer.

        Args:
            input_shape: Input shape of the layer.

        Returns:
            Tuple of ints holding the output shape of the layer.
        """
        return input_shape

    def get_config(self) -> dict:
        """
        Obtain a key-value representation of the layer config.

        Returns:
            A dict holding the configuration of the layer.
        """
        config = dict(normalization_epsilon=self.normalization_epsilon)
        base_config = super(NormalizeStandardScore, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
