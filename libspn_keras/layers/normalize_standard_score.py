from typing import Optional, Tuple, Union

import tensorflow as tf
from tensorflow import keras


class NormalizeStandardScore(keras.layers.Layer):
    """
    Normalizes samples to a standard score.

    In other words, the output is the input minus its mean and divided by the standard deviation.
    This can be used to achieve the same kind of normalization as used in (Poon and Domingos, 2011).

    Args:
        normalization_epsilon: Small positive constant to prevent division by zero,
            but could also be used a 'smoothing' factor.
        sample_wise: If ``True``, will compute z-scores where statistics are computed sample-wise, otherwise,
            computes z-scores through computing cross-sample statistics. To use cross-sample statistics, call
            ``NormalizeStandardScore.adapt(train_ds)`` where ``train_ds`` is an instance of
            ``tf.data.Dataset``.
        **kwargs: kwargs to pass on to the keras.Layer super class

    References:
        Sum-Product Networks, a New Deep Architecture
        `Poon and Domingos, 2011 <https://arxiv.org/abs/1202.3732>`_
    """

    def __init__(
        self, normalization_epsilon: float = 1e-8, sample_wise: bool = True, **kwargs
    ):
        super(NormalizeStandardScore, self).__init__(**kwargs)
        self.normalization_epsilon = normalization_epsilon
        self.sample_wise = sample_wise
        self.mean = None
        self.stddev = None

    def adapt(self, ds: tf.data.Dataset) -> None:
        """
        Compute cross-sample statistics. Assumes that ``ds`` is a batched dataset.

        Args:
            ds: Instance of ``tf.data.Dataset`` containing train data to compute statistics from.
        """
        shape = tf.shape(next(ds.take(1).as_numpy_iterator())[0])
        sum_x = tf.zeros(shape)
        sum_x2 = tf.zeros(shape)
        count = tf.constant(0.0)

        for x in ds:
            sum_x += tf.reduce_sum(tf.cast(x, tf.float32), axis=0)
            sum_x2 += tf.reduce_sum(tf.square(tf.cast(x, tf.float32)), axis=0)
            count += tf.cast(tf.shape(x)[0], tf.float32)

        self.mean = self.add_weight(
            "mean",
            initializer=tf.keras.initializers.Constant(sum_x / count),
            trainable=False,
            shape=shape,
        )
        self.stddev = self.add_weight(
            "stddev",
            initializer=tf.keras.initializers.Constant(
                tf.sqrt(sum_x2 / count - tf.square(sum_x / count))
            ),
            shape=shape,
            trainable=False,
        )

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

        Raises:
            ValueError: if ``NoramalizeStandardScore.adapt(train_ds)`` wasn't called before calling this
                layer's forward pass.
        """
        if self.sample_wise:
            normalization_axes_indices = tf.range(1, tf.rank(x))
            mean = tf.reduce_mean(x, axis=normalization_axes_indices, keepdims=True)
            stddev = tf.math.reduce_std(
                x, axis=normalization_axes_indices, keepdims=True
            )
        else:
            if self.mean is None or self.stddev is None:
                raise ValueError(
                    "Cannot compute normalized values before calling adapt()"
                )
            mean = self.mean
            stddev = self.stddev
        normalized_input = (x - mean) / (stddev + self.normalization_epsilon)
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
