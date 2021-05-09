from enum import Enum
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union

import tensorflow as tf
from tensorflow import keras


class NormalizeAxes(Enum):
    """
    Enum for normalization axes.

    Enumerates possible choices for normalization axes. :attr:`~libspn_keras.layers.NormalizeAxes.SAMPLE`
    orresponds to normalizing each sample. :attr:`~libspn_keras.layers.NormalizeAxes.VARIABLE_WISE` is
    for normalizing the values for each variable using statistics across all samples, while
    :attr:`~libspn_keras.layers.NormalizeAxes.GLOBAL` corresponds to statistics gathered from all input
    values (no specific axes excluded from reduction).
    """

    SAMPLE_WISE = "sample-wise"
    """Normalize each sample"""

    VARIABLE_WISE = "variable-wise"
    """Normalize each variable"""

    GLOBAL = "global"
    """Normalize using all variables across all samples"""


class NormalizeStandardScore(keras.layers.Layer):
    """
    Normalizes samples to a standard score.

    In other words, the output is the input minus its mean and divided by the standard deviation.
    This can be used to achieve the same kind of normalization as used in (Poon and Domingos, 2011).

    Args:
        normalization_epsilon: Small positive constant to prevent division by zero,
            but could also be used a 'smoothing' factor.
        axes: If ``NormalizationAxes.SAMPLE_WISE``, will compute z-scores where statistics are computed
            sample-wise, otherwise, computes z-scores through computing cross-sample statistics. To use
            cross-sample statistics, call ``NormalizeStandardScore.adapt(train_ds)`` where ``train_ds``
            is an instance of ``tf.data.Dataset``.
        **kwargs: kwargs to pass on to the keras.Layer super class

    References:
        Sum-Product Networks, a New Deep Architecture
        `Poon and Domingos, 2011 <https://arxiv.org/abs/1202.3732>`_
    """

    def __init__(
        self,
        normalization_epsilon: float = 1e-8,
        axes: NormalizeAxes = NormalizeAxes.SAMPLE_WISE,
        **kwargs
    ):
        super(NormalizeStandardScore, self).__init__(**kwargs)
        self.normalization_epsilon = normalization_epsilon
        self.axes = axes
        self.mean = None
        self.stddev = None

    def adapt(self, ds: tf.data.Dataset) -> None:
        """
        Compute cross-sample statistics. Assumes that ``ds`` is a batched dataset.

        Args:
            ds: Instance of ``tf.data.Dataset`` containing train data to compute statistics from.

        Raises:
            RuntimeError: Raised when axes are set to SAMPLE_WISE, in which case there is no point in
                calling adapt.
        """
        if self.axes == NormalizeAxes.SAMPLE_WISE:
            raise RuntimeError("No point in adapting when axes are SAMPLE_WISE")

        axis: Optional[int]
        if self.axes == NormalizeAxes.VARIABLE_WISE:
            axis = 0
            shape = tf.shape(next(ds.take(1).as_numpy_iterator())[0])
        else:
            axis = None
            shape = ()
        mean_x = tf.zeros(shape)
        mean_x2 = tf.zeros(shape)
        count = tf.constant(0.0)

        for x in ds:
            elem_mean_x = tf.reduce_mean(tf.cast(x, tf.float32), axis=axis)
            elem_mean_x2 = tf.reduce_mean(tf.square(tf.cast(x, tf.float32)), axis=axis)
            elem_count = tf.cast(tf.size(x) / tf.size(elem_mean_x), tf.float32)
            new_count = count + elem_count
            mean_x = mean_x * (count / new_count) + elem_mean_x * (
                elem_count / new_count
            )
            mean_x2 = mean_x2 * (count / new_count) + elem_mean_x2 * (
                elem_count / new_count
            )
            count = new_count

        self.mean = self.add_weight(
            "mean",
            initializer=tf.keras.initializers.Constant(mean_x),
            trainable=False,
            shape=shape,
        )
        self.stddev = self.add_weight(
            "stddev",
            initializer=tf.keras.initializers.Constant(
                tf.sqrt(mean_x2 - tf.square(mean_x))
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
            return_stats: Whether to return mean and standard deviation
            kwargs: Remaining keyword arguments.

        Returns:
            A normalized tensor or a tuple of the normalized tensor, the mean and the standard deviation.

        Raises:
            ValueError: if ``NoramalizeStandardScore.adapt(train_ds)`` wasn't called before calling this
                layer's forward pass.
        """
        if self.axes == NormalizeAxes.SAMPLE_WISE:
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

    @classmethod
    def from_config(
        cls: Type["NormalizeStandardScore"], config: dict
    ) -> "NormalizeStandardScore":
        """Create layer from its config.

        This method is the reverse of `get_config`,
        capable of instantiating the same layer from the config
        dictionary. It does not handle layer connectivity
        (handled by Network), nor weights (handled by `set_weights`).

        Arguments:
            config: A Python dictionary, typically the
                output of get_config.

        Returns:
            A layer instance.
        """
        axes = NormalizeAxes(config.pop("axes", "sample-wise"))
        return cls(axes=axes, **config)

    def get_config(self) -> dict:
        """
        Obtain a key-value representation of the layer config.

        Returns:
            A dict holding the configuration of the layer.
        """
        config = dict(
            normalization_epsilon=self.normalization_epsilon, axes=self.axes.value
        )
        base_config = super(NormalizeStandardScore, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
