import tensorflow as tf
from tensorflow import keras

from libspn_keras.normalizationaxes import NormalizationAxes


class ZScoreNormalization(keras.layers.Layer):
    """
    Normalizes the input along the specified axes. Currently only supports ``PER_SAMPLE`` axes. This
    can be used to achieve the same kind of normalization as used in (Poon and Domingos, 2011)

    Args:
        with_evidence_mask: If True, will account for the evidence as given as the second
            input of this layer when computing the mean and stddev.
        axes: Normalization axes. Currently only supports ``NormalizationAxes.PER_SAMPLE``
        normalization_epsilon: Small positive constant to prevent division by zero, but could
            also be used a 'smoothing' factor.
        **kwargs: kwargs to pass on to the keras.Layer super class

    References:
        Sum-Product Networks, a New Deep Architecture
        `Poon and Domingos, 2011 <https://arxiv.org/abs/1202.3732>`_
    """

    def __init__(self, axes=NormalizationAxes.PER_SAMPLE, with_evidence_mask=False,
                 normalization_epsilon=1e-8, return_mean_and_stddev=False, **kwargs):
        # TODO verify whether value of axes is allowed or just omit it alltogether
        super(ZScoreNormalization, self).__init__(**kwargs)
        self.axes = axes
        self.with_evidence_mask = with_evidence_mask
        self.normalization_epsilon = normalization_epsilon
        self.return_mean_and_stddev = return_mean_and_stddev

    def call(self, x):
        if self.with_evidence_mask:
            data_input, evidence_mask = x
        else:
            data_input, evidence_mask = x, None
        if self.axes == NormalizationAxes.PER_SAMPLE:
            normalization_axes_indices = tf.range(1, tf.rank(data_input))
            if evidence_mask is None:
                n = tf.cast(tf.reduce_prod(tf.shape(data_input)[1:]), data_input.dtype)
            else:
                n = tf.reduce_sum(
                    tf.cast(evidence_mask, data_input.dtype),
                    axis=normalization_axes_indices, keepdims=True
                )

            if evidence_mask is not None:
                data_input *= tf.cast(evidence_mask, data_input.dtype)

            mean = tf.reduce_sum(
                data_input, axis=normalization_axes_indices, keepdims=True) / n

            sq_diff = tf.math.squared_difference(data_input, mean)
            if evidence_mask is not None:
                sq_diff *= tf.cast(evidence_mask, data_input.dtype)

            stddev = tf.sqrt(
                tf.reduce_sum(sq_diff, axis=normalization_axes_indices, keepdims=True) / n)
            normalized_input = (data_input - mean) / (stddev + self.normalization_epsilon)
        else:
            raise ValueError("Normalization axes other than PER_SAMPLE not supported")
        return normalized_input, mean, stddev

    def compute_output_shape(self, input_shape):
        if self.with_evidence_mask:
            data_shape, _ = input_shape
        else:
            data_shape = input_shape

        if self.axes == NormalizationAxes.PER_SAMPLE:
            stats_shape = data_shape[:1] + [1 for _ in range(1, len(data_shape))]
            return data_shape, stats_shape, stats_shape
        else:
            ValueError("Unknown normalization axes")

    def get_config(self):
        config = dict(
            normalization_epsilon=self.normalization_epsilon,
            with_evidence_mask=self.with_evidence_mask,
            axes=self.axes,
            return_mean_and_stddev=self.return_mean_and_stddev
        )
        base_config = super(ZScoreNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
