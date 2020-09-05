from typing import Optional, Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.utils import tf_utils


class LogDropout(keras.layers.Layer):
    """
    Log dropout layer.

    Applies dropout in log-space. Should not precede product layers in an SPN, since their scope probability
    then potentially becomes -inf, resulting in NaN-values during training.

    Args:
        rate: Rate at which to randomly dropout inputs.
        noise_shape: Shape of dropout noise tensor
        seed: Random seed
        **kwargs: kwargs to pass on to the keras.Layer super class
    """

    def __init__(
        self,
        rate: float,
        noise_shape: Optional[Tuple[int, ...]] = None,
        seed: int = None,
        axis_at_least_one: Optional[int] = None,
        **kwargs
    ):

        super(LogDropout, self).__init__(**kwargs)
        self.rate = min(1.0, max(0.0, rate))
        self.noise_shape = noise_shape
        self.seed = seed
        self.supports_masking = True
        self.axis_at_least_one = axis_at_least_one

    def _get_noise_shape(self, inputs: tf.Tensor) -> tf.Tensor:
        if self.noise_shape is None:
            return tf.shape(inputs)

        return tf.concat([[tf.shape(inputs)[0]], self.noise_shape], axis=0)

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """
        Randomly drop out parts of incoming tensor by setting the log probability to -Inf.

        Args:
            inputs: Input log-probability Tensor.
            training: Whether or not the network is training.

        Returns:
            Tensor with some nodes dropped out.
        """
        if training is None:
            training = keras.backend.learning_phase()

        def dropped_inputs() -> tf.Tensor:
            noise_tensor = tf.random.uniform(
                shape=self._get_noise_shape(inputs), seed=self.seed
            )
            keep_tensor = tf.greater(noise_tensor, self.rate)
            if self.axis_at_least_one is not None:
                keep_tensor = tf.logical_or(
                    keep_tensor,
                    tf.equal(
                        noise_tensor,
                        tf.reduce_max(
                            noise_tensor, axis=self.axis_at_least_one, keepdims=True
                        ),
                    ),
                )
            return tf.reshape(
                tf.where(keep_tensor, inputs, float("-inf")), shape=tf.shape(inputs)
            )

        if self.rate == 0.0:
            return tf.identity(inputs)

        output = tf_utils.smart_cond(
            training, dropped_inputs, lambda: tf.identity(inputs)
        )
        return output

    def get_config(self) -> dict:
        """
        Obtain a key-value representation of the layer config.

        Returns:
            A dict holding the configuration of the layer.
        """
        config = dict(rate=self.rate, noise_shape=self.noise_shape, seed=self.seed)
        base_config = super(LogDropout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

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
