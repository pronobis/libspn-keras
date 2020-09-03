import abc
from typing import Optional, Tuple

import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp


class BaseLeaf(keras.layers.Layer, abc.ABC):
    """
    Computes probabilities from raw input.

    Args:
        num_components: Number of components per variable.
        dtype: DType of the input.
    """

    def __init__(self, num_components: int, dtype: tf.DType = tf.float32, **kwargs):
        super(BaseLeaf, self).__init__(dtype=dtype, **kwargs)
        self.num_components = num_components

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the internal components for this leaf layer.

        Args:
            input_shape: Shape of the input Tensor.
        """
        _, *scope_dims, multivariate_size = input_shape
        distribution_shape = 1, *scope_dims, self.num_components, multivariate_size
        self._num_scopes, self._num_decomps = scope_dims
        self._build_distribution(distribution_shape)
        super(BaseLeaf, self).build(input_shape)

    @abc.abstractmethod
    def _build_distribution(self, shape: Tuple[Optional[int], ...]) -> None:
        pass

    @abc.abstractmethod
    def _get_distribution(self) -> tfp.distributions.Distribution:
        pass

    def call(self, x: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Compute the probability of the leaf nodes.

        Args:
            x: Spatial or region Tensor with raw input values.
            kwargs: Remaining keyword arguments.

        Returns:
            A Tensor with the probabilities per component.
        """
        x = tf.expand_dims(x, axis=-2)
        distribution = self._get_distribution()
        return tf.reduce_sum(distribution.log_prob(x), axis=-1)

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
        *outer_dims, _ = input_shape
        return (*outer_dims, self.num_components)

    def get_config(self) -> dict:
        """
        Obtain a key-value representation of the layer config.

        Returns:
            A dict holding the configuration of the layer.
        """
        config = dict(num_components=self.num_components)
        base_config = super(BaseLeaf, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def get_modes(self) -> tf.Tensor:
        """
        Obtain the distribution modes.

        This can be used for e.g. MPE estimates of inputs.

        Raises:
            NotImplementedError: Not all descendants of BaseLeaf have this implemented.Not
        """
        raise NotImplementedError(
            "A {} does not implement distribution modes.".format(
                self.__class__.__name__
            )
        )
