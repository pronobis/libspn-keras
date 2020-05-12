from tensorflow import keras
import tensorflow as tf


class BaseLeaf(keras.layers.Layer):

    def __init__(
        self, num_components, dtype=tf.float32, use_cdf=False, multivariate=False, **kwargs
    ):
        super(BaseLeaf, self).__init__(dtype=dtype, **kwargs)
        self.num_components = num_components
        self.use_cdf = use_cdf
        self.multivariate = multivariate
        self._num_scopes = self._num_decomps = None

    def build(self, input_shape):
        _, *scope_dims, multivariate_size = input_shape
        distribution_shape = [1] + scope_dims + [self.num_components, multivariate_size]
        self._num_scopes, self._num_decomps = scope_dims
        self._build_distribution(distribution_shape)
        super(BaseLeaf, self).build(input_shape)

    def _build_distribution(self, shape):
        raise NotImplementedError("Implement distribution build in descendant class")

    def _get_distribution(self):
        raise NotImplementedError("Implement distribution in descendant class")

    def call(self, x):
        x = tf.expand_dims(x, axis=-2)
        distribution = self._get_distribution()
        log_prob = distribution.log_cdf(x) if self.use_cdf else distribution.log_prob(x)
        return tf.reduce_sum(log_prob, axis=-1)

    def compute_output_shape(self, input_shape):
        *outer_dims, _ = input_shape
        out_shape = outer_dims + [self.num_components]
        return out_shape

    def get_config(self):
        config = dict(
            num_components=self.num_components,
            use_cdf=self.use_cdf
        )
        base_config = super(BaseLeaf, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def get_modes(self):
        raise NotImplementedError("A {} does not implement distribution modes.".format(self.__class__.__name__))
