from tensorflow import keras
import tensorflow as tf

from libspn_keras.dimension_permutation import DimensionPermutation, infer_dimension_permutation


class BaseLeaf(keras.layers.Layer):

    def __init__(
        self, num_components, dtype=tf.float32, dimension_permutation=DimensionPermutation.AUTO,
        use_cdf=False, multivariate=False, **kwargs
    ):
        super(BaseLeaf, self).__init__(dtype=dtype, **kwargs)
        self.num_components = num_components
        self.dimension_permutation = dimension_permutation
        self.use_cdf = use_cdf
        self.multivariate = multivariate
        self._distribution = self._num_scopes = self._num_decomps = None

    def build(self, input_shape):
        effective_dimension_permutation = infer_dimension_permutation(input_shape) \
            if self.dimension_permutation == DimensionPermutation.AUTO \
            else self.dimension_permutation

        if effective_dimension_permutation == DimensionPermutation.SPATIAL:
            _, *scope_and_decomp_dims, multivariate_size = input_shape
            distribution_shape = [1] + scope_and_decomp_dims + [
                self.num_components, multivariate_size]
        else:
            *scope_and_decomp_dims, _, multivariate_size = input_shape
            distribution_shape = scope_and_decomp_dims + \
                 [1, self.num_components, multivariate_size]

        self._num_scopes, self._num_decomps = scope_and_decomp_dims
        self._distribution = self._build_distribution(distribution_shape)
        super(BaseLeaf, self).build(input_shape)

    def _build_distribution(self, shape):
        raise NotImplementedError("Implement distribution build in descendant class")

    def call(self, x):
        x = tf.expand_dims(x, axis=-2)
        log_prob = self._distribution.log_cdf(x) if self.use_cdf \
            else self._distribution.log_prob(x)
        return tf.reduce_sum(log_prob, axis=-1)

    def compute_output_shape(self, input_shape):

        inferred_dimension_permutation = infer_dimension_permutation(input_shape) \
            if self.dimension_permutation == DimensionPermutation.AUTO \
            else self.dimension_permutation
        if inferred_dimension_permutation == DimensionPermutation.SPATIAL:
            _, *scope_and_decomp_dims, _ = input_shape
            out_shape = [None] + scope_and_decomp_dims + [self.num_components]
        else:
            *scope_and_decomp_dims, _, _ = input_shape
            out_shape = scope_and_decomp_dims + [None, self.num_components]

        return out_shape

    def get_config(self):
        config = dict(
            num_components=self.num_components,
            dimension_permutation=self.dimension_permutation,
            use_cdf=self.use_cdf
        )
        base_config = super(BaseLeaf, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def get_modes(self):
        raise NotImplementedError("A {} does not implement distribution modes.".format(self.__class__.__name__))
