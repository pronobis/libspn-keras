from tensorflow import keras
import tensorflow as tf

from libspn_keras.dimension_permutation import DimensionPermutation, infer_dimension_permutation


class BaseLeaf(keras.layers.Layer):

    def __init__(
        self, num_components, dtype=tf.float32, dimension_permutation=DimensionPermutation.AUTO,
        use_cdf=False
    ):
        super(BaseLeaf, self).__init__(dtype=dtype)
        self.num_components = num_components
        self.dimension_permutation = dimension_permutation
        self.use_cdf = use_cdf
        self._distribution = self._num_scopes = self._num_decomps = None

    def build(self, input_shape):
        inferred_dimension_permutation = infer_dimension_permutation(input_shape) \
            if self.dimension_permutation == DimensionPermutation.AUTO \
            else self.dimension_permutation
        if inferred_dimension_permutation == DimensionPermutation.BATCH_FIRST:
            _, *scope_and_decomp_dims, _ = input_shape
            distribution_shape = [1] + scope_and_decomp_dims + [self.num_components]
        else:
            *scope_and_decomp_dims, _, _ = input_shape
            distribution_shape = scope_and_decomp_dims + [1, self.num_components]
        self._num_scopes, self._num_decomps = scope_and_decomp_dims
        self._distribution = self._build_distribution(distribution_shape)
        super(BaseLeaf, self).build(input_shape)

    def _build_distribution(self, shape):
        raise NotImplementedError()

    def call(self, x):
        return self._distribution.log_cdf(x) if self.use_cdf else self._distribution.log_prob(x)

    def compute_output_shape(self, input_shape):

        inferred_dimension_permutation = infer_dimension_permutation(input_shape) \
            if self.dimension_permutation == DimensionPermutation.AUTO \
            else self.dimension_permutation
        if inferred_dimension_permutation == DimensionPermutation.BATCH_FIRST:
            _, *scope_and_decomp_dims, _ = input_shape
            out_shape = (None,) + scope_and_decomp_dims + (self.num_components,)
        else:
            *scope_and_decomp_dims, _, _ = input_shape
            out_shape = scope_and_decomp_dims + (None, self.num_components,)

        return out_shape

    def get_config(self):
        config = dict(
            num_components=self.num_components
        )
        base_config = super(BaseLeaf, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
