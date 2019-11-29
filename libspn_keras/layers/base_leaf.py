from tensorflow import keras


class BaseLeaf(keras.layers.Layer):

    def __init__(self, num_components):
        super(BaseLeaf, self).__init__()
        self._num_components = num_components
        self._distribution = self._num_scopes = self._num_decomps = None

    def build(self, input_shape):
        self._num_scopes, self._num_decomps, _, _ = input_shape
        self._distribution = self._build_distribution()
        super(BaseLeaf, self).build(input_shape)

    def _build_distribution(self):
        raise NotImplementedError("Must implement in descendant class")

    @property
    def num_components(self):
        return self._num_components

    def call(self, x):
        return self._distribution.log_prob(x)

    def compute_output_shape(self, input_shape):
        num_scopes, num_decomps, num_batch, _ = input_shape
        return num_scopes, num_decomps, num_batch, self._num_components

