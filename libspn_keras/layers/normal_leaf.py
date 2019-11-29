from libspn_keras.layers.base_leaf import BaseLeaf
import tensorflow_probability as tfp
from tensorflow import initializers


class NormalLeaf(BaseLeaf):

    def _build_distribution(self):
        shape = [self._num_scopes, self._num_decomps, 1, self._num_components]
        self._loc = self.add_weight(
            name="Location", shape=shape, initializer=initializers.TruncatedNormal(stddev=0.5))
        self._scale = self.add_weight(
            name="Scale", shape=shape, initializer=initializers.Ones(), trainable=False)
        return tfp.distributions.Normal(loc=self._loc, scale=self._scale)