from libspn_keras.layers.base_leaf import BaseLeaf
import tensorflow_probability as tfp
from tensorflow import initializers
import tensorflow as tf


class NormalLeaf(BaseLeaf):

    def __init__(
        self, num_components, dtype=tf.float32, location_initializer=None,
        location_trainable=True, scale_initializer=None, scale_trainable=False,
        use_cdf=False
    ):
        super(NormalLeaf, self).__init__(
            num_components=num_components, dtype=dtype, use_cdf=use_cdf)
        self.location_initializer = location_initializer or initializers.TruncatedNormal(stddev=1.0)
        self.location_trainable = location_trainable
        self.scale_initializer = scale_initializer or initializers.Ones()
        self.scale_trainable = scale_trainable
        self._distribution = self._num_scopes = self._num_decomps = None

    def _build_distribution(self, shape):
        # shape = [self._num_scopes, self._num_decomps, 1, self.num_components]
        loc = self.add_weight(
            name="location", shape=shape, initializer=self.location_initializer)
        scale = self.add_weight(
            name="scale", shape=shape, initializer=self.scale_initializer, trainable=False)
        return tfp.distributions.Normal(loc=loc, scale=scale)

    def get_config(self):
        config = dict(
            scale_initializer=initializers.serialize(self.scale_initializer),
            location_initializer=initializers.serialize(self.location_initializer),
            scale_trainable=self.scale_trainable,
            location_trainable=self.location_trainable
        )
        base_config = super(NormalLeaf, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
