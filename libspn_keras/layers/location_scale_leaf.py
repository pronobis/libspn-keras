import abc

from libspn_keras.layers.base_leaf import BaseLeaf
import tensorflow_probability as tfp
from tensorflow import initializers


class LocationScaleLeafBase(BaseLeaf):

    def __init__(
        self, num_components, location_initializer=None,
        location_trainable=True, scale_initializer=None, scale_trainable=False, **kwargs
    ):
        """
        Computes the log probability of multiple components per variable along the final axis.

        Args:
            num_components: Number of components per variable
            location_initializer: Initializer for location variable
            location_trainable: Boolean that indicates whether location is trainable
            scale_initializer: Initializer for scale variable
            scale_trainable: Boolean that indicates whether scale is trainable
            **kwargs: kwargs to pass on to the keras.Layer super class
        """
        super(LocationScaleLeafBase, self).__init__(num_components=num_components, **kwargs)
        self.location_initializer = location_initializer or initializers.TruncatedNormal(stddev=1.0)
        self.location_trainable = location_trainable
        self.scale_initializer = scale_initializer or initializers.Ones()
        self.scale_trainable = scale_trainable
        self._distribution = self._num_scopes = self._num_decomps = None

    def _build_distribution(self, shape):
        loc, scale = self._add_location_and_scale(shape)
        return self._build_distribution_from_loc_and_scale(loc=loc, scale=scale)

    @abc.abstractmethod
    def _build_distribution_from_loc_and_scale(self, loc, scale):
        """ Implement in descendant classes"""

    def _add_location_and_scale(self, shape):
        loc = self.add_weight(
            name="location", shape=shape, initializer=self.location_initializer,
            trainable=self.location_trainable)
        scale = self.add_weight(
            name="scale", shape=shape, initializer=self.scale_initializer, trainable=False)
        return loc, scale

    def get_config(self):
        config = dict(
            scale_initializer=initializers.serialize(self.scale_initializer),
            location_initializer=initializers.serialize(self.location_initializer),
            scale_trainable=self.scale_trainable,
            location_trainable=self.location_trainable
        )
        base_config = super(LocationScaleLeafBase, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def get_modes(self):
        return self._distribution.mode()


class NormalLeaf(LocationScaleLeafBase):
    """
    Computes the log probability of multiple components per variable along the final axis. Each
    component is modelled as a univariate normal distribution.

    Args:
        num_components: Number of components per variable
        location_initializer: Initializer for location variable
        location_trainable: Boolean that indicates whether location is trainable
        scale_initializer: Initializer for scale variable
        scale_trainable: Boolean that indicates whether scale is trainable
        **kwargs: kwargs to pass on to the keras.Layer super class
    """

    def _build_distribution_from_loc_and_scale(self, loc, scale):
        return tfp.distributions.Normal(loc=loc, scale=scale)


class CauchyLeaf(LocationScaleLeafBase):
    """
    Computes the log probability of multiple components per variable along the final axis. Each
    component is modelled as a univariate Cauchy distribution.

    Args:
        num_components: Number of components per variable
        location_initializer: Initializer for location variable
        location_trainable: Boolean that indicates whether location is trainable
        scale_initializer: Initializer for scale variable
        scale_trainable: Boolean that indicates whether scale is trainable
        **kwargs: kwargs to pass on to the keras.Layer super class
    """

    def _build_distribution_from_loc_and_scale(self, loc, scale):
        return tfp.distributions.Cauchy(loc=loc, scale=scale)


class LaplaceLeaf(LocationScaleLeafBase):
    """
    Computes the log probability of multiple components per variable along the final axis. Each
    component is modelled as a univariate Laplace distribution.

    Args:
        num_components: Number of components per variable
        location_initializer: Initializer for location variable
        location_trainable: Boolean that indicates whether location is trainable
        scale_initializer: Initializer for scale variable
        scale_trainable: Boolean that indicates whether scale is trainable
        **kwargs: kwargs to pass on to the keras.Layer super class
    """

    def _build_distribution_from_loc_and_scale(self, loc, scale):
        return tfp.distributions.Laplace(loc=loc, scale=scale)

