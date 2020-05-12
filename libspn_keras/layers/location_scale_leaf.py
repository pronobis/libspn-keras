import abc

from libspn_keras.layers.base_leaf import BaseLeaf
import tensorflow_probability as tfp
from tensorflow import initializers
import tensorflow as tf

from libspn_keras.math.soft_em_grads import LocationScaleEMGradWrapper, LocationEMGradWrapper


class LocationScaleLeafBase(BaseLeaf):
    """
    Computes the log probability of multiple components per variable along the final axis.

    Args:
        num_components: Number of components per variable
        location_initializer: Initializer for location variable
        location_trainable: Boolean that indicates whether location is trainable
        scale_initializer: Initializer for scale variable
        **kwargs: kwargs to pass on to the keras.Layer super class
    """
    def __init__(
        self, num_components, location_initializer=None,
        location_trainable=True, scale_initializer=None, scale_trainable=None,
        accumulator_initializer=None, use_accumulators=False, **kwargs
    ):
        super(LocationScaleLeafBase, self).__init__(num_components=num_components, **kwargs)
        self.location_initializer = location_initializer or initializers.TruncatedNormal(stddev=1.0)
        self.location_trainable = location_trainable
        self.scale_initializer = scale_initializer or initializers.Ones()
        self.scale_trainable = scale_trainable
        self.accumulator_initializer = accumulator_initializer or initializers.Ones()
        self.use_accumulators = use_accumulators
        self._num_scopes = self._num_decomps = None

    def _build_distribution(self, shape):
        if self.use_accumulators:
            self._create_loc_scale_accumulators(shape)
            self._get_distribution = self._get_distribution_from_accumulators
        else:
            self._create_loc_scale_vars(shape)
            self._get_distribution = self._get_distribution_from_vars

    def _get_distribution_from_vars(self):
        return self._build_distribution_from_loc_and_scale(loc=self.loc, scale=self.scale)

    def _get_distribution_from_accumulators(self):
        loc = self.first_order_moment_num_accum / self.first_order_moment_denom_accum
        if self.scale_trainable:
            scale = tf.sqrt(self.second_order_moment_num_accum / self.second_order_moment_denom_accum - tf.square(loc))
            dist = self._build_distribution_from_loc_and_scale(loc=loc, scale=scale)
            return LocationScaleEMGradWrapper(
                dist, self.first_order_moment_denom_accum, self.first_order_moment_num_accum,
                self.second_order_moment_denom_accum, self.second_order_moment_num_accum)
        else:
            dist = self._build_distribution_from_loc_and_scale(loc=loc, scale=self.scale)
            return LocationEMGradWrapper(dist, self.first_order_moment_denom_accum, self.first_order_moment_num_accum)

    def _create_loc_scale_accumulators(self, shape):
        self.first_order_moment_denom_accum = self.add_weight(
            name="first_order_moment_denom_accum", shape=shape, initializer=self.accumulator_initializer,
            trainable=self.location_trainable)
        self.first_order_moment_num_accum = self.add_weight(
            name="first_order_moment_num_accum", shape=shape, initializer=self.location_initializer,
            trainable=self.location_trainable)
        self.first_order_moment_num_accum.assign(
            self.first_order_moment_num_accum * self.first_order_moment_denom_accum)

        if self.scale_trainable:
            self.second_order_moment_denom_accum = self.add_weight(
                name="second_order_moment_denom_accum", shape=shape, initializer=self.accumulator_initializer,
                trainable=True)
            self.second_order_moment_num_accum = self.add_weight(
                name="second_order_moment_num_accum", shape=shape, initializer=self.scale_initializer,
                trainable=True)
            loc = self.first_order_moment_num_accum / self.first_order_moment_denom_accum
            self.second_order_moment_num_accum.assign(
                (tf.square(self.second_order_moment_num_accum) + tf.square(loc)) * self.second_order_moment_denom_accum
            )
        else:
            self.scale = self.add_weight(
                name="scale", shape=shape, initializer=self.scale_initializer, trainable=False
            )

    @abc.abstractmethod
    def _build_distribution_from_loc_and_scale(self, loc, scale):
        """ Implement in descendant classes"""

    def _create_loc_scale_vars(self, shape):
        self.loc = self.add_weight(
            name="location", shape=shape, initializer=self.location_initializer,
            trainable=self.location_trainable)
        self.scale = self.add_weight(
            name="scale", shape=shape, initializer=self.scale_initializer, trainable=False)

    def get_config(self):
        config = dict(
            scale_initializer=initializers.serialize(self.scale_initializer),
            location_initializer=initializers.serialize(self.location_initializer),
            accumulator_initializer=initializers.serialize(self.accumulator_initializer),
            scale_trainable=self.scale_trainable,
            use_accumulators=self.use_accumulators,
            location_trainable=self.location_trainable,
        )
        base_config = super(LocationScaleLeafBase, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def get_modes(self):
        return self._get_distribution().mode()


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
        **kwargs: kwargs to pass on to the keras.Layer super class
    """
    def __init__(self, num_components, location_initializer=None, location_trainable=True, scale_initializer=None,
                 accumulator_initializer=None, use_accumulators=False, **kwargs):
        super().__init__(
            num_components=num_components,
            location_initializer=location_initializer,
            location_trainable=location_trainable,
            scale_initializer=scale_initializer,
            scale_trainable=False,
            accumulator_initializer=accumulator_initializer,
            use_accumulators=use_accumulators,
            **kwargs)

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
        **kwargs: kwargs to pass on to the keras.Layer super class
    """
    def __init__(self, num_components, location_initializer=None, location_trainable=True, scale_initializer=None,
                 accumulator_initializer=None, use_accumulators=False, **kwargs):
        super().__init__(
            num_components=num_components,
            location_initializer=location_initializer,
            location_trainable=location_trainable,
            scale_initializer=scale_initializer,
            scale_trainable=False,
            accumulator_initializer=accumulator_initializer,
            use_accumulators=use_accumulators,
            **kwargs)

    def _build_distribution_from_loc_and_scale(self, loc, scale):
        return tfp.distributions.Laplace(loc=loc, scale=scale)

