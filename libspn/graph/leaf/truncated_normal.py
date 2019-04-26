import tensorflow as tf
from tensorflow_probability.python import distributions as tfp

from libspn.graph.leaf.location_scale import LocationScaleLeaf
from libspn.utils.initializers import Equidistant
from libspn.utils.serialization import register_serializable


@register_serializable
class TruncatedNormalLeaf(LocationScaleLeaf):

    """A node representing uni-variate truncated normal distributions for continuous input
    variables. Each variable will have *k* components. Each component has its
    own location (mean) and scale (standard deviation). These parameters can be learned or fixed.

    Lack of evidence must be provided explicitly through
    feeding :py:attr:`~libspn.TruncatedNormalLeaf.evidence`. By default, evidence is set to ``True``
    for all variables.

    Args:
        feed (Tensor): Tensor feeding this node or ``None``. If ``None``,
                       an internal placeholder will be used to feed this node.
        num_vars (int): Number of random variables.
        num_components (int): Number of components per random variable.
        name (str): Name of the node
        loc_init (float or numpy.ndarray): If a float and there's no ``initialization_data``,
                                            all components are initialized with ``loc_init``. If
                                            an numpy.ndarray, must have shape
                                            ``[num_vars, num_components]``.
        scale_init (float): If a float and there's no ``initialization_data``, scales are
                            initialized with ``variance_init``.
        trainable_loc (bool): Whether to make the location ``Variable`` trainable.
        trainable_scale (bool): Whether to make the scale ``Variable`` trainable.
        truncate_min (float): Truncate minimum.
        truncate_max (float): Truncate maximum..
        evidence_indicator_feed (Tensor): Tensor feeding this node's evidence indicator. If
                                          ``None``, an internal placeholder with default value will
                                          be created.
    """

    def __init__(self, feed=None, num_vars=1, num_components=2, name="TruncatedNormalLeaf",
                 trainable_scale=True, trainable_loc=True,
                 loc_init=Equidistant(), scale_init=1.0,
                 min_scale=1e-2, evidence_indicator_feed=None, softplus_scale=False,
                 truncate_min=0.0, truncate_max=1.0, share_locs_across_vars=False,
                 share_scales=False):
        self._truncate_min = truncate_min
        self._truncate_max = truncate_max
        super().__init__(
            feed=feed, evidence_indicator_feed=evidence_indicator_feed,
            num_vars=num_vars, num_components=num_components, trainable_loc=trainable_loc,
            trainable_scale=trainable_scale, loc_init=loc_init, scale_init=scale_init,
            min_scale=min_scale, softplus_scale=softplus_scale, name=name, dimensionality=1,
            share_locs_across_vars=share_locs_across_vars, share_scales=share_scales)

    def _create_dist(self):
        if self._softplus_scale:
            return tfp.TruncatedNormal(
                self._loc_variable, tf.nn.softplus(self._scale_variable), low=self._truncate_min,
                high=self._truncate_max)
        return tfp.TruncatedNormal(
            self._loc_variable, self._scale_variable,
            low=self._truncate_min, high=self._truncate_max)