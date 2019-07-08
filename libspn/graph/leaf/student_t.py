import tensorflow as tf
from tensorflow_probability import distributions as tfd
from libspn import utils, conf
from libspn.graph.node import Node
from libspn.graph.leaf.location_scale import LocationScaleLeaf
from libspn.utils.initializers import Equidistant
from libspn.utils.serialization import register_serializable


@register_serializable
class StudentTLeaf(LocationScaleLeaf):

    """A node representing uni-variate Student-T distributions for continuous input
    variables. Each variable will have *k* components. Each component has its
    own location (mean) and scale (standard deviation). These parameters can be learned or fixed.

    Lack of evidence must be provided explicitly through
    feeding :py:attr:`~libspn.StudentTLeaf.evidence`. By default, evidence is set to ``True``
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
        evidence_indicator_feed (Tensor): Tensor feeding this node's evidence indicator. If
                                          ``None``, an internal placeholder with default value will
                                          be created.
    """

    def __init__(self, feed=None, num_vars=1, num_components=2, name="StudentTLeaf",
                 trainable_scale=True, trainable_loc=True,
                 loc_init=Equidistant(), scale_init=1.0,
                 min_scale=1e-2, evidence_indicator_feed=None, softplus_scale=False,
                 trainable_df=False, df_init=tf.initializers.constant(1.0),
                 share_locs_across_vars=False, share_scales=False, share_dfs=False):
        self._trainable_df = trainable_df
        self._df_init = df_init
        self._share_dfs = share_dfs

        super().__init__(
            feed=feed, evidence_indicator_feed=evidence_indicator_feed,
            num_vars=num_vars, num_components=num_components, trainable_loc=trainable_loc,
            trainable_scale=trainable_scale, loc_init=loc_init, scale_init=scale_init,
            min_scale=min_scale, softplus_scale=softplus_scale, name=name, dimensionality=1,
            share_locs_across_vars=share_locs_across_vars, share_scales=share_scales)

    def _create_dist(self):
        if self._softplus_scale:
            return tfd.StudentTWithAbsDfSoftplusScale(
                self._df_variable, self._loc_variable, self._scale_variable)
        return tfd.StudentT(self._df_variable, self._loc_variable, self._scale_variable)

    @utils.docinherit(Node)
    def _create(self):
        super()._create()
        with tf.variable_scope(self._name):
            # Initialize locations
            shape = self._variable_shape(
                1 if self._share_dfs else self._num_vars,
                1 if self._share_dfs else self._num_components,
                self._dimensionality)
            shape_kwarg = dict(shape=shape) if callable(self._df_init) else dict()
            self._df_variable = tf.get_variable(
                "Df", initializer=self._df_init, dtype=conf.dtype,
                trainable=self._trainable_df, **shape_kwarg)

    @property
    def variables(self):
        """Returns mean and variance variables. """
        return self._df_variable, self._loc_variable, self._scale_variable