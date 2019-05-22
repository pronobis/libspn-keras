import tensorflow as tf
from tensorflow_probability import distributions as tfd
from libspn import utils
from libspn.graph.node import Node
from libspn.graph.leaf.location_scale import LocationScaleLeaf
from libspn.utils.serialization import register_serializable


@register_serializable
class MultivariateNormalDiagLeaf(LocationScaleLeaf):
    """A node representing multiple multi-variate Normal distributions for continuous input
    variables. Each variable will have ``num_components`` Normal components. Each Normal 
    component has its own location (mean) and scale (standard deviation). These parameters can be 
    learned or fixed.
    
    Lack of evidence must be provided explicitly through
    feeding :py:attr:`~libspn.MultivariateNormalDiagLeaf.evidence`. By default, evidence is set 
    to ``True`` for all variables.

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
        min_scale (float): Minimum value for scale. Used for avoiding numerical instabilities.
        evidence_indicator_feed (Tensor): Tensor feeding this node's evidence indicator. If
                                          ``None``, an internal placeholder with default value will
                                          be created.
    """

    def __init__(self, feed=None, num_vars=1, num_components=2, dimensionality=2,
                 name="MultivariateNormalDiagLeaf", total_counts_init=1.0,
                 trainable_scale=True, trainable_loc=True,
                 loc_init=tf.initializers.random_uniform(0.0, 1.0),
                 scale_init=1.0, min_scale=1e-2, evidence_indicator_feed=None,
                 softplus_scale=False, share_locs_across_vars=False, share_scales=False,
                 samplewise_normalization=False):
        super().__init__(
            feed=feed, name=name, dimensionality=dimensionality,
            num_components=num_components, num_vars=num_vars,
            evidence_indicator_feed=evidence_indicator_feed, component_axis=-2,
            total_counts_init=total_counts_init, loc_init=loc_init, trainable_loc=trainable_loc,
            trainable_scale=trainable_scale, scale_init=scale_init, min_scale=min_scale,
            softplus_scale=softplus_scale, share_locs_across_vars=share_locs_across_vars,
            share_scales=share_scales, samplewise_normalization=samplewise_normalization)

    def _create_dist(self):
        if self._softplus_scale:
            return tfd.MultivariateNormalDiag(self._loc_variable, self._scale_variable)
        return tfd.MultivariateNormalDiagWithSoftplusScale(self._loc_variable, self._scale_variable)

    @utils.docinherit(Node)
    @utils.lru_cache
    def _compute_hard_em_update(self, counts):
        counts_reshaped = tf.reshape(counts, (-1, self._num_vars, self._num_components, 1))
        # Determine accumulates per component
        accum = tf.reduce_sum(counts_reshaped, axis=0)

        # Tile the feed
        # tiled_feed = self._tile_num_components(self._feed)
        tiled_feed = self._preprocessed_feed()
        # Accumulate data (broadcast along dim axis)
        data_per_component = tf.multiply(counts_reshaped, tiled_feed, name="DataPerComponent")
        squared_data_per_component = tf.multiply(
            counts_reshaped, tf.square(tiled_feed), name="SquaredDataPerComponent")
        sum_data = tf.reduce_sum(data_per_component, axis=0)
        sum_data_squared = tf.reduce_sum(squared_data_per_component, axis=0)
        return {'accum': accum, "sum_data": sum_data, "sum_data_squared": sum_data_squared}

    def assign(self, accum, sum_data, sum_data_squared):
        """
        Assigns new values to variables based on accumulated tensors. It updates the distribution
        parameters based on what can be found in "Online Structure Learning for Sum-Product Networks
        with Gaussian Leaves" by Hsu et al. (2017) https://arxiv.org/pdf/1701.05265.pdf

        Args:
            accum (Tensor): A ``Variable`` with accumulated counts per component.
            sum_data (Tensor): A ``Variable`` with the accumulated sum of data per component.
            sum_data_squared (Tensor): A ``Variable`` with the accumulated sum of squares of data
                                       per component.
        Returns:
            Tuple: A tuple containing assignment operations for the new total counts, the variance
            and the mean.
        """
        n = tf.expand_dims(
            tf.maximum(self._total_count_variable, tf.ones_like(self._total_count_variable)),
            axis=-1)
        k = tf.expand_dims(accum, axis=-1)
        mean = (n * self._loc_variable + sum_data) / tf.cast(n + k, tf.float32)

        current_var = tf.square(self.scale_variable) if not self._softplus_scale else \
            tf.square(tf.nn.softplus(self._scale_variable))
        variance = (n * current_var + sum_data_squared -
                    2 * self.loc_variable * sum_data + k * tf.square(self.loc_variable)) / \
                   (n + k) - tf.square(mean - self._loc_variable)

        scale = tf.sqrt(variance)
        if self._softplus_scale:
            scale = tfd.softplus_inverse(scale)
        scale = tf.maximum(scale, self._min_scale)
        with tf.control_dependencies([n, mean, scale]):
            return (
                tf.assign_add(self._total_count_variable, tf.squeeze(k, axis=-1)),
                tf.assign(self._scale_variable, scale) if self._trainable_scale else tf.no_op(),
                tf.assign(self._loc_variable, mean) if self._trainable_loc else tf.no_op())