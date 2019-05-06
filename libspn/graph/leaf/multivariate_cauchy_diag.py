import tensorflow as tf
from tensorflow.contrib.distributions.python.ops import distribution_util
from tensorflow_probability import distributions as tfp, bijectors
from libspn.graph.leaf.location_scale import LocationScaleLeaf
from libspn.utils.serialization import register_serializable


@register_serializable
class MultivariateCauchyDiagLeaf(LocationScaleLeaf):

    def __init__(self, feed=None, num_vars=1, num_components=2, dimensionality=2,
                 name="MultivariateCauchyDiagLeaf", total_counts_init=1,
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
        scale_diag = tf.nn.softplus(self._scale_variable) if self._softplus_scale \
            else self._scale_variable
        scale = distribution_util.make_diag_scale(
            loc=self._loc_variable,
            scale_diag=scale_diag,
            validate_args=False,
            assert_positive=False)
        batch_shape, event_shape = distribution_util.shapes_from_loc_and_scale(
            self._loc_variable, scale)

        return tfp.TransformedDistribution(
            distribution=tfp.Cauchy(
                loc=tf.zeros([], dtype=scale.dtype),
                scale=tf.ones([], dtype=scale.dtype)),
            bijector=bijectors.AffineLinearOperator(
                shift=self._loc_variable, scale=scale),
            batch_shape=batch_shape,
            event_shape=event_shape,
            name="MultivariateCauchyDiag" + ("SoftplusScale" if self._softplus_scale else ""))

    def mode(self):
        return self._loc_variable