import abc
import typing

from tensorflow import keras
from libspn_keras.layers.bernoulli_condition import BernoulliCondition
from libspn_keras.layers.dense_product import DenseProduct
from libspn_keras.layers.base_leaf import BaseLeaf
from libspn_keras.layers.z_score_normalization import ZScoreNormalization
import tensorflow as tf


class SumProductNetworkBase(keras.models.Model, abc.ABC):

    def __init__(
        self, leaf: BaseLeaf, sum_product_stack: typing.List[keras.layers.Layer],
        with_evidence_mask=False, input_dropout_rate=None, cdf_rate=None,
        completion_by_posterior_marginal=False, normalization_axes=None,
        normalization_epsilon=1e-10, with_evidence_mask_for_normalization=False, **kwargs
    ):
        """
        Sum Product Network base class. Requires inheriting classes to implement call(...).

        Args:
            leaf: Leaf distribution
            sum_product_stack: List of sum and product layers, typically in an alternating fashion.
            with_evidence_mask: Whether to expect a evidence mask a second input to the SPN.
            input_dropout_rate: Input dropout rate. If not None, will apply dropout by 
                randomly marginalizing out input variables during training.
            cdf_rate: The rate at which to randomly use the cumulative distribution function (cdf) 
                instead of the pdf at the leaf node. 
            completion_by_posterior_marginal: Whether to perform completion by posterior marginals
                as proposed in [Poon and Domingos (2011)] and first described in [Darwiche (2003)].
            normalization_axes: Normalization axes. Currently only supports 
                NormalizationAxes.PER_SAMPLE
            normalization_epsilon: Small positive constant for stabilizing normalization
            with_evidence_mask_for_normalization: Whether to use evidence mask for normalization. If
                True, the mean and stddev that are computed for sample-wise normalization will
                exclude the inputs that are not part of the evidence. Although this is the only
                'fair' way of doing it, setting it to False corresponds to what can be found in
                [Poon and Domingos (2011)].
            **kwargs: Passed on to keras.Model super class.
        """
        super(SumProductNetworkBase, self).__init__(**kwargs)

        # Set layers
        self.leaf = leaf
        if cdf_rate is not None:
            self.leaf_cdf = leaf.__class__(num_components=self.leaf.num_components, use_cdf=True)
            self.bernoulli_cond_cdf = BernoulliCondition(rate=cdf_rate, name="cdf_gate")
        if input_dropout_rate is not None:
            self.bernoulli_cond_input_dropout = BernoulliCondition(
                rate=1 - input_dropout_rate, name="input_dropout")
        if normalization_axes is not None:
            self.normalize = ZScoreNormalization(
                with_evidence_mask=with_evidence_mask, normalization_epsilon=normalization_epsilon)
        self.completion_by_posterior_marginal = completion_by_posterior_marginal
        self.sum_product_stack = sum_product_stack
        self.evidence_mask = with_evidence_mask
        self.cdf_rate = cdf_rate
        self.input_dropout_rate = input_dropout_rate
        self.normalization_axes = normalization_axes
        self.normalization_epsilon = normalization_epsilon
        self.use_evidence_mask_for_normalization = with_evidence_mask_for_normalization

    def _maybe_apply_evidence_mask(self, x, evidence_mask):
        if evidence_mask is None:
            return x
        return tf.where(evidence_mask, x, tf.zeros_like(x))

    def _maybe_normalize_input(self, data_input, evidence_mask=None):
        if self.normalization_axes is None:
            return data_input, None, None
        else:
            if evidence_mask is not None:
                evidence_mask = evidence_mask if self.use_evidence_mask_for_normalization \
                    else tf.ones_like(evidence_mask)
                return self.normalize([data_input, evidence_mask])
            else:
                return self.normalize(data_input)

    def _parse_inputs(self, inputs):
        require_evidence_mask = self.completion_by_posterior_marginal or self.evidence_mask
        evidence_mask_input = None
        if isinstance(inputs, list):
            if require_evidence_mask and len(inputs) != 2:
                raise ValueError("Second input must be evidence mask")
            elif require_evidence_mask:
                data_input, evidence_mask_input = inputs
                evidence_mask_input = tf.cast(evidence_mask_input, tf.bool)
            elif len(inputs) != 1:
                raise ValueError(
                    "More than 1 input, while no evidence mask is required for the graph")
            else:
                data_input = inputs[0]
        else:
            data_input = inputs
        return data_input, evidence_mask_input

    def _apply_stack(self, leaf_out):
        sum_product_stack_out = leaf_out
        for layer in self.sum_product_stack:
            sum_product_stack_out = layer(sum_product_stack_out)
        return sum_product_stack_out

    def _maybe_apply_input_dropout(self, leaf_out):
        if self.input_dropout_rate is not None:
            noise_shape = tf.concat([tf.shape(leaf_out)[:-1], [1]], axis=0)
            leaf_out = self.bernoulli_cond_input_dropout([leaf_out, tf.zeros(noise_shape)])
        return leaf_out

    def _maybe_apply_cdf(self, leaf_out, normalized_input):
        if self.cdf_rate is not None:
            leaf_cdf_out = self.leaf_cdf(normalized_input)
            leaf_out = self.bernoulli_cond_cdf([leaf_cdf_out, leaf_out])
        return leaf_out


class DenseSumProductNetwork(SumProductNetworkBase):
    """
    Dense Sum-Product Network. Currently only supports random scope decompositions. Assumes the
    input is a raw data tensor with shape [batch, num_variables].

    Args:
            leaf: Leaf distribution
            sum_product_stack: List of sum and product layers, typically in an alternating fashion.
            with_evidence_mask: Whether to expect a evidence mask a second input to the SPN.
            input_dropout_rate: Input dropout rate. If not None, will apply dropout by
                randomly marginalizing out input variables during training.
            cdf_rate: The rate at which to randomly use the cumulative distribution function (cdf)
                instead of the pdf at the leaf node.
            completion_by_posterior_marginal: Whether to perform completion by posterior marginals
                as proposed in [Poon and Domingos (2011)] and first described in [Darwiche (2003)].
            normalization_axes: Normalization axes. Currently only supports
                NormalizationAxes.PER_SAMPLE
            normalization_epsilon: Small positive constant for stabilizing normalization
            with_evidence_mask_for_normalization: Whether to use evidence mask for normalization. If
                True, the mean and stddev that are computed for sample-wise normalization will
                exclude the inputs that are not part of the evidence. Although this is the only
                'fair' way of doing it, setting it to False corresponds to what can be found in
                [Poon and Domingos (2011)].
            **kwargs: Passed on to keras.Model super class.

    """

    def call(self, inputs):

        data_input, _ = self._parse_inputs(inputs)

        normalized_input, mean, stddev = self._maybe_normalize_input(data_input)

        num_vars = data_input.shape.as_list()[1]
        self.decomposer.generate_permutations(
            self._gather_product_factors(self.sum_product_stack),
            num_vars_spn_input=num_vars
        )

        data_decomposed = self.decomposer(normalized_input)
        leaf_out = self.leaf(data_decomposed)
        leaf_out = self._maybe_apply_cdf(leaf_out, normalized_input)
        leaf_out = self._maybe_apply_input_dropout(leaf_out)
        sum_product_stack_out = self._apply_stack(leaf_out)

        if self.completion_by_posterior_marginal:
            raise NotImplementedError(
                "Completion by posterior marginal not yet implemented for DenseSumProductNetwork")

        return sum_product_stack_out

    @staticmethod
    def _gather_product_factors(sum_product_stack):

        factors = []

        for layer in sum_product_stack:
            if isinstance(layer, DenseProduct):
                factors.append(layer.num_factors)

        return factors


class SpatialSumProductNetwork(SumProductNetworkBase):
    """
    Spatial Sum-Product Network which assumes the input tensor is a raw data tensor with dimensions
    [batch, rows, cols, channels].

    Args:
            leaf: Leaf distribution
            sum_product_stack: List of sum and product layers, typically in an alternating fashion.
            with_evidence_mask: Whether to expect a evidence mask a second input to the SPN.
            input_dropout_rate: Input dropout rate. If not None, will apply dropout by
                randomly marginalizing out input variables during training.
            cdf_rate: The rate at which to randomly use the cumulative distribution function (cdf)
                instead of the pdf at the leaf node.
            completion_by_posterior_marginal: Whether to perform completion by posterior marginals
                as proposed in [Poon and Domingos (2011)] and first described in [Darwiche (2003)].
            normalization_axes: Normalization axes. Currently only supports
                NormalizationAxes.PER_SAMPLE
            normalization_epsilon: Small positive constant for stabilizing normalization
            with_evidence_mask_for_normalization: Whether to use evidence mask for normalization. If
                True, the mean and stddev that are computed for sample-wise normalization will
                exclude the inputs that are not part of the evidence. Although this is the only
                'fair' way of doing it, setting it to False corresponds to what can be found in
                [Poon and Domingos (2011)].
            **kwargs: Passed on to keras.Model super class.

    """

    def call(self, inputs):
        data_input, evidence_mask_input = self._parse_inputs(inputs)

        normalized_input, mean, stddev = self._maybe_normalize_input(
            data_input, evidence_mask_input)

        leaf_out = self.leaf(normalized_input)
        leaf_out = self._maybe_apply_cdf(leaf_out, normalized_input)
        leaf_out = self._maybe_apply_input_dropout(leaf_out)
        leaf_out = self._maybe_apply_evidence_mask(leaf_out, evidence_mask_input)

        if self.completion_by_posterior_marginal:
            with tf.GradientTape() as g:
                g.watch(leaf_out)
                sum_product_stack_out = self._apply_stack(leaf_out)
            dlog_root_dlog_leaf = g.gradient(sum_product_stack_out, leaf_out)
            leaf_modes = self.leaf.get_modes()
            dlog_root_dlog_leaf = tf.expand_dims(dlog_root_dlog_leaf, axis=-1)
            completion_nominator = tf.reduce_sum(
                leaf_modes * dlog_root_dlog_leaf, axis=-2)
            completion_denominator = tf.reduce_sum(
                dlog_root_dlog_leaf, axis=-2)
            completion_out = completion_nominator / completion_denominator
            completion_out = completion_out * (stddev + self.normalization_epsilon) + mean
            completion_out = tf.where(evidence_mask_input, data_input, completion_out)

            multiples = tf.cast(tf.concat(
                [tf.ones([tf.rank(data_input) - 1]), [tf.shape(data_input)[-1]]], axis=0), tf.int32)
            evidence_mask_input = tf.tile(evidence_mask_input, multiples=multiples)
            completion_out_completed_only = \
                tf.boolean_mask(completion_out, tf.logical_not(evidence_mask_input))
            input_completed_only = \
                tf.boolean_mask(data_input, tf.logical_not(evidence_mask_input))
            self.add_metric(tf.math.squared_difference(
                completion_out_completed_only, input_completed_only),
                name='completion_mse', aggregation='mean'
            )
            return completion_out
        else:
            sum_product_stack_out = self._apply_stack(leaf_out)

        return sum_product_stack_out
