import typing

from tensorflow import keras

from libspn_keras.layers.bernoulli_condition import BernoulliCondition
from libspn_keras.layers.dense_product import DenseProduct
from libspn_keras.layers.base_leaf import BaseLeaf
from libspn_keras.layers.decompose import Decompose
import tensorflow as tf

from libspn_keras.layers.dense_sum import DenseSum
from libspn_keras.layers.indicator_leaf import IndicatorLeaf
from libspn_keras.normalizationaxes import NormalizationAxes


class DenseSumProductNetwork(keras.models.Model):

    def __init__(
            self, decomposer: Decompose, leaf: BaseLeaf,
            sum_product_stack: typing.List[typing.Union[DenseProduct, DenseSum]],
            evidence_mask=False, input_dropout_rate=None, cdf_rate=False, completion_by_posterior_marginal=False,
            **kwargs
    ):
        super(DenseSumProductNetwork, self).__init__(**kwargs)

        # Set layers
        self.leaf = leaf
        if cdf_rate is not None:
            self.leaf_cdf = leaf.__class__(num_components=self.leaf.num_components, use_cdf=True)
            self.bernoulli_cond_cdf = BernoulliCondition(rate=cdf_rate, name="cdf_gate")
        if input_dropout_rate is not None:
            self.bernoulli_cond_input_dropout = BernoulliCondition(rate=input_dropout_rate, name="input_dropout")
        self.completion_by_posterior_marginal = completion_by_posterior_marginal
        if completion_by_posterior_marginal:
            raise NotImplementedError("Completion by posterior marginal not yet implemented for DenseSumProductNetwork")
        self.decomposer = decomposer
        self.sum_product_stack = sum_product_stack
        self.evidence_mask = evidence_mask
        self.cdf_rate = cdf_rate
        self.input_dropout_rate = input_dropout_rate

    def call(self, inputs):
        require_evidence_mask = self.completion_by_posterior_marginal or self.evidence_mask
        if isinstance(inputs, list):
            if require_evidence_mask and len(inputs) != 2:
                raise ValueError("Second input must be evidence mask")
            elif require_evidence_mask:
                data_input, evidence_mask_input = inputs
        else:
            data_input = inputs

        # Gather product factors for decomposer
        num_vars = data_input.shape.as_list()[1]
        self.decomposer.generate_permutations(
            self._gather_product_factors(self.sum_product_stack),
            num_vars_spn_input=num_vars
        )

        data_decomposed = self.decomposer(data_input)
        leaf_out = self.leaf(data_decomposed)

        if self.cdf_rate is not None:
            leaf_cdf_out = self.leaf_cdf(data_decomposed)
            leaf_out = self.bernoulli_cond_cdf([leaf_cdf_out, leaf_out])

        if self.input_dropout_rate is not None:
            leaf_out = self.bernoulli_cond_input_dropout([tf.zeros_like(leaf_out), leaf_out])

        sum_product_stack_out = leaf_out
        for layer in self.sum_product_stack:
            sum_product_stack_out = layer(sum_product_stack_out)

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


class SpatialSumProductNetwork(keras.models.Model):

    def __init__(
        self, leaf: BaseLeaf, sum_product_stack: typing.List[keras.layers.Layer],
        evidence_mask=False, input_dropout_rate=None, cdf_rate=False,
        completion_by_posterior_marginal=False, normalization_axes=None,
        normalization_epsilon=1e-4, **kwargs
    ):
        super(SpatialSumProductNetwork, self).__init__(**kwargs)

        # Set layers
        self.leaf = leaf
        if cdf_rate is not None:
            self.leaf_cdf = leaf.__class__(num_components=self.leaf.num_components, use_cdf=True)
            self.bernoulli_cond_cdf = BernoulliCondition(rate=cdf_rate, name="cdf_gate")
        if input_dropout_rate is not None:
            self.bernoulli_cond_input_dropout = BernoulliCondition(
                rate=1 - input_dropout_rate, name="input_dropout")
        self.completion_by_posterior_marginal = completion_by_posterior_marginal
        self.sum_product_stack = sum_product_stack
        self.evidence_mask = evidence_mask
        self.cdf_rate = cdf_rate
        self.input_dropout_rate = input_dropout_rate
        self.normalization_axes = normalization_axes
        self.normalization_epsilon = normalization_epsilon

    def call(self, inputs):
        require_evidence_mask = self.completion_by_posterior_marginal or self.evidence_mask
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

        mean = stddev = None
        if self.normalization_axes == NormalizationAxes.PER_SAMPLE:
            normalization_axes_indices = [1, 2, 3]
            mean = tf.reduce_mean(data_input, axis=normalization_axes_indices, keepdims=True)
            stddev = tf.math.reduce_std(data_input, axis=normalization_axes_indices, keepdims=True)
            normalized_input = (data_input - mean) / (stddev + self.normalization_epsilon)
        elif self.normalization_axes is None:
            normalized_input = data_input
        else:
            raise ValueError("Normalization axes other than PER_SAMPLE not supported")

        leaf_out = self.leaf(normalized_input)

        if self.cdf_rate is not None:
            leaf_cdf_out = self.leaf_cdf(normalized_input)
            leaf_out = self.bernoulli_cond_cdf([leaf_cdf_out, leaf_out])

        if self.input_dropout_rate is not None:
            noise_shape = tf.concat([tf.shape(leaf_out)[:-1], [1]], axis=0)
            leaf_out = self.bernoulli_cond_input_dropout([leaf_out, tf.zeros(noise_shape)])

        sum_product_stack_out = leaf_out
        for layer in self.sum_product_stack:
            sum_product_stack_out = layer(sum_product_stack_out)

        if self.completion_by_posterior_marginal:
            dlog_root_dlog_leaf = tf.gradients(sum_product_stack_out, leaf_out)[0]
            leaf_modes = self.leaf.get_modes()
            completion_nominator = tf.reduce_sum(
                leaf_modes * dlog_root_dlog_leaf, axis=-1, keepdims=True)
            completion_denominator = tf.reduce_sum(
                dlog_root_dlog_leaf, axis=-1, keepdims=True) + 1e-8
            completion_out = tf.where(
                evidence_mask_input, normalized_input, completion_nominator / completion_denominator)

            completion_out = completion_out * (stddev + self.normalization_epsilon) + mean

            completion_out_completed_only = \
                tf.boolean_mask(completion_out, tf.logical_not(evidence_mask_input))
            input_completed_only = \
                tf.boolean_mask(data_input, tf.logical_not(evidence_mask_input))
            self.add_metric(tf.math.squared_difference(
                completion_out_completed_only, input_completed_only),
                name='completion_mse', aggregation='mean'
            )
            return completion_out

        return sum_product_stack_out
