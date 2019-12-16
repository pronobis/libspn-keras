from tensorflow import keras

from libspn_keras.layers.bernoulli_condition import BernoulliCondition
from libspn_keras.layers.dense_product import DenseProduct
from libspn_keras.layers.base_leaf import BaseLeaf
from libspn_keras.layers.decompose import Decompose
import numpy as np
import tensorflow as tf

from libspn_keras.layers.indicator_leaf import IndicatorLeaf


def build_ratspn(
    sum_product_stack: keras.models.Sequential,
    leaf: BaseLeaf,
    decomposer: Decompose,
    num_vars: int,
    evidence_mask=False,
    bounded_marginalization=False
):

    factors = []

    for layer in sum_product_stack.layers:
        if isinstance(layer, DenseProduct):
            factors.append(layer.num_factors)

    decomposer.generate_permutations(factors, num_vars_spn_input=num_vars)

    inputs = []

    data_input = keras.layers.Input(
        shape=(num_vars,), name='spn_data_input',
        dtype=tf.int32 if isinstance(leaf, IndicatorLeaf) else tf.float32
    )
    decomposed = decomposer(data_input)
    leaf_prob = leaf(decomposed)

    inputs.append(data_input)

    if bounded_marginalization:
        bounded_marginalization_mask = keras.layers.Input(
            shape=(num_vars,), name='bounded_marginalization_mask_input')
        bounded_marginalization_mask_decomposed = decomposer(bounded_marginalization_mask)
        leaf_bounded_marginalization = leaf.__class__(
            num_components=leaf.num_components, use_cdf=True)
        leaf_prob_bounded_marginalized = leaf_bounded_marginalization(data_input)
        leaf_prob = tf.where(
            bounded_marginalization_mask_decomposed, leaf_prob_bounded_marginalized, leaf_prob)
        inputs.append(bounded_marginalization_mask)

    if evidence_mask:
        evidence_mask_input = keras.layers.Input(shape=(num_vars,), name='spn_evidence_mask_input')
        evidence_mask_decomposed = decomposer(evidence_mask_input)
        leaf_prob = keras.layers.Multiply()([leaf_prob, evidence_mask_decomposed])
        inputs.append(evidence_mask_input)

    root_prob = sum_product_stack(leaf_prob)

    return keras.models.Model(inputs=inputs, outputs=[root_prob])


def build_dgcspn(
    sum_product_stack: keras.models.Sequential,
    leaf: BaseLeaf,
    input_shape,
    evidence_mask=False,
    bounded_marginalization_rate=None,
    input_dropout_rate=None,
):
    inputs = []

    data_input = keras.layers.Input(
        shape=input_shape, name='spn_data_input',
        dtype=tf.int32 if isinstance(leaf, IndicatorLeaf) else tf.float32
    )
    leaf_prob = leaf(data_input)

    inputs.append(data_input)

    if bounded_marginalization_rate is not None:
        leaf_bounded_marginalization = leaf.__class__(
            num_components=leaf.num_components, use_cdf=True)
        leaf_bounded_marginalization_prob = leaf_bounded_marginalization(data_input)
        leaf_bounded_marginalization.set_weights(leaf.get_weights())
        leaf_prob = BernoulliCondition(rate=bounded_marginalization_rate)(
            [leaf_prob, leaf_bounded_marginalization_prob])

    if input_dropout_rate is not None:
        leaf_prob = BernoulliCondition(rate=input_dropout_rate)(
            [leaf_prob, tf.zeros_like(leaf_prob)])

    if evidence_mask:
        evidence_mask_input = keras.layers.Input(
            shape=input_shape, name='spn_evidence_mask_input')
        leaf_prob = keras.layers.Multiply()([leaf_prob, evidence_mask_input])
        inputs.append(evidence_mask_input)

    root_prob = sum_product_stack(leaf_prob)

    return keras.models.Model(inputs=inputs, outputs=[root_prob])
