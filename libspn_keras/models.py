from tensorflow import keras

from libspn_keras.layers.across_scope_outer_product import AcrossScopeOuterProduct
from libspn_keras.layers.base_leaf import BaseLeaf
from libspn_keras.layers.decompose import Decompose
import numpy as np
import tensorflow as tf

from libspn_keras.layers.indicator_leaf import IndicatorLeaf


def build_sum_product_network(
    sum_product_stack: keras.models.Sequential,
    leaf: BaseLeaf,
    decomposer: Decompose,
    num_vars: int,
    evidence_mask=False
):

    factors = []

    for layer in sum_product_stack.layers:
        if isinstance(layer, AcrossScopeOuterProduct):
            factors.append(layer.num_factors)

    decomposer.generate_permutations(factors, num_vars_spn_input=num_vars)

    inputs = []

    data_input = keras.layers.Input(
        shape=(num_vars,), name='spn_data_input', dtype=tf.int32 if isinstance(leaf, IndicatorLeaf) else tf.float32)
    decomposed = decomposer(data_input)
    leaf_prob = leaf(decomposed)

    inputs.append(data_input)

    if evidence_mask:
        evidence_mask_input = keras.layers.Input(shape=(num_vars,), name='spn_evidence_mask_input')
        evidence_mask_decomposed = decomposer(evidence_mask_input)
        leaf_prob = keras.layers.Multiply()([leaf_prob, evidence_mask_decomposed])
        inputs.append(evidence_mask_input)

    root_prob = sum_product_stack(leaf_prob)

    return keras.models.Model(inputs=inputs, outputs=[root_prob])
