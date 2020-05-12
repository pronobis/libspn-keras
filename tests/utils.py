import numpy as np
from tensorflow.keras import initializers

import libspn_keras as spnk
from libspn_keras import BackpropMode
from scipy import stats
import tensorflow as tf
from tensorflow import keras
from libspn_keras.models import SequentialSumProductNetwork

NUM_VARS = 4
NUM_COMPONENTS = 2
FIRST_SUM_WEIGHTS = np.array([[[
    [0.6, 0.1, 0.1, 0.2],
    [0.1, 0.1, 0.6, 0.2]
]], [[
    [0.4, 0.3, 0.2, 0.1],
    [0.1, 0.2, 0.3, 0.4]
]]
]).transpose((0, 1, 3, 2))
SECOND_SUM_WEIGHTS = np.array([0.25, 0.15, 0.35, 0.25])
NORMAL_COMPONENTS_LOCATIONS = np.array(
    [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]).reshape([1, 4, 1, 2, 1])
BATCH_SIZE = 16
NUM_STEPS = 3


def indicators(x):
    out = np.zeros((x.shape[0] * x.shape[1], NUM_COMPONENTS))
    out[np.arange(out.shape[0]), x.ravel()] = 1
    out = np.reshape(out, (x.shape[0], x.shape[1], 1, NUM_COMPONENTS))
    return out


def normal_leafs(x):
    dist = stats.norm(loc=NORMAL_COMPONENTS_LOCATIONS.squeeze(-1))
    out = dist.pdf(x.reshape((-1, NUM_VARS, 1, 1)))
    return out


def product0_out(x):
    x = np.reshape(x, (-1, 2, 2, 2))
    x_per_factor = np.split(x, indices_or_sections=2, axis=2)
    x_per_factor[0] = x_per_factor[0].transpose((0, 1, 3, 2))
    outer_product = x_per_factor[0] * x_per_factor[1]
    return np.reshape(outer_product, (x.shape[0], 2, 1, 4))


def product1_out(x):
    x = np.reshape(x, (-1, 1, 2, 2))
    x_per_factor = np.split(x, indices_or_sections=2, axis=2)
    x_per_factor[0] = x_per_factor[0].transpose((0, 1, 3, 2))
    outer_product = x_per_factor[0] * x_per_factor[1]
    return np.reshape(outer_product, (x.shape[0], 1, 1, 4))


def sum0_out(x):
    x = np.transpose(x, (1, 2, 0, 3))
    out = np.matmul(x, FIRST_SUM_WEIGHTS)
    return np.transpose(out, (2, 0, 1, 3))


def root_out(x):
    init_sum1 = np.array([0.25, 0.15, 0.35, 0.25])
    x = np.reshape(x, (-1, 4))
    out = np.matmul(x, np.expand_dims(init_sum1, 1))
    return out


def get_discrete_data(num_vars=None):
    num_vars = num_vars or NUM_VARS
    var_assignments = np.arange(NUM_COMPONENTS ** num_vars)
    data = []
    for a in var_assignments:
        row = []
        for i in range(num_vars):
            row.append((a // (NUM_COMPONENTS ** i)) % NUM_COMPONENTS)

        data.append(row)
    data = np.array(data, dtype=np.int32)
    return data


def get_continuous_data():
    component0 = np.random.normal(size=(BATCH_SIZE * NUM_VARS))
    component1 = np.random.normal(loc=1.0, size=(BATCH_SIZE * NUM_VARS))
    choice = np.random.random() > 0.5
    data = np.array(np.where(choice, component0, component1)).reshape(BATCH_SIZE, NUM_VARS)
    return data


def get_continuous_model(infer_no_evidence=False):
    backprop_mode = BackpropMode.GRADIENT if infer_no_evidence else BackpropMode.HARD_EM_UNWEIGHTED
    spn = SequentialSumProductNetwork([
        spnk.layers.FlatToRegions(num_decomps=1, input_shape=(NUM_VARS,)),
        spnk.layers.NormalLeaf(
            num_components=NUM_COMPONENTS,
            use_accumulators=True,
            scale_trainable=False, location_trainable=True,
            location_initializer=keras.initializers.Constant(value=NORMAL_COMPONENTS_LOCATIONS)
        ),
        spnk.layers.PermuteAndPadScopes([[0, 1, 2, 3]]),
        spnk.layers.DenseProduct(
            num_factors=2
        ),
        spnk.layers.DenseSum(
            num_sums=2, logspace_accumulators=False, backprop_mode=backprop_mode,
            accumulator_initializer=initializers.Constant(FIRST_SUM_WEIGHTS)
        ),
        spnk.layers.DenseProduct(
            num_factors=2
        ),
        spnk.layers.RootSum(
            logspace_accumulators=False, backprop_mode=backprop_mode,
            accumulator_initializer=initializers.Constant(SECOND_SUM_WEIGHTS),
            return_weighted_child_logits=False
        ),
    ], infer_no_evidence=infer_no_evidence)
    spn.summary()
    return spn


def get_dynamic_model():
    sum_kwargs = dict(logspace_accumulators=False, backprop_mode=BackpropMode.HARD_EM_UNWEIGHTED)
    template = keras.models.Sequential([
        spnk.layers.FlatToRegions(num_decomps=1, input_shape=(NUM_VARS,), dtype=tf.int32),
        spnk.layers.IndicatorLeaf(num_components=NUM_COMPONENTS),
        spnk.layers.PermuteAndPadScopes([[0, 1, 2, 3]]),
        spnk.layers.DenseProduct(
            num_factors=2
        ),
        spnk.layers.DenseSum(
            num_sums=2, accumulator_initializer=initializers.Constant(FIRST_SUM_WEIGHTS), **sum_kwargs
        ),
        spnk.layers.DenseProduct(
            num_factors=2
        ),
    ])
    top_net = keras.Sequential([
        spnk.layers.RootSum(
            accumulator_initializer=initializers.Constant(SECOND_SUM_WEIGHTS), input_shape=[1, 1, 4],
            return_weighted_child_logits=False, **sum_kwargs
        )
    ], name='top_net')
    interface_t_minus1 = keras.Sequential(
        [spnk.layers.DenseSum(num_sums=2, input_shape=[1, 1, 4], **sum_kwargs)], name='interface_t_minus_1')
    interface_t0 = keras.Sequential(
        [spnk.layers.DenseSum(num_sums=2, input_shape=[1, 1, 4], **sum_kwargs)], name='interface_t0')
    dynamic_spn = spnk.models.DynamicSumProductNetwork(
        template_network=template, interface_network_t0=interface_t0, interface_network_t_minus_1=interface_t_minus1,
        top_network=top_net)
    return dynamic_spn


def get_discrete_model():
    spn = SequentialSumProductNetwork([
        spnk.layers.FlatToRegions(num_decomps=1, input_shape=(NUM_VARS,), dtype=tf.int32),
        spnk.layers.IndicatorLeaf(num_components=NUM_COMPONENTS),
        spnk.layers.PermuteAndPadScopes([[0, 1, 2, 3]]),
        spnk.layers.DenseProduct(
            num_factors=2
        ),
        spnk.layers.DenseSum(
            num_sums=2, logspace_accumulators=False, backprop_mode=BackpropMode.HARD_EM_UNWEIGHTED,
            accumulator_initializer=initializers.Constant(FIRST_SUM_WEIGHTS)
        ),
        spnk.layers.DenseProduct(
            num_factors=2
        ),
        spnk.layers.RootSum(
            logspace_accumulators=False, backprop_mode=BackpropMode.HARD_EM_UNWEIGHTED,
            accumulator_initializer=initializers.Constant(SECOND_SUM_WEIGHTS),
            return_weighted_child_logits=False
        ),
    ])
    spn.summary()
    return spn
