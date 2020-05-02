import numpy as np
from tensorflow_core import initializers

import libspn_keras as spnk
from libspn_keras import BackpropMode
from scipy import stats
import tensorflow as tf
from tensorflow import keras


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


def get_discrete_data():
    var_assignments = np.arange(NUM_COMPONENTS ** NUM_VARS)
    data = []
    for a in var_assignments:
        row = []
        for i in range(NUM_VARS):
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


def get_continuous_model():
    spn = keras.models.Sequential([
        spnk.layers.FlatToRegions(num_decomps=1, input_shape=(NUM_VARS,)),
        spnk.layers.NormalLeaf(
            num_components=NUM_COMPONENTS,
            location_initializer=keras.initializers.Constant(value=NORMAL_COMPONENTS_LOCATIONS)
        ),
        spnk.layers.PermuteAndPadScopes([[0, 1, 2, 3]]),
        spnk.layers.DenseProduct(
            num_factors=2
        ),
        spnk.layers.DenseSum(
            num_sums=2, logspace_accumulators=False, backprop_mode=BackpropMode.HARD_EM,
            accumulator_initializer=initializers.Constant(FIRST_SUM_WEIGHTS)
        ),
        spnk.layers.DenseProduct(
            num_factors=2
        ),
        spnk.layers.RootSum(
            logspace_accumulators=False, backprop_mode=BackpropMode.HARD_EM,
            accumulator_initializer=initializers.Constant(SECOND_SUM_WEIGHTS),
            return_weighted_child_logits=False
        ),
    ])
    spn.summary()
    return spn


def get_discrete_model():
    spn = keras.models.Sequential([
        spnk.layers.FlatToRegions(num_decomps=1, input_shape=(NUM_VARS,), dtype=tf.int32),
        spnk.layers.IndicatorLeaf(num_components=NUM_COMPONENTS),
        spnk.layers.PermuteAndPadScopes([[0, 1, 2, 3]]),
        spnk.layers.DenseProduct(
            num_factors=2
        ),
        spnk.layers.DenseSum(
            num_sums=2, logspace_accumulators=False, backprop_mode=BackpropMode.HARD_EM,
            accumulator_initializer=initializers.Constant(FIRST_SUM_WEIGHTS)
        ),
        spnk.layers.DenseProduct(
            num_factors=2
        ),
        spnk.layers.RootSum(
            logspace_accumulators=False, backprop_mode=BackpropMode.HARD_EM,
            accumulator_initializer=initializers.Constant(SECOND_SUM_WEIGHTS),
            return_weighted_child_logits=False
        ),
    ])
    spn.summary()
    return spn
