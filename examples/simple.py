import tensorflow as tf
from tensorflow import keras

import examples.mnist.data
from libspn_keras.backprop_mode import BackpropMode
from libspn_keras.layers.dense_product import DenseProduct
from libspn_keras.layers.indicator_leaf import IndicatorLeaf
from libspn_keras.layers.root_sum import RootSum
from libspn_keras.layers.undecompose import Undecompose
from libspn_keras.models import build_ratspn
from libspn_keras.layers.decompose import Decompose
from libspn_keras.layers.dense_sum import DenseSum
from tensorflow import initializers
import numpy as np

tf.config.experimental_run_functions_eagerly(True)


def get_data():

    (x_train, y_train), (x_test, y_test) = examples.mnist.data.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)

    x_train = (x_train - np.mean(x_train, axis=1, keepdims=True)) / \
              (np.std(x_train, axis=1, keepdims=True) + 1e-4)
    x_test = (x_test - np.mean(x_test, axis=1, keepdims=True)) / \
              (np.std(x_test, axis=1, keepdims=True) + 1e-4)

    return x_train, y_train, x_test, y_test


def get_model(num_vars, logspace_accumulators, hard_em_backward, return_weighted_child_logits):

    init_sum0 = np.array([[[
        [0.6, 0.1, 0.1, 0.2],
        [0.1, 0.1, 0.6, 0.2]
    ]], [[
        [0.4, 0.3, 0.2, 0.1],
        [0.1, 0.2, 0.3, 0.4]
    ]]
    ]).transpose((0, 1, 3, 2))

    init_sum1 = np.array([0.25, 0.15, 0.35, 0.25])

    sum_product_stack = keras.models.Sequential([
        DenseProduct(
            num_factors=2
        ),
        DenseSum(
            num_sums=2, logspace_accumulators=False, backprop_mode=BackpropMode.HARD_EM,
            accumulator_initializer=initializers.Constant(init_sum0)
        ),
        DenseProduct(
            num_factors=2
        ),
        Undecompose(),
        RootSum(
            logspace_accumulators=False, backprop_mode=BackpropMode.HARD_EM,
            accumulator_initializer=initializers.Constant(init_sum1)
        ),
    ])

    # Use helper function to build the actual SPN
    return build_ratspn(
        num_vars=num_vars,
        decomposer=Decompose(num_decomps=1, permutations=[[0, 1, 2, 3]]),
        leaf=IndicatorLeaf(num_components=2),
        sum_product_stack=sum_product_stack
    ), sum_product_stack


def main():
    logspace_accumulators = False
    hard_em_backward = True
    return_weighted_child_logits = False

    num_vars = 4
    num_components = 2

    model, sum_product_stack = get_model(
        num_vars, logspace_accumulators, hard_em_backward, return_weighted_child_logits)

    # Train
    sum_weights0, sum_weights1 = sum_product_stack.trainable_variables
    var_assignments = np.arange(num_components ** num_vars)
    data = []
    for a in var_assignments:
        row = []
        for i in range(num_vars):
            row.append((a // (num_components ** i)) % num_components)

        data.append(row)

    print(tf.reduce_logsumexp(model(np.array(data))))

    with tf.GradientTape() as tape:
        log_likelihood = model(np.array(
            [[0, 0, 0, 0],
             [1, 0, 0, 0]],
        ))
        grad_sum0, grad_sum1 = tape.gradient(log_likelihood, [sum_weights0, sum_weights1])
        print(grad_sum0, grad_sum1)


if __name__ == "__main__":
    main()
