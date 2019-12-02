import argparse

import tensorflow as tf
from tensorflow import keras

from libspn_keras.backprop_mode import BackpropMode
from libspn_keras.layers.across_scope_outer_product import AcrossScopeOuterProduct
from libspn_keras.layers.root_sum import RootSum
from libspn_keras.layers.undecompose import Undecompose
from libspn_keras.losses.negative_log_joint import NegativeLogJoint
from libspn_keras.losses.negative_log_marginal import NegativeLogMarginal
from libspn_keras.metrics.log_likelihood import LogMarginal
from libspn_keras.models import build_sum_product_network
from libspn_keras.layers.decompose import Decompose
from libspn_keras.layers.normal_leaf import NormalLeaf
from libspn_keras.layers.scope_wise_sum import ScopeWiseSum
import numpy as np

from libspn_keras.optimizers.online_expectation_maximization import OnlineExpectationMaximization


def get_data():

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)

    x_train = (x_train - np.mean(x_train, axis=1, keepdims=True)) / \
              (np.std(x_train, axis=1, keepdims=True) + 1e-4)
    x_test = (x_test - np.mean(x_test, axis=1, keepdims=True)) / \
              (np.std(x_test, axis=1, keepdims=True) + 1e-4)

    return x_train, y_train, x_test, y_test


def get_mnist_model(num_vars, logspace_accumulators, backprop_mode, return_weighted_child_logits):
    sum_product_stack = []

    weight_initializer = tf.initializers.TruncatedNormal(mean=0.5)

    # The 'backbone' stack of alternating sums and products
    for _ in range(int(np.floor(np.log2(num_vars)))):
        sum_product_stack.extend([
            AcrossScopeOuterProduct(
                num_factors=2
            ),
            ScopeWiseSum(
                num_sums=4,
                logspace_accumulators=logspace_accumulators,
                accumulator_initializer=weight_initializer,
                backprop_mode=backprop_mode
            ),
        ])

    # Add another layer for joining the last two scopes to the final one, followed by a class-wise root layer
    # which is then followed by undecomposing (combining decompositions) and finally followed
    # by a root sum. In this case we return
    sum_product_stack.extend([
        AcrossScopeOuterProduct(
            num_factors=2
        ),
        ScopeWiseSum(
            num_sums=1,
            logspace_accumulators=logspace_accumulators,
            accumulator_initializer=weight_initializer,
            backprop_mode=backprop_mode
        ),
        Undecompose(),
        RootSum(
            logspace_accumulators=logspace_accumulators,
            return_weighted_child_logits=return_weighted_child_logits,
            backprop_mode=backprop_mode
        )
    ])

    # Use helper function to build the actual SPN
    return build_sum_product_network(
        num_vars=num_vars,
        decomposer=Decompose(num_decomps=10),
        leaf=NormalLeaf(num_components=4),
        sum_product_stack=keras.models.Sequential(sum_product_stack),
    )


def main(args):
    if args.mode == "generative-hard-em":
        logspace_accumulators = False
        backprop_mode = BackpropMode.HARD_EM
        loss = NegativeLogMarginal(name="NegativeLogMarginal")
        metrics = [LogMarginal(name="LogMarginal")]
        optimizer = OnlineExpectationMaximization()
        return_weighted_child_logits = False
    elif args.mode == "generative-soft-em":
        logspace_accumulators = False
        backprop_mode = BackpropMode.EM
        loss = NegativeLogMarginal(name="NegativeLogMarginal")
        metrics = [LogMarginal(name="LogMarginal")]
        optimizer = OnlineExpectationMaximization()
        return_weighted_child_logits = False
    elif args.mode == "generative-hard-em-supervised":
        # TODO Still need to verify the performance of this, accuracy seems very low, but only had the chance to
        #  use CPU so far
        logspace_accumulators = False
        backprop_mode = BackpropMode.HARD_EM
        loss = NegativeLogJoint()
        metrics = [LogMarginal(name="LogMarginal"), keras.metrics.SparseCategoricalAccuracy(name="Accuracy")]
        optimizer = OnlineExpectationMaximization()
        return_weighted_child_logits = True
    elif args.mode == "generative-gd":
        logspace_accumulators = True
        backprop_mode = BackpropMode.GRADIENT
        loss = NegativeLogMarginal(name="NegativeLogLikelihood")
        metrics = [LogMarginal(name="LogMarginal")]
        optimizer = tf.keras.optimizers.Adam()
        return_weighted_child_logits = False
    elif args.mode == "discriminative":
        logspace_accumulators = True
        backprop_mode = BackpropMode.GRADIENT
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics = [keras.metrics.SparseCategoricalAccuracy(name="Accuracy")]
        optimizer = tf.keras.optimizers.Adam()
        return_weighted_child_logits = True
    else:
        raise ValueError("Unknown mode: {}".format(args.mode))

    x_train, y_train, x_test, y_test = get_data()

    num_vars = x_train.shape[1]

    model = get_mnist_model(
        num_vars, logspace_accumulators, backprop_mode, return_weighted_child_logits)

    # Important to use from_logits=True with the cross-entropy loss
    model.compile(optimizer=optimizer, loss=loss,  metrics=metrics)

    # Train
    model.fit(x_train, y_train, epochs=5)

    # Evaluate
    model.evaluate(x_test,  y_test, verbose=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=[
            'generative-gd',
            'generative-soft-em',
            'generative-hard-em',
            'generative-hard-em-supervised',
            'discriminative'
        ],
        required=True
    )
    main(parser.parse_args())

