import argparse

import tensorflow as tf
from tensorflow import keras

from libspn_keras.backprop_mode import BackpropMode
from libspn_keras.layers.dense_product import DenseProduct
from libspn_keras.layers.log_dropout import LogDropout
from libspn_keras.layers.spatial_product import PatchWiseProduct
from libspn_keras.layers.root_sum import RootSum
from libspn_keras.layers.spatial_local_sum import SpatialLocalSum
from libspn_keras.layers.undecompose import Undecompose
from libspn_keras.losses.negative_log_joint import NegativeLogJoint
from libspn_keras.losses.negative_log_marginal import NegativeLogMarginal
from libspn_keras.metrics.log_likelihood import LogMarginal
from libspn_keras.models import build_ratspn, build_dgcspn
from libspn_keras.layers.decompose import Decompose
from libspn_keras.layers.normal_leaf import NormalLeaf
from libspn_keras.layers.dense_sum import DenseSum
import numpy as np

from libspn_keras.optimizers.online_expectation_maximization import OnlineExpectationMaximization


def get_data(spatial):

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    if not spatial:
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)
        reduce_axis = 1
    else:
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)
        reduce_axis = (1, 2, 3)

    x_train = (x_train - np.mean(x_train, axis=reduce_axis, keepdims=True)) / \
              (np.std(x_train, axis=reduce_axis, keepdims=True) + 1e-4)
    x_test = (x_test - np.mean(x_test, axis=reduce_axis, keepdims=True)) / \
              (np.std(x_test, axis=reduce_axis, keepdims=True) + 1e-4)

    return x_train, y_train, x_test, y_test


def get_ratspn_model(num_vars, logspace_accumulators, backprop_mode, return_weighted_child_logits):
    sum_product_stack = []

    accumulator_initializer = tf.initializers.TruncatedNormal(mean=0.5)

    # The 'backbone' stack of alternating sums and products
    for _ in range(int(np.floor(np.log2(num_vars)))):
        sum_product_stack.extend([
            DenseProduct(
                num_factors=2
            ),
            DenseSum(
                num_sums=4,
                logspace_accumulators=logspace_accumulators,
                accumulator_initializer=accumulator_initializer,
                backprop_mode=backprop_mode
            ),
        ])

    # Add another layer for joining the last two scopes to the final one, followed by a class-wise root layer
    # which is then followed by undecomposing (combining decompositions) and finally followed
    # by a root sum. In this case we return
    sum_product_stack.extend([
        DenseProduct(
            num_factors=2
        ),
        DenseSum(
            num_sums=1,
            logspace_accumulators=logspace_accumulators,
            accumulator_initializer=accumulator_initializer,
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
    return build_ratspn(
        num_vars=num_vars,
        decomposer=Decompose(num_decomps=10),
        leaf=NormalLeaf(num_components=4),
        sum_product_stack=keras.models.Sequential(sum_product_stack),
    )


def get_dgcspn_model(
        input_shape, logspace_accumulators, backprop_mode, return_weighted_child_logits):
    sum_product_stack = []

    spatial_dims = int(np.sqrt(np.prod(input_shape[0:2])))

    accumulator_initializer = tf.initializers.TruncatedNormal(mean=0.5)

    # The 'backbone' stack of alternating sums and products
    for i in range(2):
        sum_product_stack.extend([
            PatchWiseProduct(
                strides=[2, 2],
                dilations=[1, 1],
                num_channels=16,
                kernel_size=[2, 2],
                padding='valid'
            ),
            SpatialLocalSum(
                num_sums=16,
                logspace_accumulators=logspace_accumulators,
                accumulator_initializer=accumulator_initializer,
                backprop_mode=backprop_mode
            ),
        ])

    stack_size = int(np.ceil(np.log2(spatial_dims // 4)))
    for i in range(stack_size):
        sum_product_stack.extend([
            PatchWiseProduct(
                strides=[1, 1],
                dilations=[2 ** i, 2 ** i],
                num_channels=16,
                kernel_size=[2, 2],
                padding='full'
            ),
            SpatialLocalSum(
                num_sums=16,
                logspace_accumulators=logspace_accumulators,
                accumulator_initializer=accumulator_initializer,
                backprop_mode=backprop_mode
            ),
        ])

    # Add another layer for joining the last two scopes to the final one, followed by a
    # class-wise root layer which is then followed by undecomposing (combining decompositions) and
    # finally followed by a root sum. In this case we return
    sum_product_stack.extend([
        PatchWiseProduct(
            strides=[1, 1],
            dilations=[2 ** stack_size, 2 ** stack_size],
            num_channels=32,
            kernel_size=[2, 2],
            padding='final'
        ),
        SpatialLocalSum(
            num_sums=32,
            logspace_accumulators=logspace_accumulators,
            accumulator_initializer=accumulator_initializer,
            backprop_mode=backprop_mode
        ),
        LogDropout(rate=0.25),
        keras.layers.Flatten(),
        keras.layers.Lambda(
            lambda x: tf.reshape(x, tf.concat([[1, 1], tf.shape(x)], axis=0))
        ),
        DenseSum(
            num_sums=10,
            logspace_accumulators=logspace_accumulators,
            accumulator_initializer=accumulator_initializer,
            backprop_mode=backprop_mode
        ),
        RootSum(
            logspace_accumulators=logspace_accumulators,
            return_weighted_child_logits=return_weighted_child_logits,
            backprop_mode=backprop_mode
        )
    ])

    # Use helper function to build the actual SPN
    return build_dgcspn(
        input_shape=input_shape,
        leaf=NormalLeaf(num_components=4),
        sum_product_stack=keras.models.Sequential(sum_product_stack),
    )


def main(args):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    if args.mode == "generative-hard-em":
        logspace_accumulators = False
        backprop_mode = BackpropMode.HARD_EM
        loss = NegativeLogMarginal(name="NegativeLogMarginal")
        metrics = [LogMarginal(name="LogMarginal")]
        optimizer = OnlineExpectationMaximization()
        return_weighted_child_logits = False
    elif args.mode == "generative-hard-em-unweighted":
        logspace_accumulators = False
        backprop_mode = BackpropMode.HARD_EM_UNWEIGHTED
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
        logspace_accumulators = False
        backprop_mode = BackpropMode.HARD_EM
        loss = NegativeLogJoint()
        metrics = [
            LogMarginal(name="LogMarginal"),
            keras.metrics.SparseCategoricalAccuracy(name="Accuracy")
        ]
        optimizer = OnlineExpectationMaximization()
        return_weighted_child_logits = True
    elif args.mode == "generative-hard-em-unweighted-supervised":
        logspace_accumulators = False
        backprop_mode = BackpropMode.HARD_EM_UNWEIGHTED
        loss = NegativeLogJoint()
        metrics = [
            LogMarginal(name="LogMarginal"),
            keras.metrics.SparseCategoricalAccuracy(name="Accuracy")
        ]
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
        optimizer = tf.keras.optimizers.Adam(5e-3)
        return_weighted_child_logits = True
    else:
        raise ValueError("Unknown mode: {}".format(args.mode))

    x_train, y_train, x_test, y_test = get_data(args.model == 'dgcspn')

    if args.model == 'ratspn':
        num_vars = x_train.shape[1]
        model = get_ratspn_model(
            num_vars, logspace_accumulators, backprop_mode, return_weighted_child_logits)
    else:
        model = get_dgcspn_model(
            x_train.shape[1:], logspace_accumulators, backprop_mode, return_weighted_child_logits)

    # Important to use from_logits=True with the cross-entropy loss
    model.compile(optimizer=optimizer, loss=loss,  metrics=metrics, run_eagerly=False)

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
            'generative-hard-em-unweighted',
            'generative-hard-em-supervised-unweighted',
            'discriminative'
        ],
        required=True
    )
    parser.add_argument("--model", default='ratspn', choices=['ratspn', 'dgcspn'])
    main(parser.parse_args())

