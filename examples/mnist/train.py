import argparse

import tensorflow as tf
from tensorflow import keras

from libspn_keras.backprop_mode import BackpropMode
from libspn_keras.initializers.equidistant import Equidistant
from libspn_keras.layers.conv_product_depthwise import ConvProductDepthwise
from libspn_keras.layers.dense_product import DenseProduct
from libspn_keras.layers.log_dropout import LogDropout
from libspn_keras.layers.conv_product import ConvProduct
from libspn_keras.layers.reshape_spatial_to_dense import ReshapeSpatialToDense
from libspn_keras.layers.root_sum import RootSum
from libspn_keras.layers.spatial_local_sum import SpatialLocalSum
from libspn_keras.layers.undecompose import Undecompose
from libspn_keras.losses.negative_log_joint import NegativeLogJoint
from libspn_keras.losses.negative_log_marginal import NegativeLogMarginal
from libspn_keras.metrics.log_likelihood import LogMarginal
from libspn_keras.models import DenseSumProductNetwork, SpatialSumProductNetwork
from libspn_keras.layers.decompose import Decompose
from libspn_keras.layers.normal_leaf import NormalLeaf
from libspn_keras.layers.dense_sum import DenseSum
import numpy as np

from libspn_keras.normalizationaxes import NormalizationAxes
from libspn_keras.optimizers.online_expectation_maximization import OnlineExpectationMaximization


def get_data(spatial):

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    if not spatial:
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)
    else:
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)

    return x_train.astype(np.float32), y_train, x_test.astype(np.float32), y_test


def get_ratspn_model(num_vars, logspace_accumulators, backprop_mode, return_weighted_child_logits):
    sum_product_stack = []

    accumulator_initializer = tf.initializers.TruncatedNormal(mean=0.5, stddev=args.weight_stddev)
    location_initializer = Equidistant(minval=-2.0, maxval=2.0)

    # The 'backbone' stack of alternating sums and products
    region_depth = int(np.floor(np.log2(num_vars)))
    for i in range(region_depth):
        sum_product_stack.extend([
            DenseProduct(
                num_factors=2,
                name="dense_product_{}".format(i)
            ),
            DenseSum(
                num_sums=4,
                logspace_accumulators=logspace_accumulators,
                accumulator_initializer=accumulator_initializer,
                backprop_mode=backprop_mode,
                name="dense_sum_{}".format(i)
            ),
        ])

    sum_product_stack.extend([
        DenseProduct(
            num_factors=2,
            name="dense_product_{}".format(region_depth)
        ),
        DenseSum(
            num_sums=1,
            logspace_accumulators=logspace_accumulators,
            accumulator_initializer=accumulator_initializer,
            backprop_mode=backprop_mode,
            name="dense_sum_{}".format(region_depth)
        ),
        Undecompose(name="undecompose"),
        RootSum(
            logspace_accumulators=logspace_accumulators,
            return_weighted_child_logits=return_weighted_child_logits,
            backprop_mode=backprop_mode,
            name="root"
        )
    ])

    return DenseSumProductNetwork(
        decomposer=Decompose(num_decomps=10, name="decompose"),
        leaf=NormalLeaf(
            num_components=4, location_initializer=location_initializer, name="normal_leaf", input_shape=(num_vars,)
        ),
        sum_product_stack=sum_product_stack
    )


def get_dgcspn_model(input_shape, logspace_accumulators, backprop_mode, return_weighted_child_logits,
                     completion_by_posterior_marginal=False):
    sum_product_stack = []

    spatial_dims = int(np.sqrt(np.prod(input_shape[0:2])))

    location_initializer = Equidistant(minval=-2.0, maxval=2.0)
    leaf = NormalLeaf(
        num_components=4, location_initializer=location_initializer,
        location_trainable=args.mode == 'discriminative',
        scale_trainable=False
    )

    accumulator_initializer = (
        tf.initializers.TruncatedNormal(mean=1.0, stddev=args.weight_stddev)
        if logspace_accumulators
        else tf.initializers.Ones()
    )

    # The 'backbone' stack of alternating sums and products
    sum_product_stack.extend([
        ConvProduct(
            strides=[2, 2],
            dilations=[1, 1],
            num_channels=32,
            kernel_size=[2, 2],
            padding='valid',
        ),
        LogDropout(rate=args.dropout_rate),
        SpatialLocalSum(
            num_sums=32,
            logspace_accumulators=logspace_accumulators,
            accumulator_initializer=accumulator_initializer,
            backprop_mode=backprop_mode
        ),
        ConvProductDepthwise(
            strides=[2, 2],
            dilations=[1, 1],
            kernel_size=[2, 2],
            padding='valid'
        ),
        LogDropout(rate=args.dropout_rate),
        SpatialLocalSum(
            num_sums=32,
            logspace_accumulators=logspace_accumulators,
            accumulator_initializer=accumulator_initializer,
            backprop_mode=backprop_mode
        )
    ])

    stack_size = int(np.ceil(np.log2(spatial_dims // 4)))
    for i in range(stack_size):
        sum_product_stack.extend([
            ConvProductDepthwise(
                strides=[1, 1],
                dilations=[2 ** i, 2 ** i],
                kernel_size=[2, 2],
                padding='full'
            ),
            LogDropout(rate=args.dropout_rate),
            SpatialLocalSum(
                num_sums=32,
                logspace_accumulators=logspace_accumulators,
                accumulator_initializer=accumulator_initializer,
                backprop_mode=backprop_mode
            )
        ])

    sum_product_stack.extend([
        ConvProductDepthwise(
            strides=[1, 1],
            dilations=[2 ** stack_size, 2 ** stack_size],
            kernel_size=[2, 2],
            padding='final'
        ),
        LogDropout(rate=args.dropout_rate),
        SpatialLocalSum(
            num_sums=32,
            logspace_accumulators=logspace_accumulators,
            accumulator_initializer=accumulator_initializer,
            backprop_mode=backprop_mode
        ),
        ReshapeSpatialToDense()
    ])

    if args.mode == 'discriminative':
        sum_product_stack.append(
            DenseSum(
                num_sums=10,
                logspace_accumulators=logspace_accumulators,
                accumulator_initializer=accumulator_initializer,
                backprop_mode=backprop_mode
            )
        )

    sum_product_stack.append(
        RootSum(
            logspace_accumulators=logspace_accumulators,
            return_weighted_child_logits=return_weighted_child_logits,
            backprop_mode=backprop_mode
        )
    )

    return SpatialSumProductNetwork(
        leaf=leaf,
        sum_product_stack=sum_product_stack,
        input_dropout_rate=args.input_dropout,
        normalization_axes=NormalizationAxes.PER_SAMPLE,
        cdf_rate=args.cdf_rate,
        completion_by_posterior_marginal=completion_by_posterior_marginal
    )


def main():
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
        loss = NegativeLogMarginal(
            name="NegativeLogMarginal", reduction=keras.losses.Reduction.SUM)
        metrics = [LogMarginal(name="LogMarginal")]
        optimizer = OnlineExpectationMaximization()
        return_weighted_child_logits = False
    elif args.mode == "generative-hard-em-unweighted":
        logspace_accumulators = False
        backprop_mode = BackpropMode.HARD_EM_UNWEIGHTED
        loss = NegativeLogMarginal(
            name="NegativeLogMarginal", reduction=keras.losses.Reduction.SUM)
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
        optimizer = tf.keras.optimizers.Adam(7e-3)
        return_weighted_child_logits = True
    else:
        raise ValueError("Unknown mode: {}".format(args.mode))

    x_train, y_train, x_test, y_test = get_data(spatial=args.model == 'dgcspn')

    if args.model == 'ratspn':
        num_vars = x_train.shape[1]
        model = get_ratspn_model(
            num_vars, logspace_accumulators, backprop_mode, return_weighted_child_logits)
    else:
        model = get_dgcspn_model(
            x_train.shape[1:], logspace_accumulators, backprop_mode, return_weighted_child_logits,
            completion_by_posterior_marginal=False
        )
        model_completion = get_dgcspn_model(
            x_train.shape[1:], logspace_accumulators, BackpropMode.GRADIENT,
            return_weighted_child_logits, completion_by_posterior_marginal=True
        )
        model_completion.set_weights(model.get_weights())

    # Important to use from_logits=True with the cross-entropy loss
    model.compile(optimizer=optimizer, loss=loss,  metrics=metrics)

    # Train
    model.fit(
        x_train, y_train, epochs=args.epochs, validation_split=1/6, batch_size=args.batch_size
    )

    if args.completion:
        completion_mask = np.ones_like(x_test).astype(bool)
        completion_mask_left, completion_mask_right, completion_mask_top, completion_mask_bottom = \
            [completion_mask.copy() for _ in range(4)]
        mid = 28 // 2
        completion_mask_left[:, :, mid:, :] = False
        completion_mask_right[:, :, :mid, :] = False
        completion_mask_top[:, :mid, :, :] = False
        completion_mask_bottom[:, mid:, :, :] = False

        model_completion.compile(loss=loss)
        model_completion.evaluate(
            [x_test, completion_mask_left], x_test, verbose=2)
        model_completion.evaluate(
            [x_test, completion_mask_right], x_test, verbose=2)
        model_completion.evaluate(
            [x_test, completion_mask_top], x_test, verbose=2)
        model_completion.evaluate(
            [x_test, completion_mask_bottom], x_test, verbose=2)
    else:
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
    parser.add_argument("--input-dropout", default=None, type=float)
    parser.add_argument("--dropout-rate", default=0.0, type=float)
    parser.add_argument("--cdf-rate", default=None, type=float)
    parser.add_argument("--bounded-marginalization", default=None, type=float)
    parser.add_argument("--location-stddev", default=1.0, type=float)
    parser.add_argument("--weight-stddev", default=0.1, type=float)
    parser.add_argument("--model", default='ratspn', choices=['ratspn', 'dgcspn'])
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--completion", action='store_true', dest='completion')
    parser.add_argument("--batch-size", default=32, type=int)
    parser.set_defaults(completion=False)
    args = parser.parse_args()
    main()

