import numpy as np
import tensorflow as tf

from libspn_keras.dimension_permutation import DimensionPermutation
from libspn_keras.initializers.epsilon_inverse_fan_in import EpsilonInverseFanIn
from libspn_keras.initializers.equidistant import Equidistant
from libspn_keras.initializers.poon_domingos import PoonDomingosMeanOfQuantileSplit
from libspn_keras.layers.conv_product import ConvProduct
from libspn_keras.layers.dense_product import DenseProduct
from libspn_keras.layers.dense_sum import DenseSum
from libspn_keras.layers.log_dropout import LogDropout
from libspn_keras.layers.location_scale_leaf import NormalLeaf, CauchyLeaf, LaplaceLeaf
from libspn_keras.layers.random_decompositions import RandomDecompositions
from libspn_keras.layers.reshape_spatial_to_dense import ReshapeSpatialToDense
from libspn_keras.layers.root_sum import RootSum
from libspn_keras.layers.spatial_local_sum import SpatialLocalSum
from libspn_keras.layers.undecompose import Undecompose
from libspn_keras.models import SpatialSumProductNetwork, DenseSumProductNetwork
from libspn_keras.normalizationaxes import NormalizationAxes
from tensorflow.keras import initializers
from itertools import cycle


class ArchConfig:

    def __init__(self, depthwise, num_non_overlapping, sum_num_channels, num_components,
                 prod_num_channels):
        self.depthwise = depthwise
        self.num_non_overlapping = num_non_overlapping
        self.sum_num_channels = sum_num_channels
        self.num_components = num_components
        self.prod_num_channels = prod_num_channels


def get_config(name):
    if name == "olivetti":
        # This architecture was used in "Deep Generalized Convolutional Sum-Product Networks for
        # Probablistic Image Representations, Jos van de Wolfshaar, Andrzej Pronobis (2019)"
        return ArchConfig(
            depthwise=iter([True, False, False, False, False, False, False]),
            num_non_overlapping=0,
            sum_num_channels=cycle([2]),
            num_components=4,
            prod_num_channels=cycle([None])
        )
    if name == "mnist":
        return ArchConfig(
            depthwise=iter([True, True, True, True, True, True]),
            num_non_overlapping=2,
            sum_num_channels=cycle([8]),
            num_components=8,
            prod_num_channels=cycle([None])
        )
    if name == "cifar10":
        return ArchConfig(
            depthwise=iter([True, True, True, True, True, True]),
            num_non_overlapping=1,
            sum_num_channels=iter([32, 32, 64, 64, 64]),
            num_components=32,
            prod_num_channels=cycle([None])
        )
    else:
        raise ValueError("Unknown config")


def construct_dgcspn_model(
    input_shape, logspace_accumulators, backprop_mode, return_weighted_child_logits,
    completion_by_posterior_marginal=False, initialization_data=None,
    location_trainable=False, weight_stddev=0.1, discriminative=False,
    dropout_rate=None, cdf_rate=None, input_dropout_rate=None, accumulator_init_epsilon=1e-4,
    config_name='olivetti', normalization_epsilon=1e-8, accumulator_regularizer=None,
    with_evidence_mask=False, leaf_type=None
):
    sum_product_stack = []

    spatial_dims = int(np.sqrt(np.prod(input_shape[0:2])))

    if input_shape[-1] > 1:
        location_initializer = initializers.TruncatedNormal(stddev=1.0)
    elif initialization_data is not None:
        location_initializer = PoonDomingosMeanOfQuantileSplit(
            data=initialization_data.squeeze(), normalization_epsilon=normalization_epsilon)
    else:
        location_initializer = Equidistant(minval=-2.0, maxval=2.0)

    config = get_config(config_name)
    leaf = dict(normal=NormalLeaf, cauchy=CauchyLeaf, laplace=LaplaceLeaf)[leaf_type](
        num_components=config.num_components, location_initializer=location_initializer,
        location_trainable=location_trainable,
        scale_trainable=False, dimension_permutation=DimensionPermutation.BATCH_FIRST
    )
    accumulator_initializer = (
        tf.initializers.TruncatedNormal(mean=1.0, stddev=weight_stddev)
        if logspace_accumulators
        else EpsilonInverseFanIn(axis=2, epsilon=accumulator_init_epsilon)
    )
    sum_kwargs = dict(
        logspace_accumulators=logspace_accumulators,
        accumulator_initializer=accumulator_initializer,
        backprop_mode=backprop_mode,
        accumulator_regularizer=accumulator_regularizer
    )

    # The 'backbone' stack of alternating sums and products
    for _ in range(config.num_non_overlapping):
        sum_product_stack.append(
            ConvProduct(
                strides=[2, 2], dilations=[1, 1], kernel_size=[2, 2], padding='valid',
                depthwise=next(config.depthwise), num_channels=next(config.prod_num_channels)
            )
        )
        if dropout_rate is not None:
            sum_product_stack.append(LogDropout(rate=dropout_rate))

        sum_product_stack.append(
            SpatialLocalSum(num_sums=next(config.sum_num_channels), **sum_kwargs)
        )

    stack_size = int(np.ceil(np.log2(spatial_dims // 2 ** config.num_non_overlapping)))
    for i in range(stack_size):
        sum_product_stack.append(
            ConvProduct(
                strides=[1, 1], dilations=[2 ** i, 2 ** i], kernel_size=[2, 2], padding='full',
                depthwise=next(config.depthwise), num_channels=next(config.prod_num_channels)
            ),
        )

        if dropout_rate is not None:
            sum_product_stack.append(LogDropout(rate=dropout_rate))

        sum_product_stack.append(
            SpatialLocalSum(num_sums=next(config.sum_num_channels), **sum_kwargs)
        )

    sum_product_stack.append(
        ConvProduct(
            strides=[1, 1], dilations=[2 ** stack_size, 2 ** stack_size], kernel_size=[2, 2],
            padding='final', depthwise=next(config.depthwise),
            num_channels=next(config.prod_num_channels)
        )
    )
    if dropout_rate is not None:
        LogDropout(rate=dropout_rate),

    sum_product_stack.extend([
        ReshapeSpatialToDense()
    ])

    if discriminative:
        sum_product_stack.append(DenseSum(num_sums=10, **sum_kwargs))

    accumulator_initializer = (
        tf.initializers.TruncatedNormal(mean=1.0, stddev=weight_stddev)
        if logspace_accumulators
        else EpsilonInverseFanIn(axis=0, epsilon=accumulator_init_epsilon)
    )
    sum_kwargs['accumulator_initializer'] = accumulator_initializer
    sum_product_stack.append(
        RootSum(return_weighted_child_logits=return_weighted_child_logits,
                dimension_permutation=DimensionPermutation.SCOPES_DECOMPS_FIRST, **sum_kwargs)
    )

    return SpatialSumProductNetwork(
        leaf=leaf,
        sum_product_stack=sum_product_stack,
        input_dropout_rate=input_dropout_rate,
        normalization_axes=NormalizationAxes.PER_SAMPLE,
        cdf_rate=cdf_rate,
        completion_by_posterior_marginal=completion_by_posterior_marginal,
        normalization_epsilon=normalization_epsilon,
        with_evidence_mask=with_evidence_mask,
        with_evidence_mask_for_normalization=False
    )


def construct_ratspn_model(
    num_vars, logspace_accumulators, backprop_mode, return_weighted_child_logits, weight_stddev
):
    sum_product_stack = []

    accumulator_initializer = tf.initializers.TruncatedNormal(mean=0.5, stddev=weight_stddev)
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
        decomposer=RandomDecompositions(num_decomps=10, name="decompose"),
        leaf=NormalLeaf(
            num_components=4, location_initializer=location_initializer, name="normal_leaf",
            input_shape=(num_vars,)
        ),
        sum_product_stack=sum_product_stack
    )