import numpy as np
import tensorflow as tf

from libspn_keras.initializers.epsilon_inverse_fan_in import EpsilonInverseFanIn
from libspn_keras.initializers.equidistant import Equidistant
from libspn_keras.initializers.poon_domingos import PoonDomingosMeanOfQuantileSplit
from libspn_keras.layers.conv_product import ConvProduct
from libspn_keras.layers.dense_sum import DenseSum
from libspn_keras.layers.log_dropout import LogDropout
from libspn_keras.layers.normal_leaf import NormalLeaf
from libspn_keras.layers.reshape_spatial_to_dense import ReshapeSpatialToDense
from libspn_keras.layers.root_sum import RootSum
from libspn_keras.layers.spatial_local_sum import SpatialLocalSum
from libspn_keras.models import SpatialSumProductNetwork
from libspn_keras.normalizationaxes import NormalizationAxes
from itertools import cycle


class ArchConfig:

    def __init__(self, depthwise, num_non_overlapping, sum_num_channels):
        self.depthwise = depthwise
        self.num_non_overlapping = num_non_overlapping
        self.sum_num_channels = sum_num_channels


def get_config(name):
    if name == "olivetti":
        return ArchConfig(
            depthwise=iter([False, False, False, False, False, False, True]),
            num_non_overlapping=0,
            sum_num_channels=iter([4, 4, 2, 2, 2, 16])
        )
    if name == "mnist":
        ArchConfig(
            depthwise=iter([False, True, True, True, True, True]),
            num_non_overlapping=2,
            sum_num_channels=cycle([32])
        )
    else:
        raise ValueError("Unknown config")


def get_dgcspn_model(
    input_shape, logspace_accumulators, backprop_mode, return_weighted_child_logits,
    completion_by_posterior_marginal=False, initialization_data=None,
    location_trainable=False, weight_stddev=0.1, discriminative=False,
    dropout_rate=None, cdf_rate=None, input_dropout_rate=None, accumulator_init_epsilon=1e-4,
    config_name='olivetti'
):
    sum_product_stack = []

    spatial_dims = int(np.sqrt(np.prod(input_shape[0:2])))

    if initialization_data is not None:
        location_initializer = PoonDomingosMeanOfQuantileSplit(data=initialization_data.squeeze())
    else:
        location_initializer = Equidistant(minval=-2.0, maxval=2.0)

    leaf = NormalLeaf(
        num_components=4, location_initializer=location_initializer,
        location_trainable=location_trainable,
        scale_trainable=False
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
    )
    config = get_config(config_name)

    # The 'backbone' stack of alternating sums and products
    for _ in range(config.num_non_overlapping):
        sum_product_stack.append(
            ConvProduct(
                strides=[2, 2], dilations=[1, 1], kernel_size=[2, 2], padding='valid',
                depthwise=next(config.depthwise)
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
                depthwise=next(config.depthwise)
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
            padding='final', depthwise=next(config.depthwise)
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
        RootSum(return_weighted_child_logits=return_weighted_child_logits, **sum_kwargs)
    )

    return SpatialSumProductNetwork(
        leaf=leaf,
        sum_product_stack=sum_product_stack,
        input_dropout_rate=input_dropout_rate,
        normalization_axes=NormalizationAxes.PER_SAMPLE,
        cdf_rate=cdf_rate,
        completion_by_posterior_marginal=completion_by_posterior_marginal
    )