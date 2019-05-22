from libspn.graph.op.local_sums import LocalSums
from libspn.graph.op.conv_products import ConvProducts
from libspn.graph.op.conv_products_depthwise import ConvProductsDepthwise
from libspn.graph.op.parallel_sums import ParallelSums
from libspn.graph.op.sum import Sum
import numpy as np


def wicker_convspn_two_non_overlapping(
        in_var, num_channels_prod, num_channels_sums, num_classes=10, edge_size=28,
        first_depthwise=False, supervised=True):
    stack_size = int(np.ceil(np.log2(edge_size))) - 2

    if first_depthwise:
        prod0 = ConvProductsDepthwise(
            in_var, padding='valid', kernel_size=2, strides=2,
            spatial_dim_sizes=[edge_size, edge_size])
    else:
        prod0 = ConvProducts(
            in_var, num_channels=num_channels_prod[0], padding='valid', kernel_size=2, strides=2,
            spatial_dim_sizes=[edge_size, edge_size])
    sum0 = LocalSums(prod0, num_channels=num_channels_sums[0])
    prod1 = ConvProductsDepthwise(sum0, padding='valid', kernel_size=2, strides=2)
    h = LocalSums(prod1, num_channels=num_channels_sums[1])

    for i in range(stack_size):
        dilation_rate = 2 ** i
        h = ConvProductsDepthwise(
            h, padding='full', kernel_size=2, strides=1, dilation_rate=dilation_rate)
        h = LocalSums(h, num_channels=num_channels_sums[2 + i])

    full_scope_prod = ConvProductsDepthwise(
        h, padding='wicker_top', kernel_size=2, strides=1, dilation_rate=2 ** stack_size)
    if supervised:
        class_roots = ParallelSums(full_scope_prod, num_sums=num_classes)
        root = Sum(class_roots)
        return root, class_roots

    return Sum(full_scope_prod), None


def full_wicker(in_var, num_channels_prod, num_channels_sums, num_classes=10, edge_size=28,
                first_depthwise=False, supervised=True):
    stack_size = int(np.ceil(np.log2(edge_size))) - 1

    if first_depthwise:
        prod0 = ConvProductsDepthwise(
            in_var, padding='full', kernel_size=2, strides=1,
            spatial_dim_sizes=[edge_size, edge_size])
    else:
        prod0 = ConvProducts(
            in_var, num_channels=num_channels_prod[0], padding='full', kernel_size=2, strides=1,
            spatial_dim_sizes=[edge_size, edge_size])
    h = LocalSums(prod0, num_channels=num_channels_sums[0])

    for i in range(stack_size):
        dilation_rate = 2 ** (i + 1)
        h = ConvProductsDepthwise(
            h, padding='full', kernel_size=2, strides=1, dilation_rate=dilation_rate)
        h = LocalSums(h, num_channels=num_channels_sums[1 + i])

    full_scope_prod = ConvProductsDepthwise(
        h, padding='wicker_top', kernel_size=2, strides=1, dilation_rate=2 ** (stack_size + 1))
    if supervised:
        class_roots = ParallelSums(full_scope_prod, num_sums=num_classes)
        root = Sum(class_roots)
        return root, class_roots
    return Sum(full_scope_prod), None