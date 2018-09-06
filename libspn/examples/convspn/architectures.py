from libspn.generation.spatial import ConvSPN
import libspn as spn
from libspn.graph.spatialsum import SpatialSum
from libspn.log import get_logger
import numpy as np


logger = get_logger()


def _preprocess_prod_num_channels(*inp_nodes, prod_num_channels, kernel_size):
    kernel_surface = kernel_size ** 2 if isinstance(kernel_size, int) else np.prod(kernel_size)

    if not all(isinstance(node, (spn.GaussianLeaf, spn.IVs)) for node in inp_nodes):
        logger.warn("Preprocessing skipped. Preprocessing only works for IVs and "
                    "GaussianLeaf nodes.")
        return prod_num_channels
    first_num_channels = 1
    for node in inp_nodes:
        if isinstance(node, spn.IVs):
            first_num_channels *= node.num_vals ** kernel_surface
        else:
            first_num_channels *= node.num_components ** kernel_surface
    logger.warn("Replacing first number of prod channels '{}' with '{}', since there are "
                "no more possible permutations.".format(
        prod_num_channels[0], first_num_channels))
    return (first_num_channels,) + prod_num_channels[1:]


def full_wicker(
        *inp_nodes, spatial_dims=(28, 28), strides=(1, 2, 2, 1, 1),
        sum_node_types='local', kernel_size=2,
        sum_num_channels=(32, 32, 32, 64, 64), prod_num_channels=(16, 32, 32, 64, 64),
        num_channels_top=32, prod_node_types='default'):
    conv_spn_gen = ConvSPN()
    stack_size = int(np.ceil(np.log(spatial_dims[0]) / np.log(kernel_size)))
    if not isinstance(prod_node_types, list) and prod_node_types == 'depthwise' and \
            any(isinstance(n, spn.VarNode) for n in inp_nodes):
        prod_node_types = ['default'] + (stack_size - 1) * ['depthwise']
    elif not isinstance(prod_node_types, (tuple, list)):
        prod_node_types = [prod_node_types] * stack_size

    prod_num_channels = _preprocess_prod_num_channels(
        *inp_nodes, prod_num_channels=prod_num_channels, kernel_size=kernel_size)

    root = conv_spn_gen.full_wicker(
        *inp_nodes, sum_num_channels=sum_num_channels,
        prod_num_channels=prod_num_channels, spatial_dims=spatial_dims,
        num_channels_top=num_channels_top, strides=strides, kernel_size=kernel_size,
        sum_node_type=sum_node_types, prod_node_type=prod_node_types)
    for node in conv_spn_gen.nodes_per_level[2]:
        node.set_dropconnect_keep_prob(1.0)
    return root


def dilate_stride_double_stride(
        *inp_nodes, spatial_dims=(28, 28), sum_node_types='local', kernel_size=2,
        sum_num_channels=(32, 32), prod_num_channels=(16, 32),
        dense_gen=None, prod_node_type='default'):
    conv_spn_gen = ConvSPN()

    if not isinstance(prod_node_type, list) and prod_node_type == 'depthwise' and \
            any(isinstance(n, spn.VarNode) for n in inp_nodes):
        prod_node_type = ['default', 'depthwise']
    elif not isinstance(prod_node_type, (tuple, list)):
        prod_node_type = [prod_node_type] * 2

    prod_num_channels = _preprocess_prod_num_channels(
        *inp_nodes, prod_num_channels=prod_num_channels, kernel_size=kernel_size)

    dilate_stride0 = conv_spn_gen.add_dilate_stride(
        *inp_nodes, sum_num_channels=sum_num_channels,
        prod_num_channels=prod_num_channels, spatial_dims=spatial_dims,
        name_prefixes="DoubleD3SBottomDilateStride",
        sum_node_type=(sum_node_types[0], 'skip'), prod_node_type=prod_node_type)
    double_stride0 = conv_spn_gen.add_double_stride(
        *inp_nodes, sum_num_channels=sum_num_channels,
        prod_num_channels=prod_num_channels, spatial_dims=spatial_dims,
        name_prefixes="DoubleD3SBottomDoubleStride",
        sum_node_type=(sum_node_types[0], 'skip'), prod_node_type=prod_node_type)
    spatial_dims = double_stride0.output_shape_spatial[:2]
    if sum_node_types[1] == 'local':
        dsds_mixtures_top = spn.LocalSum(
            dilate_stride0, double_stride0, num_channels=sum_num_channels[1],
            grid_dim_sizes=spatial_dims, name="D3SBottomMixture")
    else:
        dsds_mixtures_top = spn.ConvSum(
            dilate_stride0, double_stride0, num_channels=sum_num_channels[1],
            grid_dim_sizes=spatial_dims, name="D3SBottomMixture")
    if dense_gen is not None:
        return dense_gen.generate(dsds_mixtures_top)
    for node in conv_spn_gen.nodes_per_level[2]:
        node.set_dropconnect_keep_prob(1.0)
    return dsds_mixtures_top


def dilate_stride_double_stride_full_wicker(
        *inp_nodes, spatial_dims=(28, 28), sum_node_types='local', kernel_size=2,
        sum_num_channels=(32, 32), prod_num_channels=(16, 32), num_channels_top=32,
        prod_node_types='default', strides=None, dropconnect_from=1, dropprod_to=3):
    conv_spn_gen = ConvSPN()

    prod_num_channels = _preprocess_prod_num_channels(
        *inp_nodes, prod_num_channels=prod_num_channels, kernel_size=kernel_size)

    stack_size = int(np.ceil(np.log(spatial_dims[0]) / np.log(kernel_size)))
    if not isinstance(prod_node_types, list) and prod_node_types == 'depthwise' and \
            any(isinstance(n, spn.VarNode) for n in inp_nodes):
        prod_node_types = ['default'] + (stack_size - 1) * ['depthwise']
    elif not isinstance(prod_node_types, (tuple, list)):
        prod_node_types = [prod_node_types] * stack_size

    dilate_stride0 = conv_spn_gen.add_dilate_stride(
        *inp_nodes, sum_num_channels=sum_num_channels[:2],
        prod_num_channels=prod_num_channels[:2], spatial_dims=spatial_dims,
        name_prefixes="DoubleD3SBottomDilateStride",
        sum_node_type=(sum_node_types[0], 'skip'), prod_node_type=prod_node_types[:2])
    double_stride0 = conv_spn_gen.add_double_stride(
        *inp_nodes, sum_num_channels=sum_num_channels[:2],
        prod_num_channels=prod_num_channels[:2], spatial_dims=spatial_dims,
        name_prefixes="DoubleD3SBottomDoubleStride",
        sum_node_type=(sum_node_types[0], 'skip'), prod_node_type=prod_node_types[:2])

    level, spatial_dims, input_nodes = conv_spn_gen._prepare_inputs(dilate_stride0, double_stride0)
    spatial_dims = double_stride0.output_shape_spatial[:2]
    if sum_node_types[1] == 'local':
        dsds_mixtures_top = spn.LocalSum(
            dilate_stride0, double_stride0, num_channels=sum_num_channels[1],
            grid_dim_sizes=spatial_dims, name="D3SBottomMixture")
    else:
        dsds_mixtures_top = spn.ConvSum(
            dilate_stride0, double_stride0, num_channels=sum_num_channels[1],
            grid_dim_sizes=spatial_dims, name="D3SBottomMixture")
    conv_spn_gen._register_node(dsds_mixtures_top, level)

    spatial_dims = dsds_mixtures_top.output_shape_spatial[:2]
    root = conv_spn_gen.full_wicker(
        dsds_mixtures_top, sum_num_channels=sum_num_channels[2:],
        prod_num_channels=prod_num_channels[2:], spatial_dims=spatial_dims,
        strides=strides or 1, kernel_size=kernel_size, num_channels_top=num_channels_top,
        sum_node_type=sum_node_types[2:], prod_node_type=prod_node_types[2:])

    for i, nodes in conv_spn_gen.nodes_per_level.items():
        for n in nodes:
            if isinstance(n, spn.ConvProd2D) and i > dropprod_to * 2:
                print("Turning off dropout for {}".format(n))
                n.set_dropout_keep_prob(1.0)

    for i in range(2, 2 + 2 * dropconnect_from, 2):
        for node in conv_spn_gen.nodes_per_level[i]:
            node.set_dropconnect_keep_prob(1.0)
    return root


def double_stride_full_wicker(
        *inp_nodes, spatial_dims=(28, 28), sum_node_types='local', kernel_size=2,
        sum_num_channels=(32, 32), prod_num_channels=(16, 32), num_channels_top=32,
        prod_node_types='default', strides=None, dropconnect_from=1, dropprod_to=3):
    conv_spn_gen = ConvSPN()

    prod_num_channels = _preprocess_prod_num_channels(
        *inp_nodes, prod_num_channels=prod_num_channels, kernel_size=kernel_size)

    stack_size = int(np.ceil(np.log(spatial_dims[0]) / np.log(kernel_size)))
    if not isinstance(prod_node_types, list) and prod_node_types == 'depthwise' and \
            any(isinstance(n, spn.VarNode) for n in inp_nodes):
        prod_node_types = ['default'] + (stack_size - 1) * ['depthwise']
    elif not isinstance(prod_node_types, (tuple, list)):
        prod_node_types = [prod_node_types] * stack_size

    double_stride0 = conv_spn_gen.add_double_stride(
        *inp_nodes, sum_num_channels=sum_num_channels[:2],
        prod_num_channels=prod_num_channels[:2], spatial_dims=spatial_dims,
        name_prefixes="DoubleD3SBottomDoubleStride",
        sum_node_type=(sum_node_types[0], 'skip'), prod_node_type=prod_node_types[:2])

    level, spatial_dims, input_nodes = conv_spn_gen._prepare_inputs(double_stride0)
    spatial_dims = double_stride0.output_shape_spatial[:2]
    if sum_node_types[1] == 'local':
        dsds_mixtures_top = spn.LocalSum(
            double_stride0, num_channels=sum_num_channels[1],
            grid_dim_sizes=spatial_dims, name="D3SBottomMixture")
    else:
        dsds_mixtures_top = spn.ConvSum(
            double_stride0, num_channels=sum_num_channels[1],
            grid_dim_sizes=spatial_dims, name="D3SBottomMixture")
    conv_spn_gen._register_node(dsds_mixtures_top, level)

    spatial_dims = dsds_mixtures_top.output_shape_spatial[:2]
    root = conv_spn_gen.full_wicker(
        dsds_mixtures_top, sum_num_channels=sum_num_channels[2:],
        prod_num_channels=prod_num_channels[2:], spatial_dims=spatial_dims,
        strides=strides or 1, kernel_size=kernel_size, num_channels_top=num_channels_top,
        sum_node_type=sum_node_types[2:], prod_node_type=prod_node_types[2:])

    for i, nodes in conv_spn_gen.nodes_per_level.items():
        for n in nodes:
            if isinstance(n, spn.ConvProd2D) and i > dropprod_to * 2:
                print("Turning off dropout for {}".format(n))
                n.set_dropout_keep_prob(1.0)

    for i in range(2, 2 + 2 * dropconnect_from, 2):
        for node in conv_spn_gen.nodes_per_level[i]:
            node.set_dropconnect_keep_prob(1.0)
    return root


def double_dilate_stride_double_stride(
        *inp_nodes, spatial_dims=(28, 28), sum_node_types='local', kernel_size=2,
        sum_num_channels=(32, 32), prod_num_channels=(16, 32), prod_node_types='default',
        dense_gen=None):
    conv_spn_gen = ConvSPN()

    if not isinstance(prod_node_types, list) and prod_node_types == 'depthwise' and \
            any(isinstance(n, spn.VarNode) for n in inp_nodes):
        prod_node_types = ['default'] + 3 * ['depthwise']
    elif not isinstance(prod_node_types, (tuple, list)):
        prod_node_types = [prod_node_types] * 4

    prod_num_channels = _preprocess_prod_num_channels(
        *inp_nodes, prod_num_channels=prod_num_channels, kernel_size=kernel_size)

    dilate_stride0 = conv_spn_gen.add_dilate_stride(
        *inp_nodes, sum_num_channels=sum_num_channels[:2],
        prod_num_channels=prod_num_channels[:2], spatial_dims=spatial_dims,
        name_prefixes="DoubleD3SBottomDilateStride",
        sum_node_type=(sum_node_types[0], 'skip'), prod_node_type=prod_node_types[:2])
    double_stride0 = conv_spn_gen.add_double_stride(
        *inp_nodes, sum_num_channels=sum_num_channels[:2],
        prod_num_channels=prod_num_channels[:2], spatial_dims=spatial_dims,
        name_prefixes="DoubleD3SBottomDoubleStride",
        sum_node_type=(sum_node_types[0], 'skip'), prod_node_type=prod_node_types[:2])
    spatial_dims = double_stride0.output_shape_spatial[:2]

    if sum_node_types[1] == 'local':
        dsds_mixtures = spn.LocalSum(
            dilate_stride0, double_stride0, num_channels=sum_num_channels[1],
            grid_dim_sizes=spatial_dims, name="D3SBottomMixture")
    else:
        dsds_mixtures = spn.ConvSum(
            dilate_stride0, double_stride0, num_channels=sum_num_channels[1],
            grid_dim_sizes=spatial_dims, name="D3SBottomMixture")

    pad_bottom = (4 - (spatial_dims[0] % 4), None)
    pad_right = (4 - (spatial_dims[1] % 4), None)

    dilate_stride1 = conv_spn_gen.add_dilate_stride(
        dsds_mixtures, sum_num_channels=sum_num_channels[2:],
        prod_num_channels=prod_num_channels[2:], spatial_dims=spatial_dims,
        name_prefixes="DoubleD3STopDilateStride", pad_right=pad_right,
        pad_bottom=pad_bottom,
        sum_node_type=(sum_node_types[2], 'skip'), prod_node_type=prod_node_types[2:])
    double_stride1 = conv_spn_gen.add_double_stride(
        dsds_mixtures, double_stride0, sum_num_channels=sum_num_channels[2:],
        prod_num_channels=prod_num_channels[2:], spatial_dims=spatial_dims,
        name_prefixes="DoubleD3STopDoubleStride", pad_right=pad_right,
        pad_bottom=pad_bottom,
        sum_node_type=(sum_node_types[2], 'skip'), prod_node_type=prod_node_types[2:])
    spatial_dims = double_stride1.output_shape_spatial[:2]

    if sum_node_types[3] == 'local':
        dsds_mixtures_top = spn.LocalSum(
            dilate_stride1, double_stride1, num_channels=sum_num_channels[1],
            grid_dim_sizes=spatial_dims, name="D3SBottomMixture")
    else:
        dsds_mixtures_top = spn.ConvSum(
            dilate_stride1, double_stride1, num_channels=sum_num_channels[1],
            grid_dim_sizes=spatial_dims, name="D3SBottomMixture")
    if dense_gen is not None:
        return dense_gen.generate(dsds_mixtures_top)
    for node in conv_spn_gen.nodes_per_level[2]:
        node.set_dropconnect_keep_prob(1.0)
    return dsds_mixtures_top


def wicker_dense(
        *inp_nodes, spatial_dims=(28, 28), strides=(1, 2, 2),
        sum_node_types='local', kernel_size=2,
        sum_num_channels=(32, 32, 32), prod_num_channels=(16, 32, 32),
        wicker_stack_size=3, dense_gen=None, prod_node_types='default'):
    conv_spn_gen = ConvSPN()

    if not isinstance(prod_node_types, list) and prod_node_types == 'depthwise' and \
            any(isinstance(n, spn.VarNode) for n in inp_nodes):
        prod_node_types = ['default'] + (wicker_stack_size - 1) * ['depthwise']
    elif not isinstance(prod_node_types, (tuple, list)):
        prod_node_types = [prod_node_types] * wicker_stack_size

    prod_num_channels = _preprocess_prod_num_channels(
        *inp_nodes, prod_num_channels=prod_num_channels, kernel_size=kernel_size)
    root = conv_spn_gen.wicker_stack(
        *inp_nodes, stack_size=wicker_stack_size, strides=strides,
        sum_num_channels=sum_num_channels, prod_num_channels=prod_num_channels,
        name_prefix="WickerDense", dense_generator=dense_gen, spatial_dims=spatial_dims,
        sum_node_type=sum_node_types, prod_node_type=prod_node_types)
    return root

