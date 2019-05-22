from libspn.tests.test import argsprod
import tensorflow as tf
import numpy as np
import libspn as spn
from libspn.log import get_logger
import itertools
import random
logger = get_logger()


def generate_sparse_connections(num_in_channels, num_out_channels, kernel_size):
    kernel_surface = int(np.prod(kernel_size))
    total_possibilities = num_in_channels ** kernel_surface
    if num_out_channels >= total_possibilities:
        p = np.arange(total_possibilities)
        kernel_cells = []
        for _ in range(kernel_surface):
            kernel_cells.append(p % num_in_channels)
            p //= num_in_channels
        return np.stack(kernel_cells, axis=0).reshape(kernel_size + [total_possibilities])\
            .transpose((2, 0, 1))

    sparse_shape = kernel_size + [num_out_channels]
    size = int(np.prod(sparse_shape))
    return np.random.randint(num_in_channels, size=size).reshape(sparse_shape).transpose((2, 0, 1))


def grid_spn(var_node, patch_size, num_mixtures, num_alterations, input_dist, num_rows, num_cols,
             max_patch_combs=128, asymmetric=False, conv=False, dilation_rate=1, strides=2):

    if isinstance(var_node, spn.NormalLeaf):
        # TODO num_components should be property
        last_dim_size = var_node._num_components
    else:
        last_dim_size = var_node.num_vals

    def sub2ind(r, c, elem):
        return r * num_cols * last_dim_size + c * last_dim_size + elem

    def _patch_combinations(patch_size, last_dim_size):
        num_scopes = int(np.prod(patch_size))
        num_combinations = last_dim_size ** num_scopes

        if num_combinations > max_patch_combs:
            combinations = np.random.choice(num_combinations, size=max_patch_combs, replace=False)
            unique_patch_combinations = []
            for c in combinations:
                combin = []
                for i in range(num_scopes):
                    combin.append(int(c % last_dim_size))
                    c //= last_dim_size
                unique_patch_combinations.append(tuple(combin))
            assert len(set(unique_patch_combinations)) == max_patch_combs
        else:
            # unique_patch_combinations = list(itertools.product(
            #     *[range(last_dim_size) for _ in range(np.product(patch_size))]))
            unique_patch_combinations = generate_sparse_connections(
                last_dim_size, num_combinations, kernel_size=list(patch_size)).reshape(
                (-1, int(np.prod(patch_size)))).astype(int).tolist()
        return unique_patch_combinations

    if isinstance(num_mixtures, int):
        num_mixtures = [num_mixtures]
    if len(num_mixtures) == 1:
        num_mixtures *= num_alterations

    offsets = list(itertools.product(range(patch_size[0]), range(patch_size[1])))

    product_layer = input_dist == spn.DenseSPNGenerator.InputDist.RAW
    if product_layer:
        unique_patch_combinations = _patch_combinations(patch_size, last_dim_size)
        print("Unique patch combinations")
        print(unique_patch_combinations)
    else:
        unique_patch_combinations = []
    node = var_node

    num_layers = num_alterations * 2 - (1 if asymmetric else 0)
    for layer_ind in range(num_layers):
        # Go over all grid positions
        print("Generating grid for layer {}, [{} x {} x {}]".format(
            layer_ind, num_rows, num_cols,
            node.get_out_size() // (num_cols * num_rows)))

        num_mixt = num_mixtures[layer_ind // 2]

        out_rows = int(np.ceil((num_rows - (patch_size[0] - 1) * dilation_rate) / strides))
        out_cols = int(np.ceil((num_cols - (patch_size[1] - 1) * dilation_rate) / strides))

        indices = []
        print("Dilation Rate", dilation_rate)
        for row in range(out_rows):
            for col in range(out_cols):
                if product_layer:
                    # print(row, col, len(unique_patch_combinations))
                    for patch_comb in unique_patch_combinations:
                        patch_indices = []
                        assert len(offsets) == len(patch_comb), \
                            "Length of offsets and patch combinations do not match... {} vs. {}" \
                            "".format(len(offsets), len(patch_comb))
                        for (dr, dc), elem in zip(offsets, patch_comb):
                            ind = sub2ind(row * strides + dr * dilation_rate,
                                          col * strides + dc * dilation_rate, elem)
                            patch_indices.append(ind)
                            # print(row + dr * dilation_rate, col + dc * dilation_rate, end=", ")
                        # print("\n")
                        indices.extend(patch_indices)
                else:  # sums_layer
                    mixture_indices = []
                    # print(row, col)
                    for e in range(last_dim_size):
                        ind = sub2ind(row, col, e)
                        mixture_indices.append(ind)
                    for _ in range(num_mixt):
                        indices.extend(mixture_indices)

        last_dim_size = len(unique_patch_combinations) if product_layer else num_mixt
        print("Constructing layer")
        if product_layer:
            # if dilation_rate == 1:
            num_rows = int(np.ceil((num_rows - (patch_size[0] - 1) * dilation_rate) / strides))
            num_cols = int(np.ceil((num_cols - (patch_size[1] - 1) * dilation_rate) / strides))

            print("Layer out size {} x {}".format(num_rows, num_cols))
            print("Size per prod {}".format(len(indices) / (num_rows * num_cols * last_dim_size)))
            node = spn.ProductsLayer(
                (node, indices), name="GridProducts{}".format(layer_ind),
                num_or_size_prods=num_rows * num_cols * last_dim_size)
        else:
            if conv:
                node = spn.ConvSums(
                    node, num_channels=num_mixt, spatial_dim_sizes=[num_rows, num_cols])
            else:
                node = spn.SumsLayer(
                    (node, indices), name="GridSums{}".format(layer_ind),
                    num_or_size_sums=num_rows * num_cols * last_dim_size)
            if layer_ind != num_layers - 1:
                unique_patch_combinations = _patch_combinations(patch_size, last_dim_size)

        print("Added layer {}".format(node.name))
        # Switch to other layer type
        product_layer = not product_layer

    print("Built grid spn")
    return node


class TestConvProd(tf.test.TestCase):

    def test_generate_sparse_connections(self):
        grid_dims = [4, 4]
        input_channels = 2
        vars = spn.RawLeaf(num_vars=grid_dims[0] * grid_dims[1] * input_channels)

        convprod = spn.ConvProducts(
            vars, num_channels=32, strides=2, padding='valid', spatial_dim_sizes=grid_dims)

        connections = convprod.generate_sparse_kernels(32)
        connection_tuples = [tuple(c) for c in
                             connections.reshape((-1, convprod._num_channels)).transpose()]
        self.assertEqual(len(set(connection_tuples)), len(connection_tuples))

    def test_compute_log_value(self):
        grid_dims = [4, 4]
        input_channels = 2
        vars = spn.RawLeaf(num_vars=grid_dims[0] * grid_dims[1] * input_channels)
        convprod = spn.ConvProducts(vars, num_channels=32, padding='valid', strides=2,
                                    spatial_dim_sizes=grid_dims)
        connectivity = [(0, 0, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0), (1, 1, 0, 0),
                        (0, 0, 1, 0), (1, 0, 1, 0), (0, 1, 1, 0), (1, 1, 1, 0),
                        (0, 0, 0, 1), (1, 0, 0, 1), (0, 1, 0, 1), (1, 1, 0, 1),
                        (0, 0, 1, 1), (1, 0, 1, 1), (0, 1, 1, 1), (1, 1, 1, 1)]
        feed = [
            [[1, 2], [1, 2], [2, 1], [2, 1]],
            [[1, 1], [1, 1], [1, 1], [1, 1]],
            [[3, 1], [1, 3], [3, 1], [1, 3]],
            [[2, 3], [2, 3], [3, 2], [3, 2]]
        ]
        feed = np.exp(np.reshape(feed, (1, grid_dims[0] * grid_dims[1] * input_channels)))

        truth = [
            [[4, 6], [8, 10]],      # 0
            [[5, 5], [6, 8]],       # 1
            [[5, 5], [10, 12]],     # 2
            [[6, 4], [8, 10]],      # 3
            [[4, 6], [9, 9]],       # 4
            [[5, 5], [7, 7]],       # 5
            [[5, 5], [11, 11]],     # 6
            [[6, 4], [9, 9]],       # 7
            [[4, 6], [9, 9]],       # 8
            [[5, 5], [7, 7]],       # 9
            [[5, 5], [11, 11]],     # 10
            [[6, 4], [9, 9]],       # 11
            [[4, 6], [10, 8]],      # 12
            [[5, 5], [8, 6]],       # 13
            [[5, 5], [12, 10]],     # 14
            [[6, 4], [10, 8]]       # 15
        ]
        truth = np.transpose(truth, (1, 2, 0))
        logval_op = tf.reshape(convprod.get_log_value(spn.InferenceType.MARGINAL), (2, 2, 16))

        with self.test_session() as sess:
            logval_out = sess.run(logval_op, feed_dict={vars: feed})

        self.assertAllClose(logval_out, truth)

    def test_conv_min_inf(self):
        image = np.log(np.stack([np.exp(np.ones((4, 4))), np.zeros((4, 4))], axis=-1)
                       .reshape((1, 4, 4, 2)))
        image = tf.constant(image, dtype=tf.float32)
        image = tf.where(tf.is_inf(image), tf.fill([1, 4, 4, 2], value=-1e20), image)
        filter = np.concatenate([np.ones((2, 2, 1, 1)), np.zeros((2, 2, 1, 1))], axis=2)
        conv_op = tf.nn.conv2d(input=image, filter=filter, strides=[1, 2, 2, 1], padding="VALID")

        with self.test_session() as sess:
            conv_out = sess.run(conv_op)

        self.assertAllClose(conv_out, np.ones((1, 2, 2, 1)) * 4)

    def test_compute_mpe_path(self):
        grid_dims = [4, 4]
        input_channels = 2
        vars = spn.RawLeaf(num_vars=grid_dims[0] * grid_dims[1] * input_channels)
        convprod = spn.ConvProducts(vars, num_channels=32, padding='valid', strides=2,
                                    spatial_dim_sizes=grid_dims)

        valgen = spn.LogValue(inference_type=spn.InferenceType.MARGINAL)
        valgen.get_value(convprod)


        counts = np.stack([
            np.arange(16),
            np.arange(16) + 1000,
            np.arange(16) + 10000,
            np.arange(16) + 100000]).reshape((1, 2, 2, 16)).astype(np.float32)

        var_counts = tf.reshape(
            convprod._compute_log_mpe_path(counts, valgen.values[vars])[0],
            (1, 4, 4, 2))
        truth_single_square = np.asarray(
            [[0 + 2 + 4 + 6 + 8 + 10 + 12 + 14, 1 + 3 + 5 + 7 + 9 + 11 + 13 + 15],
             [0 + 1 + 4 + 5 + 8 + 9 + 12 + 13,  2 + 3 + 6 + 7 + 10 + 11 + 14 + 15],
             [0 + 1 + 2 + 3 + 8 + 9 + 10 + 11,  4 + 5 + 6 + 7 + 12 + 13 + 14 + 15],
             [0 + 1 + 2 + 3 + 4 + 5 + 6 + 7,    8 + 9 + 10 + 11 + 12 + 13 + 14 + 15]]).reshape(
            (2, 2, 2))
        truth_top_squares = np.concatenate(
            [truth_single_square, truth_single_square + 8000], axis=1)
        truth_bottom_squares = np.concatenate(
            [truth_single_square + 80000, truth_single_square + 800000], axis=1)
        truth = np.concatenate((truth_top_squares, truth_bottom_squares), axis=0).reshape(
            (1, 4, 4, 2))

        with self.test_session() as sess:
            var_counts_out = sess.run(var_counts, feed_dict={vars: np.random.rand(1, 4 * 4 * 2)})

        self.assertAllClose(truth, var_counts_out)

    def test_compute_mpe_path_dilated(self):
        grid_dims = [4, 4]
        input_channels = 2
        vars = spn.RawLeaf(num_vars=grid_dims[0] * grid_dims[1] * input_channels)
        convprod = spn.ConvProducts(vars, num_channels=32, padding='valid', strides=1,
                                    spatial_dim_sizes=grid_dims, dilation_rate=2)

        valgen = spn.LogValue(inference_type=spn.InferenceType.MARGINAL)
        valgen.get_value(convprod)

        counts = np.stack([
            np.arange(16) + 23 * i for i in range(4)]).reshape((1, 2, 2, 16)).astype(np.float32)

        var_counts = tf.reshape(
            convprod._compute_log_mpe_path(counts, valgen.values[vars])[0],
            (1, 4, 4, 2))

        with self.test_session() as sess:
            var_counts_out = sess.run(var_counts, feed_dict={vars: np.random.rand(1, 4 * 4 * 2)})

        print(var_counts_out)
        # self.assertAllClose(truth, var_counts_out)

    def test_compute_value_padding(self):
        grid_dims = [2, 2]
        vars = spn.RawLeaf(num_vars=4)
        convprod = spn.ConvProducts(vars, num_channels=1, strides=1, kernel_size=2,
                                    padding='full', spatial_dim_sizes=grid_dims)
        value_op = convprod.get_log_value()

        var_feed = np.exp(np.arange(16, dtype=np.float32).reshape((4, 4)))

        truth = [[0, 1, 1, 2, 6, 4, 2, 5, 3],
                 [4, 9, 5, 10, 22, 12, 6, 13, 7],
                 [8, 17, 9, 18, 38, 20, 10, 21, 11],
                 [12, 25, 13, 26, 54, 28, 14, 29, 15]]

        with self.test_session() as sess:
            value_out = sess.run(value_op, feed_dict={vars: var_feed})

        self.assertAllClose(value_out, truth)

    def test_compute_mpe_path_padding(self):
        grid_dims = [2, 2]
        vars = spn.RawLeaf(num_vars=4)
        convprod = spn.ConvProducts(vars, num_channels=1, strides=1, kernel_size=2,
                                    padding='full', spatial_dim_sizes=grid_dims)
        counts_feed = tf.constant(np.arange(18, dtype=np.float32).reshape((2, 9)))

        truth = [
            [0 + 1 + 3 + 4, 1 + 2 + 4 + 5, 3 + 4 + 6 + 7, 4 + 5 + 7 + 8],
            [9 + 10 + 12 + 13, 10 + 11 + 13 + 14, 12 + 13 + 15 + 16, 13 + 14 + 16 + 17]
        ]

        counts_op = convprod._compute_mpe_path_common(counts_feed, tf.ones(shape=(2, 4)))

        with self.test_session() as sess:
            counts_out = sess.run(counts_op)

        self.assertAllClose(counts_out[0], truth)

    @argsprod([False, True])
    def test_compute_scope(self, dilate):
        grid_dims = [4, 4]
        input_channels = 2
        vars = spn.IndicatorLeaf(num_vars=grid_dims[0] * grid_dims[1], num_vals=input_channels)
        strides = 1 if dilate else 2
        dilation_rate = 2 if dilate else 1
        convprod = spn.ConvProducts(vars, num_channels=32, padding='valid', strides=strides,
                                    spatial_dim_sizes=grid_dims, dilation_rate=dilation_rate)
        conv_prod_scope = convprod.get_scope()

        singular_scopes = np.asarray(
            [spn.Scope(vars, i) for i in range(grid_dims[0] * grid_dims[1])]).reshape((4, 4))

        if dilate:
            scope_truth = [
                spn.Scope.merge_scopes(singular_scopes[0::2, 0::2].ravel().tolist()),
                spn.Scope.merge_scopes(singular_scopes[0::2, 1::2].ravel().tolist()),
                spn.Scope.merge_scopes(singular_scopes[1::2, 0::2].ravel().tolist()),
                spn.Scope.merge_scopes(singular_scopes[1::2, 1::2].ravel().tolist())
            ]
        else:
            scope_truth = [
                spn.Scope.merge_scopes(singular_scopes[0:2, 0:2].ravel().tolist()),
                spn.Scope.merge_scopes(singular_scopes[0:2, 2:4].ravel().tolist()),
                spn.Scope.merge_scopes(singular_scopes[2:4, 0:2].ravel().tolist()),
                spn.Scope.merge_scopes(singular_scopes[2:4, 2:4].ravel().tolist())
            ]
        scope_truth = np.repeat(scope_truth, 16).tolist()

        self.assertAllEqual(conv_prod_scope, scope_truth)

    @argsprod([spn.DenseSPNGenerator.NodeType.SINGLE,
               spn.DenseSPNGenerator.NodeType.BLOCK,
               spn.DenseSPNGenerator.NodeType.LAYER],
              [spn.DenseSPNGenerator.InputDist.RAW,
               spn.DenseSPNGenerator.InputDist.MIXTURE])
    def test_compute_dense_gen_two_spatial_decomps(self, node_type, input_dist):
        input_channels = 2
        grid_dims = [8, 8]
        num_vars = grid_dims[0] * grid_dims[1]
        vars = spn.IndicatorLeaf(num_vars=num_vars, num_vals=input_channels)

        convert_after = False
        if input_dist == spn.DenseSPNGenerator.InputDist.RAW and \
            node_type in [spn.DenseSPNGenerator.NodeType.BLOCK,
                          spn.DenseSPNGenerator.NodeType.LAYER]:
            node_type = spn.DenseSPNGenerator.NodeType.SINGLE
            convert_after = True

        # First decomposition
        convprod_dilate0 = spn.ConvProducts(
            vars, spatial_dim_sizes=grid_dims, num_channels=16, padding='valid', dilation_rate=2,
            strides=1, kernel_size=2)
        convprod_dilate1 = spn.ConvProducts(
            convprod_dilate0, spatial_dim_sizes=[6, 6], num_channels=512, padding='valid',
            dilation_rate=1, strides=4, kernel_size=2)
        convsum_dilate = spn.ConvSums(convprod_dilate1, num_channels=2, spatial_dim_sizes=[2, 2])

        # Second decomposition
        convprod_stride0 = spn.ConvProducts(
            vars, spatial_dim_sizes=grid_dims, num_channels=16, padding='valid', dilation_rate=1,
            strides=2, kernel_size=2)
        convprod_stride1 = spn.ConvProducts(
            convprod_stride0, spatial_dim_sizes=[4, 4], num_channels=512, padding='valid',
            dilation_rate=1, strides=2, kernel_size=2)
        convsum_stride = spn.ConvSums(convprod_stride1, num_channels=2, spatial_dim_sizes=[2, 2])

        dense_gen = spn.DenseSPNGenerator(
            num_mixtures=2, num_decomps=1, num_subsets=2, node_type=node_type,
            input_dist=input_dist)
        root = dense_gen.generate(convsum_dilate, convsum_stride)
        if convert_after:
            root = dense_gen.convert_to_layer_nodes(root)

        # Assert valid
        self.assertTrue(root.is_valid())

        # Setup the remaining Ops
        spn.generate_weights(root)
        init = spn.initialize_weights(root)
        value_op = tf.squeeze(root.get_log_value())

        with self.test_session() as sess:
            sess.run(init)
            value_out = sess.run(value_op, {vars: -np.ones((1, num_vars), dtype=np.int32)})

        self.assertAllClose(value_out, 0.0)

    @argsprod([spn.DenseSPNGenerator.NodeType.SINGLE,
               spn.DenseSPNGenerator.NodeType.BLOCK,
               spn.DenseSPNGenerator.NodeType.LAYER],
              [spn.DenseSPNGenerator.InputDist.RAW,
               spn.DenseSPNGenerator.InputDist.MIXTURE])
    def test_compute_dense_gen_two_spatial_decomps_v2(self, node_type, input_dist):
        input_channels = 2
        grid_dims = [32, 32]
        num_vars = grid_dims[0] * grid_dims[1]
        vars = spn.IndicatorLeaf(num_vars=num_vars, num_vals=input_channels)

        convert_after = False
        if input_dist == spn.DenseSPNGenerator.InputDist.RAW and \
                node_type in [spn.DenseSPNGenerator.NodeType.BLOCK,
                              spn.DenseSPNGenerator.NodeType.LAYER]:
            node_type = spn.DenseSPNGenerator.NodeType.SINGLE
            convert_after = True

        # First decomposition
        convprod_dilate0 = spn.ConvProducts(
            vars, spatial_dim_sizes=grid_dims, num_channels=16, padding='valid', dilation_rate=2,
            strides=1, kernel_size=2)
        convprod_dilate1 = spn.ConvProducts(
            convprod_dilate0, spatial_dim_sizes=[30, 30], num_channels=512, padding='valid',
            dilation_rate=1, strides=4, kernel_size=2)
        convsum_dilate = spn.ConvSums(convprod_dilate1, num_channels=2, spatial_dim_sizes=[8, 8])

        # Second decomposition
        convprod_stride0 = spn.ConvProducts(
            vars, spatial_dim_sizes=grid_dims, num_channels=16, padding='valid', dilation_rate=1,
            strides=2, kernel_size=2)
        convprod_stride1 = spn.ConvProducts(
            convprod_stride0, spatial_dim_sizes=[16, 16], num_channels=512, padding='valid',
            dilation_rate=1, strides=2, kernel_size=2)
        convsum_stride = spn.ConvSums(convprod_stride1, num_channels=2, spatial_dim_sizes=[8, 8])

        # First decomposition level 2
        convprod_dilate0_l2 = spn.ConvProducts(
            convsum_stride, convsum_dilate, spatial_dim_sizes=[8, 8], num_channels=512,
            padding='valid', dilation_rate=2, strides=1, kernel_size=2)
        convprod_dilate1_l2 = spn.ConvProducts(
            convprod_dilate0_l2, spatial_dim_sizes=[6, 6], num_channels=512, padding='valid',
            dilation_rate=1, kernel_size=2, strides=4)
        convsum_dilate_l2 = spn.ConvSums(convprod_dilate1_l2, num_channels=2, spatial_dim_sizes=[4, 4])
        
        # Second decomposition level 2
        convprod_stride0_l2 = spn.ConvProducts(
            convsum_stride, convsum_dilate, spatial_dim_sizes=[8, 8], num_channels=512,
            padding='valid', dilation_rate=1, strides=2, kernel_size=2)
        convprod_stride1_l2 = spn.ConvProducts(
            convprod_stride0_l2, spatial_dim_sizes=[4, 4], num_channels=512,
            padding='valid', dilation_rate=1, strides=2, kernel_size=2)
        convsum_stride_l2 = spn.ConvSums(convprod_stride1_l2, num_channels=2, spatial_dim_sizes=[4, 4])

        dense_gen = spn.DenseSPNGenerator(
            num_mixtures=2, num_decomps=1, num_subsets=2, node_type=node_type,
            input_dist=input_dist)
        root = dense_gen.generate(convsum_stride_l2, convsum_dilate_l2)
        if convert_after:
            root = dense_gen.convert_to_layer_nodes(root)

        # Assert valid
        self.assertTrue(root.is_valid())

        # Setup the remaining Ops
        spn.generate_weights(root)
        init = spn.initialize_weights(root)
        value_op = tf.squeeze(root.get_log_value())

        with self.test_session() as sess:
            sess.run(init)
            value_out = sess.run(value_op, {vars: -np.ones((1, num_vars), dtype=np.int32)})

        self.assertAllClose(value_out, 0.0)

