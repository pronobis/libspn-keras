from libspn.graph.basesum import BaseSum
from libspn.tests.test import argsprod
import tensorflow as tf
import numpy as np
import libspn as spn
from libspn.log import get_logger
from libspn.generation.spatial import ConvSPN
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

    if isinstance(var_node, spn.GaussianLeaf):
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

    product_layer = input_dist == spn.DenseSPNGeneratorLayerNodes.InputDist.RAW
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
                node = spn.ConvSum(node, num_channels=num_mixt, grid_dim_sizes=[num_rows, num_cols])
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

    @argsprod([4, 2, 1], [4, 2, 1])
    def test_compare_v1_and_v2_value(self, stride, dilate):
        if dilate > 1:
            if stride > 1:
                # Not supported by TF's convolution
                return
        ivs_rows, ivs_cols = 16, 16
        batch_size = 32
        num_vars = ivs_rows * ivs_cols
        num_vals = 3
        ivs = spn.ContVars(num_vars=num_vars * 2)
        ivs2 = spn.ContVars(num_vars=num_vars * 2)

        localsum0 = spn.LocalSum(ivs, grid_dim_sizes=[ivs_rows, ivs_cols], num_channels=2)
        localsum1 = spn.LocalSum(ivs2, grid_dim_sizes=[ivs_rows, ivs_cols], num_channels=2)

        ivs_concat = spn.Concat(localsum0, localsum1, axis=3)

        convprod_v1 = spn.ConvProd2DV2(
            ivs_concat, grid_dim_sizes=[ivs_rows, ivs_cols], num_channels=256,
            strides=stride, dilation_rate=dilate)
        convprod_v2 = spn.ConvProd2D(
            ivs_concat, grid_dim_sizes=[ivs_rows, ivs_cols], num_channels=256,
            strides=stride, dilation_rate=dilate)

        spn.generate_weights(convprod_v1)
        spn.generate_weights(convprod_v2)
        init_v1 = spn.initialize_weights(convprod_v1)
        init_v2 = spn.initialize_weights(convprod_v2)

        val_v1 = convprod_v1.get_log_value()
        val_v2 = convprod_v2.get_log_value()

        # ivs_feed = np.random.randint(-1, 2, size=num_vars * batch_size).reshape(
        #     (batch_size, num_vars))
        ivs_feed = np.random.rand(batch_size, num_vars * 2)
        ivs_feed2 = np.random.rand(batch_size, num_vars * 2)

        with self.test_session() as sess:
            sess.run([init_v1, init_v2])
            val_v1_out, val_v2_out = sess.run(
                [val_v1, val_v2], {ivs: ivs_feed, ivs2: ivs_feed2})

        self.assertAllClose(np.exp(val_v1_out), np.exp(val_v2_out))

    @argsprod([4, 2, 1], [4, 2, 1], [0, 1])
    def test_compare_v1_and_v2_counts(self, stride, dilate, pad_size):
        if dilate > 1:
            if stride > 1:
                # Not supported by TF's convolution, it seems to be a bug...
                return
        ivs_rows, ivs_cols = 16, 16
        batch_size = 32
        num_vars = ivs_rows * ivs_cols
        num_vals = 3
        ivs = spn.ContVars(num_vars=num_vars * 2)
        ivs2 = spn.ContVars(num_vars=num_vars * 2)

        localsum0 = spn.LocalSum(ivs, grid_dim_sizes=[ivs_rows, ivs_cols], num_channels=2)
        localsum1 = spn.LocalSum(ivs2, grid_dim_sizes=[ivs_rows, ivs_cols], num_channels=2)

        ivs_concat = spn.Concat(localsum0, localsum1, axis=3)

        convprod_v1 = spn.ConvProd2DV2(
            ivs_concat, grid_dim_sizes=[ivs_rows, ivs_cols], num_channels=256,
            strides=stride, dilation_rate=dilate, pad_left=pad_size, pad_right=pad_size,
            pad_top=pad_size, pad_bottom=pad_size)
        convprod_v2 = spn.ConvProd2D(
            ivs_concat, grid_dim_sizes=[ivs_rows, ivs_cols], num_channels=256,
            strides=stride, dilation_rate=dilate, pad_left=pad_size, pad_right=pad_size,
            pad_top=pad_size, pad_bottom=pad_size)


        spn.generate_weights(convprod_v1)
        spn.generate_weights(convprod_v2)
        init_v1 = spn.initialize_weights(convprod_v1)
        init_v2 = spn.initialize_weights(convprod_v2)


        mpe_path_gen_v1 = spn.MPEPath()
        mpe_path_gen_v2 = spn.MPEPath()

        mpe_path_gen_v1.get_mpe_path(convprod_v1)
        mpe_path_gen_v2.get_mpe_path(convprod_v2)

        ivs_feed = np.random.rand(batch_size, num_vars * 2)
        ivs_feed2 = np.random.rand(batch_size, num_vars * 2)
        with self.test_session() as sess:
            sess.run([init_v1, init_v2])
            val_v1_out, val_v2_out = sess.run(
                [mpe_path_gen_v1.counts[ivs], mpe_path_gen_v2.counts[ivs]],
                {ivs: ivs_feed, ivs2: ivs_feed2})

        # print(val_v1_out[0])
        # print(val_v2_out[0])
        self.assertAllClose(val_v1_out, val_v2_out)

    def test_generate_sparse_connections(self):
        grid_dims = [4, 4]
        input_channels = 2
        vars = spn.ContVars(num_vars=grid_dims[0] * grid_dims[1] * input_channels)

        convprod = spn.ConvProd2D(vars, num_channels=32, strides=2, padding_algorithm='valid',
                                  grid_dim_sizes=grid_dims)

        connections = convprod.generate_sparse_kernels(32)
        connection_tuples = [tuple(c) for c in
                             connections.reshape((-1, convprod._num_channels)).transpose()]
        self.assertEqual(len(set(connection_tuples)), len(connection_tuples))

    def test_compute_log_value(self):
        grid_dims = [4, 4]
        input_channels = 2
        vars = spn.ContVars(num_vars=grid_dims[0] * grid_dims[1] * input_channels)
        convprod = spn.ConvProd2D(vars, num_channels=32, padding_algorithm='valid', strides=2,
                                  grid_dim_sizes=grid_dims)
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

    @argsprod([False, True])
    def test_compare_manual_convprod(self, stride):
        spn.conf.argmax_zero = True
        grid_dims = [4, 4]
        batch_size = 256
        num_vars = grid_dims[0] * grid_dims[1]
        ivs = spn.IVs(num_vars=num_vars, num_vals=2)

        dilation_rate = 1 if stride else 2
        stride = 2 if stride else 1

        conv_manual = grid_spn(ivs, patch_size=(2, 2), num_mixtures=[2], num_alterations=1,
                               asymmetric=True, num_rows=grid_dims[0], num_cols=grid_dims[1],
                               max_patch_combs=16,
                               input_dist=spn.DenseSPNGeneratorLayerNodes.InputDist.RAW,
                               strides=stride, dilation_rate=dilation_rate)
        conv_layer = spn.ConvProd2D(ivs, kernel_size=2, num_channels=16, grid_dim_sizes=grid_dims,
                                    strides=stride, dilation_rate=dilation_rate)

        dense_gen = spn.DenseSPNGeneratorLayerNodes(
            num_decomps=1, num_subsets=2, num_mixtures=2,
            input_dist=spn.DenseSPNGeneratorLayerNodes.InputDist.MIXTURE,
            node_type=spn.DenseSPNGeneratorLayerNodes.NodeType.LAYER)

        rnd = random.Random(x=1234)
        rnd_state = rnd.getstate()
        root_manual = dense_gen.generate(conv_manual, root_name="RootManual", rnd=rnd)
        spn.generate_weights(root_manual)
        init_manual = spn.initialize_weights(root_manual)

        rnd.setstate(rnd_state)
        root_layer = dense_gen.generate(conv_layer, root_name="RootLayer", rnd=rnd)
        spn.generate_weights(root_layer)
        init_layer = spn.initialize_weights(root_layer)

        pathgen_man = spn.MPEPath(value_inference_type=spn.InferenceType.MARGINAL, log=True)
        pathgen_man.get_mpe_path(root_manual)

        pathgen_lay = spn.MPEPath(value_inference_type=spn.InferenceType.MARGINAL, log=True)
        pathgen_lay.get_mpe_path(root_layer)

        ivs_counts_man = pathgen_man.counts[ivs]
        ivs_counts_lay = pathgen_lay.counts[ivs]

        value_man = pathgen_man.value.values[root_manual]
        value_lay = pathgen_lay.value.values[root_layer]

        random_ivs_feed = np.random.randint(-1, 2, size=num_vars * batch_size).reshape(
            (batch_size, num_vars))

        with self.test_session() as sess:
            sess.run([init_manual, init_layer])

            counts_man_out, counts_lay_out = sess.run(
                [ivs_counts_man, ivs_counts_lay], feed_dict={ivs: random_ivs_feed})

            value_man_out, value_lay_out = sess.run(
                [value_man, value_lay], feed_dict={ivs: random_ivs_feed})

        self.assertAllClose(np.exp(value_man_out), np.exp(value_lay_out))
        self.assertAllEqual(counts_man_out, counts_lay_out)

    def test_compute_mpe_path(self):
        grid_dims = [4, 4]
        input_channels = 2
        vars = spn.ContVars(num_vars=grid_dims[0] * grid_dims[1] * input_channels)
        convprod = spn.ConvProd2D(vars, num_channels=32, padding_algorithm='valid', strides=2,
                                  grid_dim_sizes=grid_dims)

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
        vars = spn.ContVars(num_vars=grid_dims[0] * grid_dims[1] * input_channels)
        convprod = spn.ConvProd2D(vars, num_channels=32, padding_algorithm='valid', strides=1,
                                  grid_dim_sizes=grid_dims, dilation_rate=2)

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
        vars = spn.ContVars(num_vars=4)
        convprod = spn.ConvProd2D(vars, num_channels=1, strides=1, kernel_size=2,
                                  pad_left=1, pad_right=1, pad_top=1, pad_bottom=1,
                                  grid_dim_sizes=grid_dims)
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
        vars = spn.ContVars(num_vars=4)
        convprod = spn.ConvProd2D(vars, num_channels=1, strides=1, kernel_size=2,
                                  pad_left=1, pad_right=1, pad_top=1, pad_bottom=1,
                                  grid_dim_sizes=grid_dims)
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
        vars = spn.IVs(num_vars=grid_dims[0] * grid_dims[1], num_vals=input_channels)
        strides = 1 if dilate else 2
        dilation_rate = 2 if dilate else 1
        convprod = spn.ConvProd2D(vars, num_channels=32, padding_algorithm='valid', strides=strides,
                                  grid_dim_sizes=grid_dims, dilation_rate=dilation_rate)
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

    def test_compute_valid(self):
        grid_dims = [16, 16]
        input_channels = 2
        vars = spn.IVs(num_vars=grid_dims[0] * grid_dims[1], num_vals=input_channels)
        convprod0 = spn.ConvProd2D(vars, num_channels=32, padding_algorithm='valid', strides=1,
                                   grid_dim_sizes=grid_dims, dilation_rate=1, name="ConvProd0")
        convprod10 = spn.ConvProd2D(convprod0, num_channels=32, padding_algorithm='valid',
                                    strides=1, dilation_rate=1, grid_dim_sizes=[15, 15],
                                    name="ConvProd00")
        self.assertFalse(convprod10.is_valid())

        convprod11 = spn.ConvProd2D(convprod0, num_channels=32, padding_algorithm='valid',
                                    strides=1, dilation_rate=2, grid_dim_sizes=[15, 15],
                                    name="ConvProd11")
        self.assertTrue(convprod11.is_valid())

        convprod_a = spn.ConvProd2D(vars, num_channels=16, padding_algorithm='valid', strides=1,
                                    dilation_rate=2, grid_dim_sizes=grid_dims, name="ConvProdA")
        convprod_a_ds = spn.ConvProd2D(convprod_a, num_channels=512, padding_algorithm='valid',
                                       strides=2, grid_dim_sizes=[14, 14],
                                       name="ConvProdADownSample")
        convprod_a_inv = spn.ConvProd2D(convprod_a, num_channels=512, padding_algorithm='valid', strides=1,
                                        dilation_rate=2, grid_dim_sizes=[14, 14],
                                        name="ConvProdAInvalid")
        self.assertTrue(convprod_a_ds.is_valid())
        self.assertFalse(convprod_a_inv.is_valid())

        convprod_b = spn.ConvProd2D(vars, num_channels=16, padding_algorithm='same', strides=1,
                                    dilation_rate=2, grid_dim_sizes=grid_dims,
                                    name="ConvProdB")
        convprod_b_ds = spn.ConvProd2D(convprod_b, num_channels=512, padding_algorithm='valid', strides=2,
                                       grid_dim_sizes=[16, 16], name="ConvProdBDownSample")
        convprod_b_inv = spn.ConvProd2D(convprod_b, num_channels=512, padding_algorithm='valid', strides=1,
                                        dilation_rate=2, grid_dim_sizes=[16, 16],
                                        name="ConvProdBInvalid")
        self.assertTrue(convprod_b_ds.is_valid())
        self.assertFalse(convprod_b_inv.is_valid())

        convprod_b_ds2 = spn.ConvProd2D(
            convprod_b_ds, num_channels=512, padding_algorithm='same', dilation_rate=2, strides=1,
            grid_dim_sizes=[8, 8], name="ConvProdBDownSample2")

        self.assertTrue(convprod_b_ds2.is_valid())

        convprod_b_ds_level2 = spn.ConvProd2D(
            convprod_b_ds2, num_channels=512, padding_algorithm='same', dilation_rate=2, strides=1,
            grid_dim_sizes=[8, 8])

        self.assertFalse(convprod_b_ds_level2.is_valid())
        convprod_b_ds_level2b = spn.ConvProd2D(
                    convprod_b_ds2, num_channels=512, padding_algorithm='same', dilation_rate=3, strides=1,
                    grid_dim_sizes=[8, 8])
        self.assertFalse(convprod_b_ds_level2b.is_valid())

        convprod_b_ds_level2c = spn.ConvProd2D(
                    convprod_b_ds2, num_channels=512, padding_algorithm='same', dilation_rate=4, strides=1,
                    grid_dim_sizes=[8, 8])
        self.assertTrue(convprod_b_ds_level2c.is_valid())

    @argsprod([spn.DenseSPNGeneratorLayerNodes.NodeType.SINGLE,
               spn.DenseSPNGeneratorLayerNodes.NodeType.BLOCK,
               spn.DenseSPNGeneratorLayerNodes.NodeType.LAYER],
              [spn.DenseSPNGeneratorLayerNodes.InputDist.RAW,
               spn.DenseSPNGeneratorLayerNodes.InputDist.MIXTURE])
    def test_compute_dense_gen_two_spatial_decomps(self, node_type, input_dist):
        input_channels = 2
        grid_dims = [8, 8]
        num_vars = grid_dims[0] * grid_dims[1]
        vars = spn.IVs(num_vars=num_vars, num_vals=input_channels)

        convert_after = False
        if input_dist == spn.DenseSPNGeneratorLayerNodes.InputDist.RAW and \
            node_type in [spn.DenseSPNGeneratorLayerNodes.NodeType.BLOCK,
                          spn.DenseSPNGeneratorLayerNodes.NodeType.LAYER]:
            node_type = spn.DenseSPNGeneratorLayerNodes.NodeType.SINGLE
            convert_after = True

        # First decomposition
        convprod_dilate0 = spn.ConvProd2D(
            vars, grid_dim_sizes=grid_dims, num_channels=16, padding_algorithm='valid', dilation_rate=2,
            strides=1, kernel_size=2)
        convprod_dilate1 = spn.ConvProd2D(
            convprod_dilate0, grid_dim_sizes=[6, 6], num_channels=512, padding_algorithm='valid',
            dilation_rate=1, strides=4, kernel_size=2)
        convsum_dilate = spn.ConvSum(convprod_dilate1, num_channels=2, grid_dim_sizes=[2, 2])

        # Second decomposition
        convprod_stride0 = spn.ConvProd2D(
            vars, grid_dim_sizes=grid_dims, num_channels=16, padding_algorithm='valid', dilation_rate=1,
            strides=2, kernel_size=2)
        convprod_stride1 = spn.ConvProd2D(
            convprod_stride0, grid_dim_sizes=[4, 4], num_channels=512, padding_algorithm='valid',
            dilation_rate=1, strides=2, kernel_size=2)
        convsum_stride = spn.ConvSum(convprod_stride1, num_channels=2, grid_dim_sizes=[2, 2])

        dense_gen = spn.DenseSPNGeneratorLayerNodes(
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

    @argsprod([spn.DenseSPNGeneratorLayerNodes.NodeType.SINGLE,
               spn.DenseSPNGeneratorLayerNodes.NodeType.BLOCK,
               spn.DenseSPNGeneratorLayerNodes.NodeType.LAYER],
              [spn.DenseSPNGeneratorLayerNodes.InputDist.RAW,
               spn.DenseSPNGeneratorLayerNodes.InputDist.MIXTURE])
    def test_compute_dense_gen_two_spatial_decomps_v2(self, node_type, input_dist):
        input_channels = 2
        grid_dims = [32, 32]
        num_vars = grid_dims[0] * grid_dims[1]
        vars = spn.IVs(num_vars=num_vars, num_vals=input_channels)

        convert_after = False
        if input_dist == spn.DenseSPNGeneratorLayerNodes.InputDist.RAW and \
                node_type in [spn.DenseSPNGeneratorLayerNodes.NodeType.BLOCK,
                              spn.DenseSPNGeneratorLayerNodes.NodeType.LAYER]:
            node_type = spn.DenseSPNGeneratorLayerNodes.NodeType.SINGLE
            convert_after = True

        # First decomposition
        convprod_dilate0 = spn.ConvProd2D(
            vars, grid_dim_sizes=grid_dims, num_channels=16, padding_algorithm='valid', dilation_rate=2,
            strides=1, kernel_size=2)
        convprod_dilate1 = spn.ConvProd2D(
            convprod_dilate0, grid_dim_sizes=[30, 30], num_channels=512, padding_algorithm='valid',
            dilation_rate=1, strides=4, kernel_size=2)
        convsum_dilate = spn.ConvSum(convprod_dilate1, num_channels=2, grid_dim_sizes=[8, 8])

        # Second decomposition
        convprod_stride0 = spn.ConvProd2D(
            vars, grid_dim_sizes=grid_dims, num_channels=16, padding_algorithm='valid', dilation_rate=1,
            strides=2, kernel_size=2)
        convprod_stride1 = spn.ConvProd2D(
            convprod_stride0, grid_dim_sizes=[16, 16], num_channels=512, padding_algorithm='valid',
            dilation_rate=1, strides=2, kernel_size=2)
        convsum_stride = spn.ConvSum(convprod_stride1, num_channels=2, grid_dim_sizes=[8, 8])

        # First decomposition level 2
        convprod_dilate0_l2 = spn.ConvProd2D(
            convsum_stride, convsum_dilate, grid_dim_sizes=[8, 8], num_channels=512,
            padding_algorithm='valid', dilation_rate=2, strides=1, kernel_size=2)
        convprod_dilate1_l2 = spn.ConvProd2D(
            convprod_dilate0_l2, grid_dim_sizes=[6, 6], num_channels=512, padding_algorithm='valid',
            dilation_rate=1, kernel_size=2, strides=4)
        convsum_dilate_l2 = spn.ConvSum(convprod_dilate1_l2, num_channels=2, grid_dim_sizes=[4, 4])
        
        # Second decomposition level 2
        convprod_stride0_l2 = spn.ConvProd2D(
            convsum_stride, convsum_dilate, grid_dim_sizes=[8, 8], num_channels=512,
            padding_algorithm='valid', dilation_rate=1, strides=2, kernel_size=2)
        convprod_stride1_l2 = spn.ConvProd2D(
            convprod_stride0_l2, grid_dim_sizes=[4, 4], num_channels=512,
            padding_algorithm='valid', dilation_rate=1, strides=2, kernel_size=2)
        convsum_stride_l2 = spn.ConvSum(convprod_stride1_l2, num_channels=2, grid_dim_sizes=[4, 4])

        dense_gen = spn.DenseSPNGeneratorLayerNodes(
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

    @argsprod([spn.DenseSPNGeneratorLayerNodes.NodeType.SINGLE,
               spn.DenseSPNGeneratorLayerNodes.NodeType.BLOCK,
               spn.DenseSPNGeneratorLayerNodes.NodeType.LAYER],
              [spn.DenseSPNGeneratorLayerNodes.InputDist.RAW,
               spn.DenseSPNGeneratorLayerNodes.InputDist.MIXTURE])
    def test_conv_spn(self, node_type, input_dist):
        input_channels = 2
        grid_dims = [32, 32]
        num_vars = grid_dims[0] * grid_dims[1]
        vars = spn.IVs(num_vars=num_vars, num_vals=input_channels)

        conv_spn_gen = ConvSPN()

        convert_after = False
        if input_dist == spn.DenseSPNGeneratorLayerNodes.InputDist.RAW and \
                node_type in [spn.DenseSPNGeneratorLayerNodes.NodeType.BLOCK,
                              spn.DenseSPNGeneratorLayerNodes.NodeType.LAYER]:
            node_type = spn.DenseSPNGeneratorLayerNodes.NodeType.SINGLE
            convert_after = True

        convsum_dilate = conv_spn_gen.add_dilate_stride(
            vars, prod_num_channels=16, sum_num_channels=2, spatial_dims=grid_dims)
        convsum_stride = conv_spn_gen.add_double_stride(
            vars, prod_num_channels=16, sum_num_channels=2, spatial_dims=grid_dims)

        convsum_dilate_level2 = conv_spn_gen.add_dilate_stride(
            convsum_dilate, convsum_stride, prod_num_channels=16, sum_num_channels=2)
        convsum_stride_level2 = conv_spn_gen.add_double_stride(
            convsum_dilate, convsum_stride, prod_num_channels=16, sum_num_channels=2)

        dense_gen = spn.DenseSPNGeneratorLayerNodes(
            num_mixtures=2, num_decomps=1, num_subsets=2, node_type=node_type,
            input_dist=input_dist)
        root = dense_gen.generate(convsum_stride_level2, convsum_dilate_level2)
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

    @argsprod([spn.DenseSPNGeneratorLayerNodes.NodeType.SINGLE,
               spn.DenseSPNGeneratorLayerNodes.NodeType.BLOCK,
               spn.DenseSPNGeneratorLayerNodes.NodeType.LAYER],
              [spn.DenseSPNGeneratorLayerNodes.InputDist.RAW,
               spn.DenseSPNGeneratorLayerNodes.InputDist.MIXTURE])
    def test_conv_spn_pad_decomps(self, node_type, input_dist):
        input_channels = 2
        grid_dims = [28, 28]
        num_vars = grid_dims[0] * grid_dims[1]
        vars = spn.IVs(num_vars=num_vars, num_vals=input_channels)

        conv_spn_gen = ConvSPN()

        convert_after = False
        if input_dist == spn.DenseSPNGeneratorLayerNodes.InputDist.RAW and \
                node_type in [spn.DenseSPNGeneratorLayerNodes.NodeType.BLOCK,
                              spn.DenseSPNGeneratorLayerNodes.NodeType.LAYER]:
            node_type = spn.DenseSPNGeneratorLayerNodes.NodeType.SINGLE
            convert_after = True

        convsum_dilate = conv_spn_gen.add_dilate_stride(
            vars, prod_num_channels=16, sum_num_channels=2, spatial_dims=grid_dims)
        convsum_stride = conv_spn_gen.add_double_stride(
            vars, prod_num_channels=16, sum_num_channels=2, spatial_dims=grid_dims)

        convsum_dilate_pad = conv_spn_gen.add_dilate_stride(
            vars, prod_num_channels=16, sum_num_channels=2, pad_all=(2, 0),
            spatial_dims=grid_dims)
        convsum_stride_pad = conv_spn_gen.add_double_stride(
            vars, prod_num_channels=16, sum_num_channels=2, pad_all=(0, 1),
            spatial_dims=grid_dims)

        dense_gen = spn.DenseSPNGeneratorLayerNodes(
            num_mixtures=2, num_decomps=1, num_subsets=2, node_type=node_type,
            input_dist=input_dist)
        root_no_pad = dense_gen.generate(convsum_dilate, convsum_stride)
        root_pad = dense_gen.generate(convsum_dilate_pad, convsum_stride_pad)
        if convert_after:
            root_no_pad = dense_gen.convert_to_layer_nodes(root_no_pad)
            root_pad = dense_gen.convert_to_layer_nodes(root_pad)

        root = spn.Sum(root_no_pad, root_pad, name="Root")

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
