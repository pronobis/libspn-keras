from libspn.graph.basesum import BaseSum
from libspn.tests.test import argsprod
import tensorflow as tf
import numpy as np
import libspn as spn
from libspn.log import get_logger
from libspn.generation.spatial import ConvSPN

logger = get_logger()

class TestConvProd(tf.test.TestCase):

    def test_generate_sparse_connections(self):
        grid_dims = [4, 4]
        input_channels = 2
        vars = spn.ContVars(num_vars=grid_dims[0] * grid_dims[1] * input_channels)

        convprod = spn.ConvProd2D(vars, num_channels=32, strides=2, padding_algorithm='valid',
                                  grid_dim_sizes=grid_dims)

        connections = convprod.generate_sparse_connections(32)
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
                                   grid_dim_sizes=grid_dims, dilation_rate=1)
        convprod10 = spn.ConvProd2D(convprod0, num_channels=32, padding_algorithm='valid', strides=1,
                                    dilation_rate=1, grid_dim_sizes=[15, 15])
        self.assertFalse(convprod10.is_valid())

        convprod11 = spn.ConvProd2D(convprod0, num_channels=32, padding_algorithm='valid', strides=1,
                                    dilation_rate=2, grid_dim_sizes=[15, 15])
        self.assertTrue(convprod11.is_valid())

        convprod_a = spn.ConvProd2D(vars, num_channels=16, padding_algorithm='valid', strides=1,
                                    dilation_rate=2, grid_dim_sizes=grid_dims, name="ConvProdA")
        convprod_a_ds = spn.ConvProd2D(convprod_a, num_channels=512, padding_algorithm='valid', strides=2,
                                       grid_dim_sizes=[14, 14], name="ConvProdADownSample")
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
        grid_dims = [16, 16]
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
