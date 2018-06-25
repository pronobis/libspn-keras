from libspn.tests.test import argsprod
import tensorflow as tf
import numpy as np
import libspn as spn


class TestConvProd(tf.test.TestCase):

    def test_generate_sparse_connections(self):
        grid_dims = [4, 4]
        input_channels = 2
        vars = spn.ContVars(num_vars=grid_dims[0] * grid_dims[1] * input_channels)

        convprod = spn.ConvProd(vars, num_channels=32, strides=2, padding='valid',
                                grid_dim_sizes=grid_dims)

        connections = convprod.generate_sparse_connections(32)
        connection_tuples = [tuple(c) for c in
                             connections.reshape((-1, convprod._num_channels)).transpose()]
        self.assertEqual(len(set(connection_tuples)), len(connection_tuples))

    def test_compute_log_value(self):
        grid_dims = [4, 4]
        input_channels = 2
        vars = spn.ContVars(num_vars=grid_dims[0] * grid_dims[1] * input_channels)
        convprod = spn.ConvProd(vars, num_channels=32, padding='valid', strides=2,
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
            [[5, 5], [12, 10]],      # 14
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
        convprod = spn.ConvProd(vars, num_channels=32, padding='valid', strides=2,
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

    def test_compute_scope(self):
        grid_dims = [4, 4]
        input_channels = 2
        vars = spn.IVs(num_vars=grid_dims[0] * grid_dims[1], num_vals=input_channels)
        convprod = spn.ConvProd(vars, num_channels=32, padding='valid', strides=2,
                                grid_dim_sizes=grid_dims)
        conv_prod_scope = convprod.get_scope()

        singular_scopes = np.asarray(
            [spn.Scope(vars, i) for i in range(grid_dims[0] * grid_dims[1])]).reshape((4, 4))

        scope_truth = [
            spn.Scope.merge_scopes(singular_scopes[0:2, 0:2].ravel().tolist()),
            spn.Scope.merge_scopes(singular_scopes[0:2, 2:4].ravel().tolist()),
            spn.Scope.merge_scopes(singular_scopes[2:4, 0:2].ravel().tolist()),
            spn.Scope.merge_scopes(singular_scopes[2:4, 2:4].ravel().tolist())
        ]
        scope_truth = np.repeat(scope_truth, 16).tolist()

        self.assertAllEqual(conv_prod_scope, scope_truth)