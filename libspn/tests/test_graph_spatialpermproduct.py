import libspn.graph.spatialpermproducts
from libspn.tests.test import argsprod
import tensorflow as tf
import numpy as np
import libspn as spn
from libspn.log import get_logger
import itertools

from libspn.utils.math import pow2_combinations

logger = get_logger()


class TestConvProd(tf.test.TestCase):

    def test_generate_permutation_matrix_2x2(self):
        iv0 = spn.IndicatorLeaf(num_vars=4, num_vals=2)
        iv1 = spn.IndicatorLeaf(num_vars=4, num_vals=2)
        spatial_dims = [2, 2]

        spp = libspn.graph.spatialpermproducts.SpatialPermProducts(iv0, iv1, grid_dim_sizes=spatial_dims)
        permutation_mat, _ = spp._generate_permutation_matrix()
        permutation_mat_target = [
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
        ]
        self.assertAllClose(permutation_mat, permutation_mat_target)

    def test_compute_value_simple(self):
        iv0 = spn.IndicatorLeaf(num_vars=4, num_vals=2)
        iv1 = spn.IndicatorLeaf(num_vars=4, num_vals=2)
        batch_size = 7
        spatial_dims = [2, 2]

        spp = libspn.graph.spatialpermproducts.SpatialPermProducts(iv0, iv1, grid_dim_sizes=spatial_dims)

        indices = [[0, 0], [1, 0], [0, 1], [1, 1]]

        ivs_data = [np.random.randint(0, 2, size=batch_size * 4).reshape(batch_size, 4)
                    for _ in range(2)]
        feed_dict = {iv0: ivs_data[0], iv1: ivs_data[1]}

        target_out = []
        ivs0_data, ivs1_data = ivs_data
        ivs0_data_onehot = np.eye(2)[ivs0_data].reshape(batch_size, 2, 2, 2)
        ivs1_data_onehot = np.eye(2)[ivs1_data].reshape(batch_size, 2, 2, 2)

        for iv0_ind, iv1_ind in indices:
            target_out.append(ivs0_data_onehot[..., iv0_ind] * ivs1_data_onehot[..., iv1_ind])
        target_out = np.stack(target_out, axis=-1).reshape(batch_size, 16)

        logval = spp.get_log_value()

        with self.test_session() as sess:
            logval_out = sess.run(logval, feed_dict=feed_dict)

        self.assertAllClose(logval_out, np.log(target_out))

    @argsprod([2], [2], [1, 2, 3])
    def test_numerical_validity(self, rows, cols, num_inputs):
        num_values = 2
        num_vars = rows * cols
        ivs = [spn.IndicatorLeaf(num_vars=num_vars, num_vals=num_values) for _ in range(num_inputs)]
        spp = libspn.graph.spatialpermproducts.SpatialPermProducts(*ivs, grid_dim_sizes=[rows, cols])

        local_sum = spn.LocalSum(spp, num_channels=1, grid_dim_sizes=[rows, cols])
        if rows == cols == 1:
            root = local_sum
        else:
            root = spn.Product(local_sum)

        combinations = pow2_combinations(num_vars)
        ivs_feed = {ivs[0]: combinations}
        for iv in ivs[1:]:
            ivs_feed[iv] = np.ones_like(combinations) * -1

        spn.generate_weights(root, init_value=spn.ValueType.RANDOM_UNIFORM(0, 1))
        init = spn.initialize_weights(root)

        logsum = tf.reduce_logsumexp(root.get_log_value())
        with self.test_session() as sess:
            sess.run(init)
            logsum_out = sess.run(logsum, ivs_feed)

        self.assertAllClose(logsum_out, 0.)

    def test_compute_mpe_path_simple(self):
        iv0 = spn.IndicatorLeaf(num_vars=4, num_vals=2)
        iv1 = spn.IndicatorLeaf(num_vars=4, num_vals=2)
        batch_size = 7
        spatial_dims = [2, 2]

        spp = libspn.graph.spatialpermproducts.SpatialPermProducts(iv0, iv1, grid_dim_sizes=spatial_dims)
        concat_node = spn.Concat(spp)

        indices = [[0, 0], [1, 0], [0, 1], [1, 1]]

        counts = np.arange(batch_size * 16).reshape((batch_size, 16))
        counts_spatial = counts.reshape(batch_size, 2, 2, 4)

        input_shape = (batch_size, 2, 2, 2)
        target_iv0, target_iv1 = np.zeros(input_shape), np.zeros(input_shape)
        for i, (iv0_ind, iv1_ind) in enumerate(indices):
            target_iv0[..., iv0_ind] += counts_spatial[..., i]
            target_iv1[..., iv1_ind] += counts_spatial[..., i]

        path_gen = spn.MPEPath()
        path_gen.get_mpe_path(concat_node)
        iv0_counts, iv1_counts = path_gen.counts[iv0], path_gen.counts[iv1]
        iv0_counts = tf.reshape(iv0_counts, input_shape)
        iv1_counts = tf.reshape(iv1_counts, input_shape)

        with self.test_session() as sess:
            iv0_counts_out, iv1_counts_out = sess.run(
                [iv0_counts, iv1_counts], feed_dict={path_gen.counts[concat_node]: counts})

        self.assertAllClose(iv0_counts_out, target_iv0)
        self.assertAllClose(iv1_counts_out, target_iv1)

    def test_compute_scope(self):
        iv0 = spn.IndicatorLeaf(num_vars=4, num_vals=2)
        iv1 = spn.IndicatorLeaf(num_vars=4, num_vals=2)
        spatial_dims = [2, 2]

        scopes_iv0 = iv0.get_scope()[::2]
        scopes_iv1 = iv1.get_scope()[::2]
        out_scopes = [spn.Scope.merge_scopes([sc0, sc1])
                      for sc0, sc1 in zip(scopes_iv0, scopes_iv1)]
        target = np.repeat(out_scopes, repeats=4).tolist()

        spp = libspn.graph.spatialpermproducts.SpatialPermProducts(iv0, iv1, grid_dim_sizes=spatial_dims)

        self.assertAllEqual(spp.get_scope(), target)

    def test_compute_valid(self):
        iv0 = spn.IndicatorLeaf(num_vars=4, num_vals=2)
        iv1 = spn.IndicatorLeaf(num_vars=4, num_vals=2)
        spatial_dims = [2, 2]
        spp = libspn.graph.spatialpermproducts.SpatialPermProducts(iv0, iv1, grid_dim_sizes=spatial_dims)
        ls = spn.LocalSum(spp, num_channels=1, grid_dim_sizes=spatial_dims)
        root = spn.Product(ls)
        self.assertTrue(root.is_valid())

        iv0 = spn.IndicatorLeaf(num_vars=4, num_vals=2)
        iv1 = spn.IndicatorLeaf(num_vars=2, num_vals=4)
        spatial_dims = [2, 2]
        spp = libspn.graph.spatialpermproducts.SpatialPermProducts(iv0, iv1, grid_dim_sizes=spatial_dims)
        ls = spn.LocalSum(spp, num_channels=1, grid_dim_sizes=spatial_dims)
        root = spn.Product(ls)
        self.assertFalse(root.is_valid())

        iv0 = spn.IndicatorLeaf(num_vars=4, num_vals=2)
        spp = libspn.graph.spatialpermproducts.SpatialPermProducts(iv0, iv0, grid_dim_sizes=spatial_dims)
        self.assertFalse(spp.is_valid())

