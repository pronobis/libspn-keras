import tensorflow as tf
from libspn.tests.test import argsprod
import libspn as spn
from libspn.graph.convsum import ConvSum
import numpy as np


class TestBaseSum(tf.test.TestCase):

    def test_compute_value(self):
        ivs = spn.IVs(num_vals=2, num_vars=2 * 2)
        values = [[0, 1, 1, 0],
                  [-1, -1, -1, 0]]
        weights = spn.Weights(init_value=[[0.2, 0.8],
                                          [0.6, 0.4]], num_sums=2, num_weights=2)
        s = ConvSum(ivs, grid_dim_sizes=[2, 2], num_channels=2, weights=weights)
        
        val = s.get_value(inference_type=spn.InferenceType.MARGINAL)
        
        with self.test_session() as sess:
            sess.run(weights.initialize())
            out = sess.run(val, {ivs: values})

                                  # 0    0 |  1    1 |  1    1  | 0   0
        self.assertAllClose(out, [[0.2, 0.6, 0.8, 0.4, 0.8, 0.4, 0.2, 0.6],
                                  # 1   0  | 1     0 | 1     0  | 0   0
                                  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.6]])

    def test_compute_mpe_path(self):
        ivs = spn.IVs(num_vals=2, num_vars=2 * 2)
        values = [[0, 1, 1, 0],
                  [-1, -1, -1, 0]]
        weights = spn.Weights(init_value=[[0.2, 0.8],
                                          [0.6, 0.4]], num_sums=2, num_weights=2)
        s = ConvSum(ivs, grid_dim_sizes=[2, 2], num_channels=2, weights=weights)

        val = spn.Value(inference_type=spn.InferenceType.MARGINAL)
        val.get_value(s)
        w_tensor = val.values[weights]
        value_tensor = val.values[ivs]

        counts = tf.reshape(tf.range(10, 26), (2, 8))
        w_counts, _, ivs_counts = s._compute_mpe_path(
            counts, w_tensor, None, value_tensor)

        with self.test_session() as sess:
            sess.run(weights.initialize())
            w_counts_out, ivs_counts_out = sess.run(
                [w_counts, ivs_counts], {ivs: values})

        counts_truth = [
            [[10 + 16, 12 + 14],
             [11 + 17, 13 + 15]],
            [[24, 18 + 20 + 22],
             [19 + 21 + 23 + 25, 0]]
        ]

        ivs_counts_truth = \
            [[10 + 11, 0, 0, 12 + 13, 0, 14 + 15, 16 + 17, 0],
             [19, 18, 21, 20, 23, 22, 24 + 25, 0]]

        self.assertAllEqual(w_counts_out, counts_truth)
        self.assertAllEqual(ivs_counts_truth, ivs_counts_out)

    def test_compute_scope(self):
        ivs = spn.IVs(num_vals=2, num_vars=2 * 2)
        weights = spn.Weights(init_value=[[0.2, 0.8],
                                          [0.6, 0.4]], num_sums=2, num_weights=2)
        s = ConvSum(ivs, grid_dim_sizes=[2, 2], num_channels=2, weights=weights)

        scope = s._compute_scope(None, None, ivs._compute_scope())

        target_scope = [spn.Scope(ivs, 0)] * 2 + \
                       [spn.Scope(ivs, 1)] * 2 + \
                       [spn.Scope(ivs, 2)] * 2 + \
                       [spn.Scope(ivs, 3)] * 2
        self.assertAllEqual(scope, target_scope)