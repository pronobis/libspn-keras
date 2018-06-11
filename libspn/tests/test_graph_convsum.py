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
        
        self.assertAllClose(out, [[0.2, 0.6, 0.8, 0.4, 0.8, 0.4, 0.2, 0.6],
                                  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.6]])

    def test_compute_mpe_path(self):
        ivs = spn.IVs(num_vals=2, num_vars=2 * 2)
        values = [[0, 1, 1, 0],
                  [-1, -1, -1, 0]]
        weights = spn.Weights(init_value=[[0.2, 0.8],
                                          [0.6, 0.4]], num_sums=2, num_weights=2)
        s = ConvSum(ivs, grid_dim_sizes=[2, 2], num_channels=2, weights=weights)

        val = s.get_value(inference_type=spn.InferenceType.MARGINAL)
        w_tensor = val.values[weights]
        ivs_tensor = val.values[ivs]

        counts = tf.reshape(tf.range(16), (2, 8))
        w_counts, _, ivs_counts = s._compute_mpe_path()

        with self.test_session() as sess:
            sess.run(weights.initialize())
            out = sess.run(val, {ivs: values})

        self.assertAllClose(out, [[0.2, 0.6, 0.8, 0.4, 0.8, 0.4, 0.2, 0.6],
                                  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.6]])