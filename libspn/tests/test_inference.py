#!/usr/bin/env python3

# ------------------------------------------------------------------------
# Copyright (C) 2016 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

import unittest
import tensorflow as tf
import numpy as np
from context import libspn as spn


class TestInference(unittest.TestCase):

    def tearDown(self):
        tf.reset_default_graph()

    @classmethod
    def setUpClass(cls):
        pass

    def test_marginal_value(self):
        """Calculation of SPN marginal value"""
        # Generate SPN
        gen = spn.TestSPNGenerator(
            spn_type=spn.TestSPNGenerator.Type.POON11_NAIVE_MIXTURE)
        ivs, root = gen.generate()
        # Set default inference type for each node
        root.set_inference_types(spn.InferenceType.MARGINAL)
        # Get values
        init = spn.initialize_weights(root)
        val_marginal = root.get_value(
            inference_type=spn.InferenceType.MARGINAL)
        val_default = root.get_value()
        val_log_marginal = root.get_log_value(
            inference_type=spn.InferenceType.MARGINAL)
        val_log_default = root.get_log_value()
        with tf.Session() as sess:
            init.run()
            out_default = sess.run(val_default, feed_dict={ivs: gen.feed})
            out_marginal = sess.run(val_marginal, feed_dict={ivs: gen.feed})
            out_log_default = sess.run(tf.exp(val_log_default),
                                       feed_dict={ivs: gen.feed})
            out_log_marginal = sess.run(tf.exp(val_log_marginal),
                                        feed_dict={ivs: gen.feed})
        # Check if values sum to 1
        # WARNING: Below does not pass test for places=7 with float32 dtype
        self.assertAlmostEqual(out_default[np.all(gen.feed >= 0, axis=1), :].sum(),
                               1.0, places=6)
        self.assertAlmostEqual(out_marginal[np.all(gen.feed >= 0, axis=1), :].sum(),
                               1.0, places=6)
        self.assertAlmostEqual(out_log_default[np.all(gen.feed >= 0, axis=1), :].sum(),
                               1.0, places=6)
        self.assertAlmostEqual(out_log_marginal[np.all(gen.feed >= 0, axis=1), :].sum(),
                               1.0, places=6)
        # Check joint probabilities
        np.testing.assert_array_almost_equal(out_default, gen.true_values)
        np.testing.assert_array_almost_equal(out_marginal, gen.true_values)
        np.testing.assert_array_almost_equal(out_log_default, gen.true_values)
        np.testing.assert_array_almost_equal(out_log_marginal, gen.true_values)

    def test_mpe_value(self):
        """Calculation of SPN MPE value"""
        # Generate SPN
        gen = spn.TestSPNGenerator(
            spn_type=spn.TestSPNGenerator.Type.POON11_NAIVE_MIXTURE)
        ivs, root = gen.generate()
        # Set default inference type for each node
        root.set_inference_types(spn.InferenceType.MPE)
        # Get values
        init = spn.initialize_weights(root)
        val_mpe = root.get_value(inference_type=spn.InferenceType.MPE)
        val_default = root.get_value()
        val_log_mpe = root.get_log_value(inference_type=spn.InferenceType.MPE)
        val_log_default = root.get_log_value()
        with tf.Session() as sess:
            init.run()
            out_default = sess.run(val_default, feed_dict={ivs: gen.feed})
            out_mpe = sess.run(val_mpe, feed_dict={ivs: gen.feed})
            out_log_default = sess.run(tf.exp(val_log_default),
                                       feed_dict={ivs: gen.feed})
            out_log_mpe = sess.run(tf.exp(val_log_mpe),
                                   feed_dict={ivs: gen.feed})
        # Check joint probabilities
        np.testing.assert_array_almost_equal(out_default, gen.true_mpe_values)
        np.testing.assert_array_almost_equal(out_mpe, gen.true_mpe_values)
        np.testing.assert_array_almost_equal(out_log_default, gen.true_mpe_values)
        np.testing.assert_array_almost_equal(out_log_mpe, gen.true_mpe_values)

    def test_mixed_value(self):
        """Calculation of a mixed MPE/marginal value"""
        # Generate SPN
        gen = spn.TestSPNGenerator(
            spn_type=spn.TestSPNGenerator.Type.POON11_NAIVE_MIXTURE)
        ivs, root = gen.generate()
        # Set default inference type for each node
        root.set_inference_types(spn.InferenceType.MARGINAL)
        root.inference_type = spn.InferenceType.MPE
        # Get values
        init = spn.initialize_weights(root)
        val_marginal = root.get_value(inference_type=spn.InferenceType.MARGINAL)
        val_mpe = root.get_value(inference_type=spn.InferenceType.MPE)
        val_default = root.get_value()
        val_log_marginal = root.get_log_value(inference_type=spn.InferenceType.MARGINAL)
        val_log_mpe = root.get_log_value(inference_type=spn.InferenceType.MPE)
        val_log_default = root.get_log_value()
        with tf.Session() as sess:
            init.run()
            out_default = sess.run(val_default, feed_dict={ivs: gen.feed})
            out_marginal = sess.run(val_marginal, feed_dict={ivs: gen.feed})
            out_mpe = sess.run(val_mpe, feed_dict={ivs: gen.feed})
            out_log_default = sess.run(tf.exp(val_log_default),
                                       feed_dict={ivs: gen.feed})
            out_log_marginal = sess.run(tf.exp(val_log_marginal),
                                        feed_dict={ivs: gen.feed})
            out_log_mpe = sess.run(tf.exp(val_log_mpe),
                                   feed_dict={ivs: gen.feed})
        # Check joint probabilities
        true_default = [[0.5],
                        [0.35],
                        [0.15],
                        [0.2],
                        [0.14],
                        [0.06],
                        [0.3],
                        [0.216],
                        [0.09]]
        np.testing.assert_array_almost_equal(out_default, true_default)
        np.testing.assert_array_almost_equal(out_marginal, gen.true_values)
        np.testing.assert_array_almost_equal(out_mpe, gen.true_mpe_values)
        np.testing.assert_array_almost_equal(out_log_default, true_default)
        np.testing.assert_array_almost_equal(out_log_marginal, gen.true_values)
        np.testing.assert_array_almost_equal(out_log_mpe, gen.true_mpe_values)

    def test_mpe_path(self):
        # Generate SPN
        gen = spn.TestSPNGenerator(
            spn_type=spn.TestSPNGenerator.Type.POON11_NAIVE_MIXTURE)
        ivs, root = gen.generate()
        # Add ops
        init = spn.initialize_weights(root)
        mpe_path_gen = spn.MPEPath(value_inference_type=spn.InferenceType.MPE,
                                   log=False)
        mpe_path_gen_log = spn.MPEPath(value_inference_type=spn.InferenceType.MPE,
                                       log=True)
        mpe_path_gen.get_mpe_path(root)
        mpe_path_gen_log.get_mpe_path(root)
        # Run
        with tf.Session() as sess:
            init.run()
            out = sess.run(mpe_path_gen.counts[ivs],
                           feed_dict={ivs: gen.feed})
            out_log = sess.run(mpe_path_gen_log.counts[ivs],
                               feed_dict={ivs: gen.feed})

        true_ivs_counts = np.array([[0., 1., 1., 0.],
                                    [0., 1., 1., 0.],
                                    [0., 1., 0., 1.],
                                    [1., 0., 1., 0.],
                                    [1., 0., 1., 0.],
                                    [1., 0., 0., 1.],
                                    [0., 1., 1., 0.],
                                    [0., 1., 1., 0.],
                                    [0., 1., 0., 1.]],
                                   dtype=spn.conf.dtype.as_numpy_dtype)

        np.testing.assert_array_equal(out, true_ivs_counts)
        np.testing.assert_array_equal(out_log, true_ivs_counts)

    def test_mpe_state(self):
        # Generate SPN
        gen = spn.TestSPNGenerator(
            spn_type=spn.TestSPNGenerator.Type.POON11_NAIVE_MIXTURE)
        ivs, root = gen.generate()
        # Add ops
        init = spn.initialize_weights(root)
        mpe_state_gen = spn.MPEState(value_inference_type=spn.InferenceType.MPE,
                                     log=False)
        mpe_state_gen_log = spn.MPEState(value_inference_type=spn.InferenceType.MPE,
                                         log=True)
        ivs_state, = mpe_state_gen.get_state(root, ivs)
        ivs_state_log, = mpe_state_gen_log.get_state(root, ivs)
        # Run
        with tf.Session() as sess:
            init.run()
            out = sess.run(ivs_state, feed_dict={ivs: [[-1, -1]]})
            out_log = sess.run(ivs_state_log, feed_dict={ivs: [[-1, -1]]})

        # For now we only compare the actual MPE state for input IVs -1
        np.testing.assert_array_equal(out.ravel(), gen.true_mpe_state)
        np.testing.assert_array_equal(out_log.ravel(), gen.true_mpe_state)


if __name__ == '__main__':
    unittest.main()
