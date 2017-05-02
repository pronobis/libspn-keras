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


class TestNodesProduct(unittest.TestCase):

    def tearDown(self):
        tf.reset_default_graph()

    def test_compute_value(self):
        """Calculating value of Product"""

        def test(inputs, feed, output):
            with self.subTest(inputs=inputs, feed=feed):
                n = spn.Product(*inputs)
                op = n.get_value(spn.InferenceType.MARGINAL)
                op_log = n.get_log_value(spn.InferenceType.MARGINAL)
                op_mpe = n.get_value(spn.InferenceType.MPE)
                op_log_mpe = n.get_log_value(spn.InferenceType.MPE)
                with tf.Session() as sess:
                    out = sess.run(op, feed_dict=feed)
                    out_log = sess.run(tf.exp(op_log), feed_dict=feed)
                    out_mpe = sess.run(op_mpe, feed_dict=feed)
                    out_log_mpe = sess.run(tf.exp(op_log_mpe), feed_dict=feed)
                np.testing.assert_array_almost_equal(
                    out,
                    np.array(output, dtype=spn.conf.dtype.as_numpy_dtype()))
                np.testing.assert_array_almost_equal(
                    out_log,
                    np.array(output, dtype=spn.conf.dtype.as_numpy_dtype()))
                np.testing.assert_array_almost_equal(
                    out_mpe,
                    np.array(output, dtype=spn.conf.dtype.as_numpy_dtype()))
                np.testing.assert_array_almost_equal(
                    out_log_mpe,
                    np.array(output, dtype=spn.conf.dtype.as_numpy_dtype()))

        # Create inputs
        v1 = spn.ContVars(num_vars=3)
        v2 = spn.ContVars(num_vars=1)

        # Multiple inputs, multi-element batch
        test([v1, v2],
             {v1: [[0.1, 0.2, 0.3],
                   [0.4, 0.5, 0.6]],
              v2: [[0.7],
                   [0.8]]},
             [[0.1 * 0.2 * 0.3 * 0.7],
              [0.4 * 0.5 * 0.6 * 0.8]])
        test([(v1, [0, 2]), (v2, [0])],
             {v1: [[0.1, 0.2, 0.3],
                   [0.4, 0.5, 0.6]],
              v2: [[0.7],
                   [0.8]]},
             [[0.1 * 0.3 * 0.7],
              [0.4 * 0.6 * 0.8]])

        # Single input with 1 value, multi-element batch
        test([v2],
             {v2: [[0.1],
                   [0.2]]},
             [[0.1],
              [0.2]])
        test([(v1, [1])],
             {v1: [[0.01, 0.1, 0.03],
                   [0.02, 0.2, 0.04]]},
             [[0.1],
              [0.2]])

        # Multiple inputs, single-element batch
        test([v1, v2],
             {v1: [[0.1, 0.2, 0.3]],
              v2: [[0.7]]},
             [[0.1 * 0.2 * 0.3 * 0.7]])
        test([(v1, [0, 2]), (v2, [0])],
             {v1: [[0.1, 0.2, 0.3]],
              v2: [[0.7]]},
             [[0.1 * 0.3 * 0.7]])

        # Single input with 1 value, single-element batch
        test([v2],
             {v2: [[0.1]]},
             [[0.1]])
        test([(v1, [1])],
             {v1: [[0.01, 0.1, 0.03]]},
             [[0.1]])

    def test_compute_valid(self):
        """Calculating validity of Product"""
        v12 = spn.IVs(num_vars=2, num_vals=4)
        v34 = spn.ContVars(num_vars=2)
        p1 = spn.Product((v12, [0, 5]))
        p2 = spn.Product((v12, [0, 3]))
        p3 = spn.Product((v12, [0, 5]), v34)
        p4 = spn.Product((v12, [0, 3]), v34)
        p5 = spn.Product((v12, [0, 5]), v34, (v12, 2))
        self.assertTrue(p1.is_valid())
        self.assertFalse(p2.is_valid())
        self.assertTrue(p3.is_valid())
        self.assertFalse(p4.is_valid())
        self.assertFalse(p5.is_valid())

    def test_compute_mpe_path(self):
        v12 = spn.IVs(num_vars=2, num_vals=4)
        v34 = spn.ContVars(num_vars=2)
        v5 = spn.ContVars(num_vars=1)
        p = spn.Product((v12, [0, 5]), v34, (v12, [3]), v5)
        counts = tf.placeholder(tf.float32, shape=(None, 1))
        op = p._compute_mpe_path(tf.identity(counts),
                                 v12.get_value(),
                                 v34.get_value(),
                                 v12.get_value(),
                                 v5.get_value())
        feed = [[0],
                [1],
                [2]]
        with tf.Session() as sess:
            out = sess.run(op, feed_dict={counts: feed})
        np.testing.assert_array_almost_equal(
            out[0], np.array([[0., 0., 0., 0., 0., 0., 0., 0.],
                              [1., 0., 0., 0., 0., 1., 0., 0.],
                              [2., 0., 0., 0., 0., 2., 0., 0.]],
                             dtype=np.float32))
        np.testing.assert_array_almost_equal(
            out[1], np.array([[0., 0.],
                              [1., 1.],
                              [2., 2.]],
                             dtype=np.float32))
        np.testing.assert_array_almost_equal(
            out[2], np.array([[0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 1., 0., 0., 0., 0.],
                              [0., 0., 0., 2., 0., 0., 0., 0.]],
                             dtype=np.float32))
        np.testing.assert_array_almost_equal(
            out[3], np.array([[0.],
                              [1.],
                              [2.]],
                             dtype=np.float32))


if __name__ == '__main__':
    unittest.main()
