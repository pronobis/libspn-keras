#!/usr/bin/env python3

# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

import unittest
import tensorflow as tf
import numpy as np
from context import libspn as spn


class TestNodesProducts(unittest.TestCase):

    def tearDown(self):
        tf.reset_default_graph()

    def test_compute_value(self):
        """Calculating value of Products"""

        def test(inputs, num_prods, feed, output):
            with self.subTest(inputs=inputs, num_prods=num_prods, feed=feed):
                n = spn.Products(*inputs, num_prods=num_prods)
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

        # Single Product node
        num_prods = 1

        # Multiple inputs, multi-element batch
        test([v1, v2],
             num_prods,
             {v1: [[0.1, 0.2, 0.3],
                   [0.4, 0.5, 0.6]],
              v2: [[0.7],
                   [0.8]]},
             [[0.1 * 0.2 * 0.3 * 0.7],
              [0.4 * 0.5 * 0.6 * 0.8]])
        test([(v1, [0, 2]), (v2, [0])],
             num_prods,
             {v1: [[0.1, 0.2, 0.3],
                   [0.4, 0.5, 0.6]],
              v2: [[0.7],
                   [0.8]]},
             [[0.1 * 0.3 * 0.7],
              [0.4 * 0.6 * 0.8]])

        # Single input with 1 value, multi-element batch
        test([v2],
             num_prods,
             {v2: [[0.1],
                   [0.2]]},
             [[0.1],
              [0.2]])
        test([(v1, [1])],
             num_prods,
             {v1: [[0.01, 0.1, 0.03],
                   [0.02, 0.2, 0.04]]},
             [[0.1],
              [0.2]])

        # Multiple inputs, single-element batch
        test([v1, v2],
             num_prods,
             {v1: [[0.1, 0.2, 0.3]],
              v2: [[0.7]]},
             [[0.1 * 0.2 * 0.3 * 0.7]])
        test([(v1, [0, 2]), (v2, [0])],
             num_prods,
             {v1: [[0.1, 0.2, 0.3]],
              v2: [[0.7]]},
             [[0.1 * 0.3 * 0.7]])

        # Single input with 1 value, single-element batch
        test([v2],
             num_prods,
             {v2: [[0.1]]},
             [[0.1]])
        test([(v1, [1])],
             num_prods,
             {v1: [[0.01, 0.1, 0.03]]},
             [[0.1]])

        # Multiple Product nodes
        num_prods = 2

        # Multiple inputs, multi-element batch
        test([v1, v2, v1, v2],
             num_prods,
             {v1: [[0.1, 0.2, 0.3],
                   [0.4, 0.5, 0.6]],
              v2: [[0.7],
                   [0.8]]},
             [[(0.1 * 0.2 * 0.3 * 0.7), (0.1 * 0.2 * 0.3 * 0.7)],
              [(0.4 * 0.5 * 0.6 * 0.8), (0.4 * 0.5 * 0.6 * 0.8)]])
        test([(v1, [0, 2]), (v2, [0]), (v1, [0, 2]), (v2, [0])],
             num_prods,
             {v1: [[0.1, 0.2, 0.3],
                   [0.4, 0.5, 0.6]],
              v2: [[0.7],
                   [0.8]]},
             [[(0.1 * 0.3 * 0.7), (0.1 * 0.3 * 0.7)],
              [(0.4 * 0.6 * 0.8), (0.4 * 0.6 * 0.8)]])

        # Single input with 1 value, multi-element batch
        test([v2, v2],
             num_prods,
             {v2: [[0.1],
                   [0.2]]},
             [[0.1, 0.1],
              [0.2, 0.2]])
        test([(v1, [1]), (v1, [1])],
             num_prods,
             {v1: [[0.01, 0.1, 0.03],
                   [0.02, 0.2, 0.04]]},
             [[0.1, 0.1],
              [0.2, 0.2]])

        # Multiple inputs, single-element batch
        test([v1, v2, v1, v2],
             num_prods,
             {v1: [[0.1, 0.2, 0.3]],
              v2: [[0.7]]},
             [[(0.1 * 0.2 * 0.3 * 0.7), (0.1 * 0.2 * 0.3 * 0.7)]])
        test([(v1, [0, 2]), (v2, [0]), (v1, [0, 2]), (v2, [0])],
             num_prods,
             {v1: [[0.1, 0.2, 0.3]],
              v2: [[0.7]]},
             [[(0.1 * 0.3 * 0.7), (0.1 * 0.3 * 0.7)]])

        # Single input with 1 value, single-element batch
        test([v2, v2],
             num_prods,
             {v2: [[0.1]]},
             [[0.1, 0.1]])
        test([(v1, [1]), (v1, [1])],
             num_prods,
             {v1: [[0.01, 0.1, 0.03]]},
             [[0.1, 0.1]])

    def test_compute_mpe_path(self):
        v12 = spn.IVs(num_vars=2, num_vals=4)
        v34 = spn.ContVars(num_vars=2)
        v5 = spn.ContVars(num_vars=1)
        p = spn.Products((v12, [0, 5]), v5, v34, (v12, [3]), num_prods=2)
        counts = tf.placeholder(tf.float32, shape=(None, 2))
        op = p._compute_mpe_path(tf.identity(counts),
                                 v12.get_value(),
                                 v5.get_value(),
                                 v34.get_value(),
                                 v12.get_value())
        feed = [[11, 12],
                [21, 22],
                [31, 32]]
        with tf.Session() as sess:
            out = sess.run(op, feed_dict={counts: feed})
        np.testing.assert_array_almost_equal(
            out[0], np.array([[11., 0., 0., 0., 0., 11., 0., 0.],
                              [21., 0., 0., 0., 0., 21., 0., 0.],
                              [31., 0., 0., 0., 0., 31., 0., 0.]],
                             dtype=np.float32))
        np.testing.assert_array_almost_equal(
            out[1], np.array([[11.],
                              [21.],
                              [31.]],
                             dtype=np.float32))
        np.testing.assert_array_almost_equal(
            out[2], np.array([[12., 12.],
                              [22., 22.],
                              [32., 32.]],
                             dtype=np.float32))

        np.testing.assert_array_almost_equal(
            out[3], np.array([[0., 0., 0., 12., 0., 0., 0., 0.],
                              [0., 0., 0., 22., 0., 0., 0., 0.],
                              [0., 0., 0., 32., 0., 0., 0., 0.]],
                             dtype=np.float32))


if __name__ == '__main__':
    unittest.main()
