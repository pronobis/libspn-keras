#!/usr/bin/env python3

# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

from context import libspn as spn
from test import TestCase
import tensorflow as tf
import numpy as np


class TestGraphSum(TestCase):

    def test_compute_marginal_value(self):
        """Calculating marginal value of Sum"""

        def test(values, ivs, weights, feed, output):
            with self.subTest(values=values, ivs=ivs, weights=weights,
                              feed=feed):
                n = spn.Sum(*values, ivs=ivs)
                n.generate_weights(weights)
                op = n.get_value(spn.InferenceType.MARGINAL)
                op_log = n.get_log_value(spn.InferenceType.MARGINAL)
                with tf.Session() as sess:
                    spn.initialize_weights(n).run()
                    out = sess.run(op, feed_dict=feed)
                    out_log = sess.run(tf.exp(op_log), feed_dict=feed)
                np.testing.assert_array_almost_equal(
                    out,
                    np.array(output, dtype=spn.conf.dtype.as_numpy_dtype()))
                np.testing.assert_array_almost_equal(
                    out_log,
                    np.array(output, dtype=spn.conf.dtype.as_numpy_dtype()))

        # Create inputs
        v1 = spn.ContVars(num_vars=3, name="ContVars1")
        v2 = spn.ContVars(num_vars=1, name="ContVars2")
        ivs = spn.IVs(num_vars=1, num_vals=4)

        # Multiple inputs, multi-element batch
        test([v1, v2],
             None,
             [0.1, 0.2, 0.4, 0.3],
             {v1: [[0.1, 0.2, 0.3],
                   [0.4, 0.5, 0.6]],
              v2: [[0.7],
                   [0.8]]},
             [[0.1 * 0.1 + 0.2 * 0.2 + 0.3 * 0.4 + 0.7 * 0.3],
              [0.4 * 0.1 + 0.5 * 0.2 + 0.6 * 0.4 + 0.8 * 0.3]])
        test([(v1, [0, 2]), (v2, [0])],
             None,
             [0.1, 0.5, 0.4],
             {v1: [[0.1, 0.2, 0.3],
                   [0.4, 0.5, 0.6]],
              v2: [[0.7],
                   [0.8]]},
             [[0.1 * 0.1 + 0.3 * 0.5 + 0.7 * 0.4],
              [0.4 * 0.1 + 0.6 * 0.5 + 0.8 * 0.4]])
        test([v1, v2],
             ivs,
             [0.1, 0.2, 0.4, 0.3],
             {v1: [[0.1, 0.2, 0.3],
                   [0.4, 0.5, 0.6]],
              v2: [[0.7],
                   [0.8]],
              ivs: [[1],
                    [-1]]},
             [[0.2 * 0.2],
              [0.4 * 0.1 + 0.5 * 0.2 + 0.6 * 0.4 + 0.8 * 0.3]])
        test([(v1, [0, 2]), (v2, [0])],
             (ivs, [0, 1, 2]),
             [0.1, 0.5, 0.4],
             {v1: [[0.1, 0.2, 0.3],
                   [0.4, 0.5, 0.6]],
              v2: [[0.7],
                   [0.8]],
              ivs: [[1],
                    [-1]]},
             [[0.3 * 0.5],
              [0.4 * 0.1 + 0.6 * 0.5 + 0.8 * 0.4]])

        # Single input with 1 value, multi-element batch
        test([v2],
             None,
             [0.5],
             {v2: [[0.1],
                   [0.2]]},
             [[0.1 * 1.0],
              [0.2 * 1.0]])
        test([(v1, [1])],
             None,
             [0.5],
             {v1: [[0.01, 0.1, 0.03],
                   [0.02, 0.2, 0.04]]},
             [[0.1 * 1.0],
              [0.2 * 1.0]])
        test([v2],
             (ivs, 0),
             [0.5],
             {v2: [[0.1],
                   [0.2],
                   [0.3]],
              ivs: [[0],
                    [-1],
                    [1]]},
             [[0.1 * 1.0],
              [0.2 * 1.0],
              [0.0]])
        test([(v1, [1])],
             (ivs, 1),
             [0.5],
             {v1: [[0.01, 0.1, 0.03],
                   [0.02, 0.2, 0.04],
                   [0.03, 0.3, 0.05]],
              ivs: [[0],
                    [-1],
                    [1]]},
             [[0.0],
              [0.2 * 1.0],
              [0.3 * 1.0]])

        # Multiple inputs, single-element batch
        test([v1, v2],
             None,
             [0.1, 0.2, 0.4, 0.3],
             {v1: [[0.1, 0.2, 0.3]],
              v2: [[0.7]]},
             [[0.1 * 0.1 + 0.2 * 0.2 + 0.3 * 0.4 + 0.7 * 0.3]])
        test([(v1, [0, 2]), (v2, [0])],
             None,
             [0.1, 0.5, 0.4],
             {v1: [[0.1, 0.2, 0.3]],
              v2: [[0.7]]},
             [[0.1 * 0.1 + 0.3 * 0.5 + 0.7 * 0.4]])
        test([v1, v2],
             ivs,
             [0.1, 0.2, 0.4, 0.3],
             {v1: [[0.1, 0.2, 0.3]],
              v2: [[0.7]],
              ivs: [[1]]},
             [[0.2 * 0.2]])
        test([(v1, [0, 2]), (v2, [0])],
             (ivs, [1, 2, 3]),
             [0.1, 0.5, 0.4],
             {v1: [[0.1, 0.2, 0.3]],
              v2: [[0.7]],
              ivs: [[-1]]},
             [[0.1 * 0.1 + 0.3 * 0.5 + 0.7 * 0.4]])

        # Single input with 1 value, single-element batch
        test([v2],
             None,
             [0.5],
             {v2: [[0.1]]},
             [[0.1 * 1.0]])
        test([(v1, [1])],
             None,
             [0.5],
             {v1: [[0.01, 0.1, 0.03]]},
             [[0.1 * 1.0]])
        test([v2],
             (ivs, [1]),
             [0.5],
             {v2: [[0.1]],
              ivs: [[-1]]},
             [[0.1 * 1.0]])
        test([(v1, [1])],
             (ivs, [1]),
             [0.5],
             {v1: [[0.01, 0.1, 0.03]],
              ivs: [[0]]},
             [[0.0]])

    def test_compute_mpe_value(self):
        """Calculating MPE value of Sum"""

        def test(values, ivs, weights, feed, output):
            with self.subTest(values=values, ivs=ivs, weights=weights,
                              feed=feed):
                n = spn.Sum(*values, ivs=ivs)
                n.generate_weights(weights)
                op = n.get_value(spn.InferenceType.MPE)
                op_log = n.get_log_value(spn.InferenceType.MPE)
                with tf.Session() as sess:
                    spn.initialize_weights(n).run()
                    out = sess.run(op, feed_dict=feed)
                    out_log = sess.run(tf.exp(op_log), feed_dict=feed)
                np.testing.assert_array_almost_equal(
                    out,
                    np.array(output, dtype=spn.conf.dtype.as_numpy_dtype()))
                np.testing.assert_array_almost_equal(
                    out_log,
                    np.array(output, dtype=spn.conf.dtype.as_numpy_dtype()))

        # Create inputs
        v1 = spn.ContVars(num_vars=3, name="ContVars1")
        v2 = spn.ContVars(num_vars=1, name="ContVars2")
        ivs = spn.IVs(num_vars=1, num_vals=4)

        # Multiple inputs, multi-element batch
        test([v1, v2],
             None,
             [0.1, 0.2, 0.4, 0.3],
             {v1: [[0.1, 0.2, 0.3],
                   [0.4, 0.5, 0.6]],
              v2: [[0.7],
                   [0.8]]},
             [[0.7 * 0.3],
              [0.8 * 0.3]])
        test([(v1, [0, 2]), (v2, [0])],
             None,
             [0.1, 0.5, 0.4],
             {v1: [[0.1, 0.2, 0.3],
                   [0.4, 0.5, 0.6]],
              v2: [[0.7],
                   [0.8]]},
             [[0.7 * 0.4],
              [0.8 * 0.4]])
        test([v1, v2],
             ivs,
             [0.1, 0.2, 0.4, 0.3],
             {v1: [[0.1, 0.2, 0.3],
                   [0.4, 0.5, 0.6]],
              v2: [[0.7],
                   [0.8]],
              ivs: [[2],
                    [-1]]},
             [[0.3 * 0.4],
              [0.8 * 0.3]])
        test([(v1, [0, 2]), (v2, [0])],
             (ivs, [0, 1, 2]),
             [0.1, 0.5, 0.4],
             {v1: [[0.1, 0.2, 0.3],
                   [0.4, 0.5, 0.6]],
              v2: [[0.7],
                   [0.8]],
              ivs: [[1],
                    [-1]]},
             [[0.3 * 0.5],
              [0.8 * 0.4]])

        # Single input with 1 value, multi-element batch
        test([v2],
             None,
             [0.5],
             {v2: [[0.1],
                   [0.2]]},
             [[0.1 * 1.0],
              [0.2 * 1.0]])
        test([(v1, [1])],
             None,
             [0.5],
             {v1: [[0.01, 0.1, 0.03],
                   [0.02, 0.2, 0.04]]},
             [[0.1 * 1.0],
              [0.2 * 1.0]])
        test([v2],
             (ivs, 0),
             [0.5],
             {v2: [[0.1],
                   [0.2]],
              ivs: [[1],
                    [-1]]},
             [[0.0],
              [0.2 * 1.0]])
        test([(v1, [1])],
             (ivs, 1),
             [0.5],
             {v1: [[0.01, 0.1, 0.03],
                   [0.02, 0.2, 0.04]],
              ivs: [[1],
                    [-1]]},
             [[0.1 * 1.0],
              [0.2 * 1.0]])

        # Multiple inputs, single-element batch
        test([v1, v2],
             None,
             [0.1, 0.2, 0.4, 0.3],
             {v1: [[0.1, 0.2, 0.3]],
              v2: [[0.7]]},
             [[0.7 * 0.3]])
        test([(v1, [0, 2]), (v2, [0])],
             None,
             [0.1, 0.5, 0.4],
             {v1: [[0.1, 0.2, 0.3]],
              v2: [[0.7]]},
             [[0.7 * 0.4]])
        test([v1, v2],
             ivs,
             [0.1, 0.2, 0.4, 0.3],
             {v1: [[0.1, 0.2, 0.3]],
              v2: [[0.7]],
              ivs: [[0]]},
             [[0.1 * 0.1]])
        test([(v1, [0, 2]), (v2, [0])],
             (ivs, [1, 2, 3]),
             [0.1, 0.5, 0.4],
             {v1: [[0.1, 0.2, 0.3]],
              v2: [[0.7]],
              ivs: [[-1]]},
             [[0.7 * 0.4]])

        # Single input with 1 value, single-element batch
        test([v2],
             None,
             [0.5],
             {v2: [[0.1]]},
             [[0.1 * 1.0]])
        test([(v1, [1])],
             None,
             [0.5],
             {v1: [[0.01, 0.1, 0.03]]},
             [[0.1 * 1.0]])
        test([v2],
             (ivs, [1]),
             [0.5],
             {v2: [[0.1]],
              ivs: [[-1]]},
             [[0.1 * 1.0]])
        test([(v1, [1])],
             (ivs, [1]),
             [0.5],
             {v1: [[0.01, 0.1, 0.03]],
              ivs: [[0]]},
             [[0.0]])

    def test_comput_scope(self):
        """Calculating scope of Sum"""
        # Create graph
        v12 = spn.IVs(num_vars=2, num_vals=4, name="V12")
        v34 = spn.ContVars(num_vars=2, name="V34")
        s1 = spn.Sum((v12, [0, 1, 2, 3]), name="S1")
        s1.generate_ivs()
        s2 = spn.Sum((v12, [4, 5, 6, 7]), name="S2")
        p1 = spn.Product((v12, [0, 7]), name="P1")
        p2 = spn.Product((v12, [3, 4]), name="P1")
        p3 = spn.Product(v34, name="P3")
        n1 = spn.Concat(s1, s2, p3, name="N1")
        n2 = spn.Concat(p1, p2, name="N2")
        p4 = spn.Product((n1, [0]), (n1, [1]), name="P4")
        p5 = spn.Product((n2, [0]), (n1, [2]), name="P5")
        s3 = spn.Sum(p4, n2, name="S3")
        p6 = spn.Product(s3, (n1, [2]), name="P6")
        s4 = spn.Sum(p5, p6, name="S4")
        s4.generate_ivs()
        # Test
        self.assertListEqual(v12.get_scope(),
                             [spn.Scope(v12, 0), spn.Scope(v12, 0),
                              spn.Scope(v12, 0), spn.Scope(v12, 0),
                              spn.Scope(v12, 1), spn.Scope(v12, 1),
                              spn.Scope(v12, 1), spn.Scope(v12, 1)])
        self.assertListEqual(v34.get_scope(),
                             [spn.Scope(v34, 0), spn.Scope(v34, 1)])
        self.assertListEqual(s1.get_scope(),
                             [spn.Scope(v12, 0) | spn.Scope(s1.ivs.node, 0)])
        self.assertListEqual(s2.get_scope(),
                             [spn.Scope(v12, 1)])
        self.assertListEqual(p1.get_scope(),
                             [spn.Scope(v12, 0) | spn.Scope(v12, 1)])
        self.assertListEqual(p2.get_scope(),
                             [spn.Scope(v12, 0) | spn.Scope(v12, 1)])
        self.assertListEqual(p3.get_scope(),
                             [spn.Scope(v34, 0) | spn.Scope(v34, 1)])
        self.assertListEqual(n1.get_scope(),
                             [spn.Scope(v12, 0) | spn.Scope(s1.ivs.node, 0),
                              spn.Scope(v12, 1),
                              spn.Scope(v34, 0) | spn.Scope(v34, 1)])
        self.assertListEqual(n2.get_scope(),
                             [spn.Scope(v12, 0) | spn.Scope(v12, 1),
                              spn.Scope(v12, 0) | spn.Scope(v12, 1)])
        self.assertListEqual(p4.get_scope(),
                             [spn.Scope(v12, 0) | spn.Scope(v12, 1) |
                              spn.Scope(s1.ivs.node, 0)])
        self.assertListEqual(p5.get_scope(),
                             [spn.Scope(v12, 0) | spn.Scope(v12, 1) |
                              spn.Scope(v34, 0) | spn.Scope(v34, 1)])
        self.assertListEqual(s3.get_scope(),
                             [spn.Scope(v12, 0) | spn.Scope(v12, 1) |
                              spn.Scope(s1.ivs.node, 0)])
        self.assertListEqual(p6.get_scope(),
                             [spn.Scope(v12, 0) | spn.Scope(v12, 1) |
                              spn.Scope(v34, 0) | spn.Scope(v34, 1) |
                              spn.Scope(s1.ivs.node, 0)])
        self.assertListEqual(s4.get_scope(),
                             [spn.Scope(v12, 0) | spn.Scope(v12, 1) |
                              spn.Scope(v34, 0) | spn.Scope(v34, 1) |
                              spn.Scope(s1.ivs.node, 0) |
                              spn.Scope(s4.ivs.node, 0)])

    def test_compute_valid(self):
        """Calculating validity of Sum"""
        # Without IVs
        v12 = spn.IVs(num_vars=2, num_vals=4)
        v34 = spn.ContVars(num_vars=2)
        s1 = spn.Sum((v12, [0, 1, 2, 3]))
        s2 = spn.Sum((v12, [0, 1, 2, 4]))
        s3 = spn.Sum((v12, [0, 1, 2, 3]), (v34, 0))
        p1 = spn.Product((v12, [0, 5]), (v34, 0))
        p2 = spn.Product((v12, [1, 6]), (v34, 0))
        p3 = spn.Product((v12, [1, 6]), (v34, 1))
        s4 = spn.Sum(p1, p2)
        s5 = spn.Sum(p1, p3)
        self.assertTrue(v12.is_valid())
        self.assertTrue(v34.is_valid())
        self.assertTrue(s1.is_valid())
        self.assertFalse(s2.is_valid())
        self.assertFalse(s3.is_valid())
        self.assertTrue(s4.is_valid())
        self.assertFalse(s5.is_valid())
        # With IVS
        s6 = spn.Sum(p1, p2)
        s6.generate_ivs()
        self.assertTrue(s6.is_valid())
        s7 = spn.Sum(p1, p2)
        s7.set_ivs(spn.ContVars(num_vars=2))
        self.assertFalse(s7.is_valid())
        s8 = spn.Sum(p1, p2)
        s8.set_ivs(spn.IVs(num_vars=2, num_vals=2))
        with self.assertRaises(spn.StructureError):
            s8.is_valid()
        s9 = spn.Sum(p1, p2)
        s9.set_ivs((v12, [0, 3]))
        self.assertTrue(s9.is_valid())

    def test_compute_mpe_path_noivs(self):
        v12 = spn.IVs(num_vars=2, num_vals=4)
        v34 = spn.ContVars(num_vars=2)
        v5 = spn.ContVars(num_vars=1)
        s = spn.Sum((v12, [0, 5]), v34, (v12, [3]), v5)
        w = s.generate_weights()
        counts = tf.placeholder(tf.float32, shape=(None, 1))
        op = s._compute_mpe_path(tf.identity(counts),
                                 w.get_value(),
                                 None,
                                 v12.get_value(),
                                 v34.get_value(),
                                 v12.get_value(),
                                 v5.get_value())
        init = w.initialize()
        counts_feed = [[10],
                       [11],
                       [12],
                       [13]]
        v12_feed = [[0, 1],
                    [1, 1],
                    [0, 0],
                    [3, 3]]
        v34_feed = [[0.1, 0.2],
                    [1.2, 0.2],
                    [0.1, 0.2],
                    [0.9, 0.8]]
        v5_feed = [[0.5],
                   [0.5],
                   [1.2],
                   [0.9]]

        with tf.Session() as sess:
            sess.run(init)
            # Skip the IVs op
            out = sess.run(op[:1] + op[2:], feed_dict={counts: counts_feed,
                                                       v12: v12_feed,
                                                       v34: v34_feed,
                                                       v5: v5_feed})
        # Weights
        np.testing.assert_array_almost_equal(
            out[0], np.array([[10., 0., 0., 0., 0., 0.],
                              [0., 0., 11., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 12.],
                              [0., 0., 0., 0., 13., 0.]],
                             dtype=np.float32))
        np.testing.assert_array_almost_equal(
            out[1], np.array([[10., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0.]],
                             dtype=np.float32))
        np.testing.assert_array_almost_equal(
            out[2], np.array([[0., 0.],
                              [11., 0.],
                              [0., 0.],
                              [0., 0.]],
                             dtype=np.float32))
        np.testing.assert_array_almost_equal(
            out[3], np.array([[0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 13., 0., 0., 0., 0.]],
                             dtype=np.float32))
        np.testing.assert_array_almost_equal(
            out[4], np.array([[0.],
                              [0.],
                              [12.],
                              [0.]],
                             dtype=np.float32))

    def test_compute_mpe_path_ivs(self):
        v12 = spn.IVs(num_vars=2, num_vals=4)
        v34 = spn.ContVars(num_vars=2)
        v5 = spn.ContVars(num_vars=1)
        s = spn.Sum((v12, [0, 5]), v34, (v12, [3]), v5)
        iv = s.generate_ivs()
        w = s.generate_weights()
        counts = tf.placeholder(tf.float32, shape=(None, 1))
        op = s._compute_mpe_path(tf.identity(counts),
                                 w.get_value(),
                                 iv.get_value(),
                                 v12.get_value(),
                                 v34.get_value(),
                                 v12.get_value(),
                                 v5.get_value())
        init = w.initialize()
        counts_feed = [[10],
                       [11],
                       [12],
                       [13],
                       [14],
                       [15],
                       [16],
                       [17]]
        v12_feed = [[0, 1],
                    [1, 1],
                    [0, 0],
                    [3, 3],
                    [0, 1],
                    [1, 1],
                    [0, 0],
                    [3, 3]]
        v34_feed = [[0.1, 0.2],
                    [1.2, 0.2],
                    [0.1, 0.2],
                    [0.9, 0.8],
                    [0.1, 0.2],
                    [1.2, 0.2],
                    [0.1, 0.2],
                    [0.9, 0.8]]
        v5_feed = [[0.5],
                   [0.5],
                   [1.2],
                   [0.9],
                   [0.5],
                   [0.5],
                   [1.2],
                   [0.9]]
        ivs_feed = [[-1], [-1], [-1], [-1], [1], [2], [3], [1]]

        with tf.Session() as sess:
            sess.run(init)
            # Skip the IVs op
            out = sess.run(op, feed_dict={counts: counts_feed,
                                          iv: ivs_feed,
                                          v12: v12_feed,
                                          v34: v34_feed,
                                          v5: v5_feed})
        # Weights
        np.testing.assert_array_almost_equal(
            out[0], np.array([[10., 0., 0., 0., 0., 0.],
                              [0., 0., 11., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 12.],
                              [0., 0., 0., 0., 13., 0.],
                              [0., 14., 0., 0., 0., 0.],
                              [0., 0., 15., 0., 0., 0.],
                              [0., 0., 0., 16., 0., 0.],
                              [17., 0., 0., 0., 0., 0.]],
                             dtype=np.float32))
        np.testing.assert_array_almost_equal(
            out[1], np.array([[10., 0., 0., 0., 0., 0.],
                              [0., 0., 11., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 12.],
                              [0., 0., 0., 0., 13., 0.],
                              [0., 14., 0., 0., 0., 0.],
                              [0., 0., 15., 0., 0., 0.],
                              [0., 0., 0., 16., 0., 0.],
                              [17., 0., 0., 0., 0., 0.]],
                             dtype=np.float32))
        np.testing.assert_array_almost_equal(
            out[2], np.array([[10., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 14., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0.],
                              [17., 0., 0., 0., 0., 0., 0., 0.]],
                             dtype=np.float32))
        np.testing.assert_array_almost_equal(
            out[3], np.array([[0., 0.],
                              [11., 0.],
                              [0., 0.],
                              [0., 0.],
                              [0., 0.],
                              [15., 0.],
                              [0., 16.],
                              [0., 0.]],
                             dtype=np.float32))
        np.testing.assert_array_almost_equal(
            out[4], np.array([[0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 13., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0.]],
                             dtype=np.float32))
        np.testing.assert_array_almost_equal(
            out[5], np.array([[0.],
                              [0.],
                              [12.],
                              [0.],
                              [0.],
                              [0.],
                              [0.],
                              [0.]],
                             dtype=np.float32))

    def test_compute_gradients(self):
        v12 = spn.IVs(num_vars=2, num_vals=4)
        v34 = spn.ContVars(num_vars=2)
        v5 = spn.ContVars(num_vars=1)
        s = spn.Sum((v12, [0, 5]), v34, (v12, [3]), v5)
        iv = s.generate_ivs()
        weights = np.random.rand(6)
        w = s.generate_weights(weights)
        gradients = tf.placeholder(tf.float32, shape=(None, 1))
        op = s._compute_gradient(tf.identity(gradients),
                                 w.get_value(),
                                 iv.get_value(),
                                 v12.get_value(),
                                 v34.get_value(),
                                 v12.get_value(),
                                 v5.get_value())
        init = w.initialize()
        batch_size = 100
        gradients_feed = np.random.rand(batch_size, 1)
        v12_feed = np.random.randint(4, size=(batch_size, 2))
        v34_feed = np.random.rand(batch_size, 2)
        v5_feed = np.random.rand(batch_size, 1)
        ivs_feed = np.random.randint(6, size=(batch_size, 1))

        with tf.Session() as sess:
            sess.run(init)
            # Skip the IVs op
            out = sess.run(op, feed_dict={gradients: gradients_feed,
                                          iv: ivs_feed,
                                          v12: v12_feed,
                                          v34: v34_feed,
                                          v5: v5_feed})

        # Calculate true outputs
        v12_inputs = np.hstack([np.eye(4)[v12_feed[:, 0]],
                                np.eye(4)[v12_feed[:, 1]]])
        input_values = np.hstack([np.expand_dims(v12_inputs[:, 0], axis=1),
                                  np.expand_dims(v12_inputs[:, 5], axis=1),
                                  v34_feed,
                                  np.expand_dims(v12_inputs[:, 3], axis=1),
                                  v5_feed])
        weights_normalised = weights / np.sum(weights)
        weights_gradients = gradients_feed * input_values
        output_gradients = np.split((gradients_feed * weights_normalised),
                                    [2, 4, 5, 6], axis=1)
        output_gradients_0 = np.zeros((batch_size, 8))
        output_gradients_0[:, 0] = output_gradients[0][:, 0]
        output_gradients_0[:, 5] = output_gradients[0][:, 1]
        output_gradients[0] = output_gradients_0
        output_gradients_2 = np.zeros((batch_size, 8))
        output_gradients_2[:, 3] = output_gradients[2][:, 0]
        output_gradients[2] = output_gradients_2

        # Weights
        np.testing.assert_array_almost_equal(
            out[0], weights_gradients)
        # IVs
        np.testing.assert_array_almost_equal(
           out[1], gradients_feed * input_values)
        # Inputs
        np.testing.assert_array_almost_equal(
           out[2], output_gradients[0])
        np.testing.assert_array_almost_equal(
           out[3], output_gradients[1])
        np.testing.assert_array_almost_equal(
           out[4], output_gradients[2])
        np.testing.assert_array_almost_equal(
           out[5], output_gradients[3])

    def test_compute_log_gradients(self):
        v12 = spn.IVs(num_vars=2, num_vals=4)
        v34 = spn.ContVars(num_vars=2)
        v5 = spn.ContVars(num_vars=1)
        s = spn.Sum((v12, [0, 5]), v34, (v12, [3]), v5)
        iv = s.generate_ivs()
        weights = np.random.rand(6)
        w = s.generate_weights(weights)
        gradients = tf.placeholder(tf.float32, shape=(None, 1))
        op = s._compute_log_gradient(tf.identity(gradients),
                                     w.get_log_value(),
                                     iv.get_log_value(),
                                     v12.get_log_value(),
                                     v34.get_log_value(),
                                     v12.get_log_value(),
                                     v5.get_log_value())
        init = w.initialize()
        batch_size = 10
        gradients_feed = np.random.rand(batch_size, 1)
        v12_feed = np.random.randint(4, size=(batch_size, 2))
        v34_feed = np.random.rand(batch_size, 2)
        v5_feed = np.random.rand(batch_size, 1)
        ivs_feed = np.random.randint(6, size=(batch_size, 1))

        with tf.Session() as sess:
            sess.run(init)
            # Skip the IVs op
            out = sess.run(op, feed_dict={gradients: gradients_feed,
                                          iv: ivs_feed,
                                          v12: v12_feed,
                                          v34: v34_feed,
                                          v5: v5_feed})

        # Calculate true outputs
        v12_inputs = np.hstack([np.eye(4)[v12_feed[:, 0]],
                                np.eye(4)[v12_feed[:, 1]]])
        input_values = np.hstack([np.expand_dims(v12_inputs[:, 0], axis=1),
                                  np.expand_dims(v12_inputs[:, 5], axis=1),
                                  v34_feed,
                                  np.expand_dims(v12_inputs[:, 3], axis=1),
                                  v5_feed])
        weights_normalised = weights / np.sum(weights)
        weights_log = np.log(weights_normalised)
        inputs_log = np.log(input_values)
        ivs_values = np.eye(6)[np.squeeze(ivs_feed, axis=1)]
        ivs_log = np.log(ivs_values)
        weighted_inputs = weights_log + (inputs_log + ivs_log)
        weighted_inputs_exp = np.exp(weighted_inputs)
        weights_gradients = gradients_feed * np.divide(weighted_inputs_exp,
                                                       np.sum(weighted_inputs_exp,
                                                              axis=1, keepdims=True))
        output_gradients = np.split(weights_gradients, [2, 4, 5, 6], axis=1)
        output_gradients_0 = np.zeros((batch_size, 8))
        output_gradients_0[:, 0] = output_gradients[0][:, 0]
        output_gradients_0[:, 5] = output_gradients[0][:, 1]
        output_gradients[0] = output_gradients_0
        output_gradients_2 = np.zeros((batch_size, 8))
        output_gradients_2[:, 3] = output_gradients[2][:, 0]
        output_gradients[2] = output_gradients_2

        # Weights
        np.testing.assert_array_almost_equal(
            out[0], weights_gradients)
        # IVs
        np.testing.assert_array_almost_equal(
           out[1], weights_gradients)
        # Inputs
        np.testing.assert_array_almost_equal(
           out[2], output_gradients[0])
        np.testing.assert_array_almost_equal(
           out[3], output_gradients[1])
        np.testing.assert_array_almost_equal(
           out[4], output_gradients[2])
        np.testing.assert_array_almost_equal(
           out[5], output_gradients[3])

    def test_compute_log_gradients_log(self):
        v12 = spn.IVs(num_vars=2, num_vals=4)
        v34 = spn.ContVars(num_vars=2)
        v5 = spn.ContVars(num_vars=1)
        s = spn.Sum((v12, [0, 5]), v34, (v12, [3]), v5)
        iv = s.generate_ivs()
        weights = np.random.rand(6)
        w = s.generate_weights(weights)
        gradients = tf.placeholder(tf.float32, shape=(None, 1))
        with_ivs = True
        op = s._compute_log_gradient(tf.identity(gradients),
                                     w.get_log_value(),
                                     iv.get_log_value(),
                                     v12.get_log_value(),
                                     v34.get_log_value(),
                                     v12.get_log_value(),
                                     v5.get_log_value(),
                                     with_ivs=with_ivs)

        init = w.initialize()
        batch_size = 10
        gradients_feed = np.random.rand(batch_size, 1)
        v12_feed = np.random.randint(4, size=(batch_size, 2))
        v34_feed = np.random.rand(batch_size, 2)
        v5_feed = np.random.rand(batch_size, 1)
        ivs_feed = np.random.randint(6, size=(batch_size, 1))

        with tf.Session() as sess:
            sess.run(init)
            # Skip the IVs op
            out = sess.run(op, feed_dict={gradients: gradients_feed,
                                          iv: ivs_feed,
                                          v12: v12_feed,
                                          v34: v34_feed,
                                          v5: v5_feed})

        # Calculate true outputs
        v12_inputs = np.hstack([np.eye(4)[v12_feed[:, 0]],
                                np.eye(4)[v12_feed[:, 1]]])
        input_values = np.hstack([np.expand_dims(v12_inputs[:, 0], axis=1),
                                  np.expand_dims(v12_inputs[:, 5], axis=1),
                                  v34_feed,
                                  np.expand_dims(v12_inputs[:, 3], axis=1),
                                  v5_feed])
        input_values_log = np.log(input_values)
        weights_normalised = weights / np.sum(weights)
        weights_log = np.log(weights_normalised)
        print("\nweights_normalised:\n", weights_normalised)
        ivs_values = np.eye(6)[np.squeeze(ivs_feed, axis=1)]
        ivs_log = np.log(ivs_values)
        if with_ivs:
            values_weighted = weights_log + (input_values_log + ivs_log)
        else:
            values_weighted = weights_log + input_values_log
        print("\nvalues_weighted:\n", values_weighted)
        log_max = np.amax(values_weighted, axis=1, keepdims=True)
        print("\nlog_max:\n", log_max)
        log_rebased = np.subtract(values_weighted, log_max)
        print("\nlog_rebased:\n", log_rebased)
        expo_logs = np.exp(log_rebased)
        print("\nexpo_logs:\n", expo_logs)
        summed_exponents = np.sum(expo_logs, axis=1, keepdims=True)
        print("\nsummed_exponents:\n", summed_exponents)

        max_indices = np.argmax(values_weighted, axis=1)
        print("\nmax_indices:\n", max_indices)
        expo_logs_normalized = np.divide(expo_logs, summed_exponents)
        print("\nexpo_logs_normalized:\n", expo_logs_normalized)
        expos_excl_max = expo_logs_normalized
        expos_excl_max[np.arange(batch_size), max_indices] = 0.0
        print("\nexpos_excl_max:\n", expos_excl_max)
        summed_expos_excl_max = np.sum(expos_excl_max, axis=1, keepdims=True)
        print("\nsummed_expos_excl_max:\n", summed_expos_excl_max)
        max_weight_gradient = 1.0 - summed_expos_excl_max
        print("\nmax_weight_gradient:\n", max_weight_gradient)
        max_weight_gradient_scattered = np.zeros_like(values_weighted)
        max_weight_gradient_scattered[np.arange(batch_size), max_indices] = 1.0
        max_weight_gradient_scattered *= max_weight_gradient
        print("\nmax_weight_gradient_scattered:\n", max_weight_gradient_scattered)
        weights_gradients = gradients_feed * (expos_excl_max + max_weight_gradient_scattered)
        output_gradients = np.split(weights_gradients, [2, 4, 5, 6], axis=1)

        output_gradients_0 = np.zeros((batch_size, 8))
        output_gradients_0[:, 0] = output_gradients[0][:, 0]
        output_gradients_0[:, 5] = output_gradients[0][:, 1]
        output_gradients[0] = output_gradients_0
        output_gradients_2 = np.zeros((batch_size, 8))
        output_gradients_2[:, 3] = output_gradients[2][:, 0]
        output_gradients[2] = output_gradients_2

        print("\nout[0]:\n", out[0])
        print("\nweights_gradients:\n", weights_gradients)

        # Weights
        np.testing.assert_array_almost_equal(
            out[0], weights_gradients, decimal=6)
        # IVs
        np.testing.assert_array_almost_equal(
           out[1], weights_gradients)
        # Inputs
        np.testing.assert_array_almost_equal(
           out[2], output_gradients[0])
        np.testing.assert_array_almost_equal(
           out[3], output_gradients[1])
        np.testing.assert_array_almost_equal(
           out[4], output_gradients[2])
        np.testing.assert_array_almost_equal(
           out[5], output_gradients[3])


if __name__ == '__main__':
    tf.test.main()
