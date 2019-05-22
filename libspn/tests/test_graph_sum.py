#!/usr/bin/env python3

from context import libspn as spn
from test import TestCase
import tensorflow as tf
import numpy as np
import libspn as spn

class TestGraphSum(TestCase):

    def test_compute_marginal_value(self):
        """Calculating marginal value of Sum"""

        def test(values, latent_indicators, weights, feed, output):
            with self.subTest(values=values, latent_indicators=latent_indicators, weights=weights,
                              feed=feed):
                n = spn.Sum(*values, latent_indicators=latent_indicators)
                n.generate_weights(tf.initializers.constant(weights))
                op = n.get_value(spn.InferenceType.MARGINAL)
                op_log = n.get_log_value(spn.InferenceType.MARGINAL)
                with self.test_session() as sess:
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
        v1 = spn.RawLeaf(num_vars=3, name="RawLeaf1")
        v2 = spn.RawLeaf(num_vars=1, name="RawLeaf2")
        latent_indicators = spn.IndicatorLeaf(num_vars=1, num_vals=4)

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
             latent_indicators,
             [0.1, 0.2, 0.4, 0.3],
             {v1: [[0.1, 0.2, 0.3],
                   [0.4, 0.5, 0.6]],
              v2: [[0.7],
                   [0.8]],
              latent_indicators: [[1],
                    [-1]]},
             [[0.2 * 0.2],
              [0.4 * 0.1 + 0.5 * 0.2 + 0.6 * 0.4 + 0.8 * 0.3]])
        test([(v1, [0, 2]), (v2, [0])],
             (latent_indicators, [0, 1, 2]),
             [0.1, 0.5, 0.4],
             {v1: [[0.1, 0.2, 0.3],
                   [0.4, 0.5, 0.6]],
              v2: [[0.7],
                   [0.8]],
              latent_indicators: [[1],
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
             (latent_indicators, 0),
             [0.5],
             {v2: [[0.1],
                   [0.2],
                   [0.3]],
              latent_indicators: [[0],
                    [-1],
                    [1]]},
             [[0.1 * 1.0],
              [0.2 * 1.0],
              [0.0]])
        test([(v1, [1])],
             (latent_indicators, 1),
             [0.5],
             {v1: [[0.01, 0.1, 0.03],
                   [0.02, 0.2, 0.04],
                   [0.03, 0.3, 0.05]],
              latent_indicators: [[0],
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
             latent_indicators,
             [0.1, 0.2, 0.4, 0.3],
             {v1: [[0.1, 0.2, 0.3]],
              v2: [[0.7]],
              latent_indicators: [[1]]},
             [[0.2 * 0.2]])
        test([(v1, [0, 2]), (v2, [0])],
             (latent_indicators, [1, 2, 3]),
             [0.1, 0.5, 0.4],
             {v1: [[0.1, 0.2, 0.3]],
              v2: [[0.7]],
              latent_indicators: [[-1]]},
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
             (latent_indicators, [1]),
             [0.5],
             {v2: [[0.1]],
              latent_indicators: [[-1]]},
             [[0.1 * 1.0]])
        test([(v1, [1])],
             (latent_indicators, [1]),
             [0.5],
             {v1: [[0.01, 0.1, 0.03]],
              latent_indicators: [[0]]},
             [[0.0]])

    def test_compute_mpe_value(self):
        """Calculating MPE value of Sum"""

        def test(values, latent_indicators, weights, feed, output):
            with self.subTest(values=values, latent_indicators=latent_indicators, weights=weights,
                              feed=feed):
                n = spn.Sum(*values, latent_indicators=latent_indicators)
                n.generate_weights(tf.initializers.constant(weights))
                op = n.get_value(spn.InferenceType.MPE)
                op_log = n.get_log_value(spn.InferenceType.MPE)
                with self.test_session() as sess:
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
        v1 = spn.RawLeaf(num_vars=3, name="RawLeaf1")
        v2 = spn.RawLeaf(num_vars=1, name="RawLeaf2")
        latent_indicators = spn.IndicatorLeaf(num_vars=1, num_vals=4)

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
             latent_indicators,
             [0.1, 0.2, 0.4, 0.3],
             {v1: [[0.1, 0.2, 0.3],
                   [0.4, 0.5, 0.6]],
              v2: [[0.7],
                   [0.8]],
              latent_indicators: [[2],
                    [-1]]},
             [[0.3 * 0.4],
              [0.8 * 0.3]])
        test([(v1, [0, 2]), (v2, [0])],
             (latent_indicators, [0, 1, 2]),
             [0.1, 0.5, 0.4],
             {v1: [[0.1, 0.2, 0.3],
                   [0.4, 0.5, 0.6]],
              v2: [[0.7],
                   [0.8]],
              latent_indicators: [[1],
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
             (latent_indicators, 0),
             [0.5],
             {v2: [[0.1],
                   [0.2]],
              latent_indicators: [[1],
                    [-1]]},
             [[0.0],
              [0.2 * 1.0]])
        test([(v1, [1])],
             (latent_indicators, 1),
             [0.5],
             {v1: [[0.01, 0.1, 0.03],
                   [0.02, 0.2, 0.04]],
              latent_indicators: [[1],
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
             latent_indicators,
             [0.1, 0.2, 0.4, 0.3],
             {v1: [[0.1, 0.2, 0.3]],
              v2: [[0.7]],
              latent_indicators: [[0]]},
             [[0.1 * 0.1]])
        test([(v1, [0, 2]), (v2, [0])],
             (latent_indicators, [1, 2, 3]),
             [0.1, 0.5, 0.4],
             {v1: [[0.1, 0.2, 0.3]],
              v2: [[0.7]],
              latent_indicators: [[-1]]},
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
             (latent_indicators, [1]),
             [0.5],
             {v2: [[0.1]],
              latent_indicators: [[-1]]},
             [[0.1 * 1.0]])
        test([(v1, [1])],
             (latent_indicators, [1]),
             [0.5],
             {v1: [[0.01, 0.1, 0.03]],
              latent_indicators: [[0]]},
             [[0.0]])

    def test_comput_scope(self):
        """Calculating scope of Sum"""
        # Create graph
        v12 = spn.IndicatorLeaf(num_vars=2, num_vals=4, name="V12")
        v34 = spn.RawLeaf(num_vars=2, name="V34")
        s1 = spn.Sum((v12, [0, 1, 2, 3]), name="S1")
        s1.generate_latent_indicators()
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
        s4.generate_latent_indicators()
        # Test
        self.assertListEqual(v12.get_scope(),
                             [spn.Scope(v12, 0), spn.Scope(v12, 0),
                              spn.Scope(v12, 0), spn.Scope(v12, 0),
                              spn.Scope(v12, 1), spn.Scope(v12, 1),
                              spn.Scope(v12, 1), spn.Scope(v12, 1)])
        self.assertListEqual(v34.get_scope(),
                             [spn.Scope(v34, 0), spn.Scope(v34, 1)])
        self.assertListEqual(s1.get_scope(),
                             [spn.Scope(v12, 0) | spn.Scope(s1.latent_indicators.node, 0)])
        self.assertListEqual(s2.get_scope(),
                             [spn.Scope(v12, 1)])
        self.assertListEqual(p1.get_scope(),
                             [spn.Scope(v12, 0) | spn.Scope(v12, 1)])
        self.assertListEqual(p2.get_scope(),
                             [spn.Scope(v12, 0) | spn.Scope(v12, 1)])
        self.assertListEqual(p3.get_scope(),
                             [spn.Scope(v34, 0) | spn.Scope(v34, 1)])
        self.assertListEqual(n1.get_scope(),
                             [spn.Scope(v12, 0) | spn.Scope(s1.latent_indicators.node, 0),
                              spn.Scope(v12, 1),
                              spn.Scope(v34, 0) | spn.Scope(v34, 1)])
        self.assertListEqual(n2.get_scope(),
                             [spn.Scope(v12, 0) | spn.Scope(v12, 1),
                              spn.Scope(v12, 0) | spn.Scope(v12, 1)])
        self.assertListEqual(p4.get_scope(),
                             [spn.Scope(v12, 0) | spn.Scope(v12, 1) |
                              spn.Scope(s1.latent_indicators.node, 0)])
        self.assertListEqual(p5.get_scope(),
                             [spn.Scope(v12, 0) | spn.Scope(v12, 1) |
                              spn.Scope(v34, 0) | spn.Scope(v34, 1)])
        self.assertListEqual(s3.get_scope(),
                             [spn.Scope(v12, 0) | spn.Scope(v12, 1) |
                              spn.Scope(s1.latent_indicators.node, 0)])
        self.assertListEqual(p6.get_scope(),
                             [spn.Scope(v12, 0) | spn.Scope(v12, 1) |
                              spn.Scope(v34, 0) | spn.Scope(v34, 1) |
                              spn.Scope(s1.latent_indicators.node, 0)])
        self.assertListEqual(s4.get_scope(),
                             [spn.Scope(v12, 0) | spn.Scope(v12, 1) |
                              spn.Scope(v34, 0) | spn.Scope(v34, 1) |
                              spn.Scope(s1.latent_indicators.node, 0) |
                              spn.Scope(s4.latent_indicators.node, 0)])

    def test_compute_valid(self):
        """Calculating validity of Sum"""
        # Without IndicatorLeaf
        v12 = spn.IndicatorLeaf(num_vars=2, num_vals=4)
        v34 = spn.RawLeaf(num_vars=2)
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
        s6.generate_latent_indicators()
        self.assertTrue(s6.is_valid())
        s7 = spn.Sum(p1, p2)
        s7.set_latent_indicators(spn.RawLeaf(num_vars=2))
        self.assertFalse(s7.is_valid())
        s8 = spn.Sum(p1, p2)
        s8.set_latent_indicators(spn.IndicatorLeaf(num_vars=2, num_vals=2))
        with self.assertRaises(spn.StructureError):
            s8.is_valid()
        s9 = spn.Sum(p1, p2)
        s9.set_latent_indicators((v12, [0, 3]))
        self.assertTrue(s9.is_valid())

    def test_compute_mpe_path_nolatent_indicators(self):
        spn.conf.argmax_zero = True
        v12 = spn.IndicatorLeaf(num_vars=2, num_vals=4)
        v34 = spn.RawLeaf(num_vars=2)
        v5 = spn.RawLeaf(num_vars=1)
        s = spn.Sum((v12, [0, 5]), v34, (v12, [3]), v5)
        w = s.generate_weights()
        counts = tf.placeholder(tf.float32, shape=(None, 1))
        op = s._compute_log_mpe_path(tf.identity(counts),
                                 w.get_log_value(),
                                 None,
                                 v12.get_log_value(),
                                 v34.get_log_value(),
                                 v12.get_log_value(),
                                 v5.get_log_value())
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

        with self.test_session() as sess:
            sess.run(init)
            # Skip the IndicatorLeaf op
            out = sess.run(op[:1] + op[2:], feed_dict={counts: counts_feed,
                                                       v12: v12_feed,
                                                       v34: v34_feed,
                                                       v5: v5_feed})
        # Weights
        np.testing.assert_array_almost_equal(
            np.squeeze(out[0]), np.array([[10., 0., 0., 0., 0., 0.],
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

    def test_compute_mpe_path_latent_indicators(self):
        spn.conf.argmax_zero = True
        v12 = spn.IndicatorLeaf(num_vars=2, num_vals=4)
        v34 = spn.RawLeaf(num_vars=2)
        v5 = spn.RawLeaf(num_vars=1)
        s = spn.Sum((v12, [0, 5]), v34, (v12, [3]), v5)
        iv = s.generate_latent_indicators()
        w = s.generate_weights()
        counts = tf.placeholder(tf.float32, shape=(None, 1))
        op = s._compute_log_mpe_path(tf.identity(counts),
                                 w.get_log_value(),
                                 iv.get_log_value(),
                                 v12.get_log_value(),
                                 v34.get_log_value(),
                                 v12.get_log_value(),
                                 v5.get_log_value())
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
        latent_indicators_feed = [[-1], [-1], [-1], [-1], [1], [2], [3], [1]]

        with self.test_session() as sess:
            sess.run(init)
            # Skip the IndicatorLeaf op
            out = sess.run(op, feed_dict={counts: counts_feed,
                                          iv: latent_indicators_feed,
                                          v12: v12_feed,
                                          v34: v34_feed,
                                          v5: v5_feed})
        # Weights
        np.testing.assert_array_almost_equal(
            np.squeeze(out[0]), np.array([[10., 0., 0., 0., 0., 0.],
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


if __name__ == '__main__':
    tf.test.main()
