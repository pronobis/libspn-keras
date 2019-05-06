#!/usr/bin/env python3

from context import libspn as spn
from test import TestCase
import tensorflow as tf
import numpy as np


class TestGraphProduct(tf.test.TestCase):

    def test_compute_value(self):
        """Calculating value of Product"""

        def test(inputs, feed, output):
            with self.subTest(inputs=inputs, feed=feed):
                n = spn.Product(*inputs)
                op = n.get_value(spn.InferenceType.MARGINAL)
                op_log = n.get_log_value(spn.InferenceType.MARGINAL)
                op_mpe = n.get_value(spn.InferenceType.MPE)
                op_log_mpe = n.get_log_value(spn.InferenceType.MPE)
                with self.test_session() as sess:
                    out = sess.run(op, feed_dict=feed)
                    out_log = sess.run(tf.exp(op_log), feed_dict=feed)
                    out_mpe = sess.run(op_mpe, feed_dict=feed)
                    out_log_mpe = sess.run(tf.exp(op_log_mpe), feed_dict=feed)
                self.assertAllClose(
                    out,
                    np.array(output, dtype=spn.conf.dtype.as_numpy_dtype()))
                self.assertAllClose(
                    out_log,
                    np.array(output, dtype=spn.conf.dtype.as_numpy_dtype()))
                self.assertAllClose(
                    out_mpe,
                    np.array(output, dtype=spn.conf.dtype.as_numpy_dtype()))
                self.assertAllClose(
                    out_log_mpe,
                    np.array(output, dtype=spn.conf.dtype.as_numpy_dtype()))

        # Create inputs
        v1 = spn.RawLeaf(num_vars=3)
        v2 = spn.RawLeaf(num_vars=1)

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

    def test_comput_scope(self):
        """Calculating scope of Product"""
        # Create a graph
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
        """Calculating validity of Product"""
        v12 = spn.IndicatorLeaf(num_vars=2, num_vals=4)
        v34 = spn.RawLeaf(num_vars=2)
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

    def test_compute_log_mpe_path(self):
        v12 = spn.IndicatorLeaf(num_vars=2, num_vals=4)
        v34 = spn.RawLeaf(num_vars=2)
        v5 = spn.RawLeaf(num_vars=1)
        p = spn.Product((v12, [0, 5]), v34, (v12, [3]), v5)
        counts = tf.placeholder(tf.float32, shape=(None, 1))
        op = p._compute_log_mpe_path(tf.identity(counts),
                                 v12.get_value(),
                                 v34.get_value(),
                                 v12.get_value(),
                                 v5.get_value())
        feed = [[0],
                [1],
                [2]]
        with self.test_session() as sess:
            out = sess.run(op, feed_dict={counts: feed})
        self.assertAllClose(
            out[0], np.array([[0., 0., 0., 0., 0., 0., 0., 0.],
                              [1., 0., 0., 0., 0., 1., 0., 0.],
                              [2., 0., 0., 0., 0., 2., 0., 0.]],
                             dtype=np.float32))
        self.assertAllClose(
            out[1], np.array([[0., 0.],
                              [1., 1.],
                              [2., 2.]],
                             dtype=np.float32))
        self.assertAllClose(
            out[2], np.array([[0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 1., 0., 0., 0., 0.],
                              [0., 0., 0., 2., 0., 0., 0., 0.]],
                             dtype=np.float32))
        self.assertAllClose(
            out[3], np.array([[0.],
                              [1.],
                              [2.]],
                             dtype=np.float32))

if __name__ == '__main__':
    tf.test.main()
