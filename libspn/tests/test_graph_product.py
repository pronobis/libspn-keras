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


class TestGraphProduct(TestCase):

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

    def test_comput_scope(self):
        """Calculating scope of Product"""
        # Create a graph
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

    def test_compute_gradients(self):
        v12 = spn.ContVars(num_vars=8)
        v34 = spn.ContVars(num_vars=2)
        v5 = spn.ContVars(num_vars=1)
        p = spn.Product((v12, [0, 5]), v34, (v12, [3]), v5)
        gradients = tf.placeholder(tf.float32, shape=(None, 1))
        op = p._compute_gradient(tf.identity(gradients),
                                 v12.get_value(),
                                 v34.get_value(),
                                 v12.get_value(),
                                 v5.get_value())
        batch_size = 100
        gradients_feed = np.random.rand(batch_size, 1)
        v12_feed = np.random.rand(batch_size, 8)
        v34_feed = np.random.rand(batch_size, 2)
        v5_feed = np.random.rand(batch_size, 1)

        with tf.Session() as sess:
            out = sess.run(op, feed_dict={gradients: gradients_feed,
                                          v12: v12_feed,
                                          v34: v34_feed,
                                          v5: v5_feed})

        # Calculate true outputs
        input_values = np.hstack([np.expand_dims(v12_feed[:, 0], axis=1),
                                  np.expand_dims(v12_feed[:, 5], axis=1),
                                  v34_feed,
                                  np.expand_dims(v12_feed[:, 3], axis=1),
                                  v5_feed])
        inputs_reduce_prod = np.prod(input_values, axis=1, keepdims=True)
        output_gradients = (inputs_reduce_prod * gradients_feed) / input_values
        output_gradients = np.split(output_gradients, [2, 4, 5, 6], axis=1)
        output_gradients_0 = np.zeros((batch_size, 8))
        output_gradients_0[:, 0] = output_gradients[0][:, 0]
        output_gradients_0[:, 5] = output_gradients[0][:, 1]
        output_gradients[0] = output_gradients_0
        output_gradients_2 = np.zeros((batch_size, 8))
        output_gradients_2[:, 3] = output_gradients[2][:, 0]
        output_gradients[2] = output_gradients_2

        np.testing.assert_array_almost_equal(
            out[0], output_gradients[0])
        np.testing.assert_array_almost_equal(
            out[1], output_gradients[1])
        np.testing.assert_array_almost_equal(
            out[2], output_gradients[2])
        np.testing.assert_array_almost_equal(
            out[3], output_gradients[3])

    def test_compute_log_gradients(self):
        v12 = spn.ContVars(num_vars=8)
        v34 = spn.ContVars(num_vars=2)
        v5 = spn.ContVars(num_vars=1)
        p = spn.Product((v12, [0, 5]), v34, (v12, [3]), v5)
        gradients = tf.placeholder(tf.float32, shape=(None, 1))
        op = p._compute_log_gradient(tf.identity(gradients),
                                     v12.get_log_value(),
                                     v34.get_log_value(),
                                     v12.get_log_value(),
                                     v5.get_log_value())
        batch_size = 100
        gradients_feed = np.random.rand(batch_size, 1)
        v12_feed = np.random.rand(batch_size, 8)
        v34_feed = np.random.rand(batch_size, 2)
        v5_feed = np.random.rand(batch_size, 1)

        with tf.Session() as sess:
            out = sess.run(op, feed_dict={gradients: gradients_feed,
                                          v12: v12_feed,
                                          v34: v34_feed,
                                          v5: v5_feed})

        # Calculate true outputs
        input_values = np.hstack([np.expand_dims(v12_feed[:, 0], axis=1),
                                  np.expand_dims(v12_feed[:, 5], axis=1),
                                  v34_feed,
                                  np.expand_dims(v12_feed[:, 3], axis=1),
                                  v5_feed])
        inputs_reduce_prod = np.prod(input_values, axis=1, keepdims=True)
        output_gradients = (inputs_reduce_prod * gradients_feed) / input_values
        output_gradients = np.split(output_gradients, [2, 4, 5, 6], axis=1)
        output_gradients_0 = np.zeros((batch_size, 8))
        output_gradients_0[:, 0] = output_gradients[0][:, 0]
        output_gradients_0[:, 5] = output_gradients[0][:, 1]
        output_gradients[0] = output_gradients_0
        output_gradients_2 = np.zeros((batch_size, 8))
        output_gradients_2[:, 3] = output_gradients[2][:, 0]
        output_gradients[2] = output_gradients_2

        np.testing.assert_array_almost_equal(
            out[0], output_gradients[0])
        np.testing.assert_array_almost_equal(
            out[1], output_gradients[1])
        np.testing.assert_array_almost_equal(
            out[2], output_gradients[2])
        np.testing.assert_array_almost_equal(
            out[3], output_gradients[3])


if __name__ == '__main__':
    tf.test.main()
