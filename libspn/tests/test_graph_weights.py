#!/usr/bin/env python3

import libspn as spn
import tensorflow as tf
import numpy as np


class TestGraphWeights(tf.test.TestCase):

    def test_single_initialization(self):
        """Single weights node initialization"""
        # Single sum
        w1 = spn.Weights(tf.initializers.constant(3), num_weights=2)
        w2 = spn.Weights(tf.initializers.constant(0.3), num_weights=4)
        w3 = spn.Weights(tf.initializers.constant([0.4, 0.4, 1.2]), num_weights=3)
        # Multi sums
        w4 = spn.Weights(tf.initializers.constant(3), num_weights=2, num_sums=2)
        w5 = spn.Weights(tf.initializers.constant(0.3), num_weights=4, num_sums=3)
        w6 = spn.Weights(tf.initializers.random_uniform(0.0, 1.0), num_weights=1, num_sums=4)
        init1 = w1.initialize()
        init2 = w2.initialize()
        init3 = w3.initialize()
        init4 = w4.initialize()
        init5 = w5.initialize()
        init6 = w6.initialize()
        with self.test_session() as sess:
            sess.run([init1, init2, init3, init4, init5, init6])
            val1 = sess.run(w1.get_value())
            val2 = sess.run(w2.get_value())
            val3 = sess.run(w3.get_value())
            val4 = sess.run(w4.get_value())
            val5 = sess.run(w5.get_value())
            val6 = sess.run(w6.get_value())
            val1_log = sess.run(tf.exp(w1.get_log_value()))
            val2_log = sess.run(tf.exp(w2.get_log_value()))
            val3_log = sess.run(tf.exp(w3.get_log_value()))
            val4_log = sess.run(tf.exp(w4.get_log_value()))
            val5_log = sess.run(tf.exp(w5.get_log_value()))
            val6_log = sess.run(tf.exp(w6.get_log_value()))

        self.assertEqual(val1.dtype, spn.conf.dtype.as_numpy_dtype())
        self.assertEqual(val2.dtype, spn.conf.dtype.as_numpy_dtype())
        self.assertEqual(val3.dtype, spn.conf.dtype.as_numpy_dtype())
        self.assertEqual(val4.dtype, spn.conf.dtype.as_numpy_dtype())
        self.assertEqual(val5.dtype, spn.conf.dtype.as_numpy_dtype())
        self.assertEqual(val6.dtype, spn.conf.dtype.as_numpy_dtype())
        np.testing.assert_array_almost_equal(val1, [[0.5, 0.5]])
        np.testing.assert_array_almost_equal(val2, [[0.25, 0.25, 0.25, 0.25]])
        np.testing.assert_array_almost_equal(val3, [[0.2, 0.2, 0.6]])
        np.testing.assert_array_almost_equal(val4, [[0.5, 0.5], [0.5, 0.5]])
        np.testing.assert_array_almost_equal(val5, [[0.25, 0.25, 0.25, 0.25],
                                                    [0.25, 0.25, 0.25, 0.25],
                                                    [0.25, 0.25, 0.25, 0.25]])
        np.testing.assert_array_almost_equal(val6, [[1.0], [1.0], [1.0], [1.0]])
        self.assertEqual(val1_log.dtype, spn.conf.dtype.as_numpy_dtype())
        self.assertEqual(val2_log.dtype, spn.conf.dtype.as_numpy_dtype())
        self.assertEqual(val3_log.dtype, spn.conf.dtype.as_numpy_dtype())
        np.testing.assert_array_almost_equal(val1_log, [[0.5, 0.5]])
        np.testing.assert_array_almost_equal(val2_log, [[0.25, 0.25, 0.25, 0.25]])
        np.testing.assert_array_almost_equal(val3_log, [[0.2, 0.2, 0.6]])
        np.testing.assert_array_almost_equal(val4_log, [[0.5, 0.5], [0.5, 0.5]])
        np.testing.assert_array_almost_equal(val5_log, [[0.25, 0.25, 0.25, 0.25],
                                                        [0.25, 0.25, 0.25, 0.25],
                                                        [0.25, 0.25, 0.25, 0.25]])
        np.testing.assert_array_almost_equal(val6_log, [[1.0], [1.0], [1.0], [1.0]])

    def test_group_initialization(self):
        """Group initialization of weights nodes"""
        v1 = spn.IndicatorLeaf(num_vars=1, num_vals=2)
        v2 = spn.IndicatorLeaf(num_vars=1, num_vals=4)
        v3 = spn.IndicatorLeaf(num_vars=1, num_vals=2)
        v4 = spn.IndicatorLeaf(num_vars=1, num_vals=2)
        # Sum
        s1 = spn.Sum(v1)
        s1.generate_weights(tf.initializers.constant([0.2, 0.3]))
        s2 = spn.Sum(v2)
        s2.generate_weights(tf.initializers.constant(5))
        # ParallelSums
        s3 = spn.ParallelSums(*[v3, v4], num_sums=2)
        s3.generate_weights(tf.initializers.constant([0.1, 0.2, 0.3, 0.4, 0.4, 0.3, 0.2, 0.1]))
        s4 = spn.ParallelSums(*[v1, v2, v3, v4], num_sums=3)
        s4.generate_weights(tf.initializers.constant(2.0))
        # Product
        p = spn.Product(s1, s2, s3, s4)
        init = spn.initialize_weights(p)

        with self.test_session() as sess:
            sess.run([init])
            val1 = sess.run(s1.weights.node.get_value())
            val2 = sess.run(s2.weights.node.get_value())
            val3 = sess.run(s3.weights.node.get_value())
            val4 = sess.run(s4.weights.node.get_value())
            val1_log = sess.run(tf.exp(s1.weights.node.get_log_value()))
            val2_log = sess.run(tf.exp(s2.weights.node.get_log_value()))
            val3_log = sess.run(tf.exp(s3.weights.node.get_log_value()))
            val4_log = sess.run(tf.exp(s4.weights.node.get_log_value()))

        self.assertEqual(val1.dtype, spn.conf.dtype.as_numpy_dtype())
        self.assertEqual(val2.dtype, spn.conf.dtype.as_numpy_dtype())
        self.assertEqual(val3.dtype, spn.conf.dtype.as_numpy_dtype())
        self.assertEqual(val4.dtype, spn.conf.dtype.as_numpy_dtype())
        np.testing.assert_array_almost_equal(val1, [[0.4, 0.6]])
        np.testing.assert_array_almost_equal(val2, [[0.25, 0.25, 0.25, 0.25]])
        np.testing.assert_array_almost_equal(val3, [[0.1, 0.2, 0.3, 0.4],
                                                    [0.4, 0.3, 0.2, 0.1]])
        np.testing.assert_array_almost_equal(val4, [[0.1] * 10,
                                                    [0.1] * 10,
                                                    [0.1] * 10])
        self.assertEqual(val1_log.dtype, spn.conf.dtype.as_numpy_dtype())
        self.assertEqual(val2_log.dtype, spn.conf.dtype.as_numpy_dtype())
        self.assertEqual(val3_log.dtype, spn.conf.dtype.as_numpy_dtype())
        self.assertEqual(val4_log.dtype, spn.conf.dtype.as_numpy_dtype())
        np.testing.assert_array_almost_equal(val1_log, [[0.4, 0.6]])
        np.testing.assert_array_almost_equal(val2_log, [[0.25, 0.25, 0.25, 0.25]])
        np.testing.assert_array_almost_equal(val3, [[0.1, 0.2, 0.3, 0.4],
                                                    [0.4, 0.3, 0.2, 0.1]])
        np.testing.assert_array_almost_equal(val4, [[0.1] * 10,
                                                    [0.1] * 10,
                                                    [0.1] * 10])

