#!/usr/bin/env python3

# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

import tensorflow as tf
import numpy as np
from context import libspn as spn


class TestGraphWeights(tf.test.TestCase):

    def test_single_initialization(self):
        """Single weights node initialization"""
        w1 = spn.Weights(3, num_weights=2)
        w2 = spn.Weights(0.3, num_weights=4)
        w3 = spn.Weights([0.4, 0.4, 1.2], num_weights=3)
        init1 = w1.initialize()
        init2 = w2.initialize()
        init3 = w3.initialize()
        with tf.Session() as sess:
            sess.run([init1, init2, init3])
            val1 = sess.run(w1.get_value())
            val2 = sess.run(w2.get_value())
            val3 = sess.run(w3.get_value())
            val1_log = sess.run(tf.exp(w1.get_log_value()))
            val2_log = sess.run(tf.exp(w2.get_log_value()))
            val3_log = sess.run(tf.exp(w3.get_log_value()))

        self.assertEqual(val1.dtype, spn.conf.dtype.as_numpy_dtype())
        self.assertEqual(val2.dtype, spn.conf.dtype.as_numpy_dtype())
        self.assertEqual(val3.dtype, spn.conf.dtype.as_numpy_dtype())
        np.testing.assert_array_almost_equal(val1, [0.5, 0.5])
        np.testing.assert_array_almost_equal(val2, [0.25, 0.25, 0.25, 0.25])
        np.testing.assert_array_almost_equal(val3, [0.2, 0.2, 0.6])
        self.assertEqual(val1_log.dtype, spn.conf.dtype.as_numpy_dtype())
        self.assertEqual(val2_log.dtype, spn.conf.dtype.as_numpy_dtype())
        self.assertEqual(val3_log.dtype, spn.conf.dtype.as_numpy_dtype())
        np.testing.assert_array_almost_equal(val1_log, [0.5, 0.5])
        np.testing.assert_array_almost_equal(val2_log, [0.25, 0.25, 0.25, 0.25])
        np.testing.assert_array_almost_equal(val3_log, [0.2, 0.2, 0.6])

    def test_single_assignment(self):
        """Single weights node assignment"""
        w1 = spn.Weights(3, num_weights=2)
        w2 = spn.Weights(0.3, num_weights=4)
        w3 = spn.Weights([0.4, 0.4, 1.2], num_weights=3)
        init1 = w1.initialize()
        # init2 = w2.initialize()  # don't initialize for testing
        init3 = w3.initialize()
        assign1 = w1.assign([1.0, 3.0])
        assign2 = w2.assign(0.5)
        assign3 = w3.assign(5)
        with tf.Session() as sess:
            sess.run([init1, init3])
            sess.run([assign1, assign2, assign3])
            val1 = sess.run(w1.get_value())
            val2 = sess.run(w2.get_value())
            val3 = sess.run(w3.get_value())
            val1_log = sess.run(tf.exp(w1.get_log_value()))
            val2_log = sess.run(tf.exp(w2.get_log_value()))
            val3_log = sess.run(tf.exp(w3.get_log_value()))

        self.assertEqual(val1.dtype, spn.conf.dtype.as_numpy_dtype())
        self.assertEqual(val2.dtype, spn.conf.dtype.as_numpy_dtype())
        self.assertEqual(val3.dtype, spn.conf.dtype.as_numpy_dtype())
        np.testing.assert_array_almost_equal(val1, [0.25, 0.75])
        np.testing.assert_array_almost_equal(val2, [0.25, 0.25, 0.25, 0.25])
        np.testing.assert_array_almost_equal(val3, [1 / 3, 1 / 3, 1 / 3])
        self.assertEqual(val1_log.dtype, spn.conf.dtype.as_numpy_dtype())
        self.assertEqual(val2_log.dtype, spn.conf.dtype.as_numpy_dtype())
        self.assertEqual(val3_log.dtype, spn.conf.dtype.as_numpy_dtype())
        np.testing.assert_array_almost_equal(val1_log, [0.25, 0.75])
        np.testing.assert_array_almost_equal(val2_log, [0.25, 0.25, 0.25, 0.25])
        np.testing.assert_array_almost_equal(val3_log, [1 / 3, 1 / 3, 1 / 3])

    def test_group_initialization(self):
        """Group initialization of weights nodes"""
        v1 = spn.IVs(num_vars=1, num_vals=2)
        v2 = spn.IVs(num_vars=1, num_vals=4)
        s1 = spn.Sum(v1)
        s1.generate_weights([0.2, 0.3])
        s2 = spn.Sum(v2)
        s2.generate_weights(5)
        p = spn.Product(s1, s2)
        init = spn.initialize_weights(p)

        with tf.Session() as sess:
            sess.run([init])
            val1 = sess.run(s1.weights.node.get_value())
            val2 = sess.run(s2.weights.node.get_value())
            val1_log = sess.run(tf.exp(s1.weights.node.get_log_value()))
            val2_log = sess.run(tf.exp(s2.weights.node.get_log_value()))

        self.assertEqual(val1.dtype, spn.conf.dtype.as_numpy_dtype())
        self.assertEqual(val2.dtype, spn.conf.dtype.as_numpy_dtype())
        np.testing.assert_array_almost_equal(val1, [0.4, 0.6])
        np.testing.assert_array_almost_equal(val2, [0.25, 0.25, 0.25, 0.25])
        self.assertEqual(val1_log.dtype, spn.conf.dtype.as_numpy_dtype())
        self.assertEqual(val2_log.dtype, spn.conf.dtype.as_numpy_dtype())
        np.testing.assert_array_almost_equal(val1_log, [0.4, 0.6])
        np.testing.assert_array_almost_equal(val2_log, [0.25, 0.25, 0.25, 0.25])

    def test_group_assignment(self):
        """Group assignment of weights nodes"""
        v1 = spn.IVs(num_vars=1, num_vals=2)
        v2 = spn.IVs(num_vars=1, num_vals=4)
        s1 = spn.Sum(v1)
        s1.generate_weights([0.2, 0.3])
        s2 = spn.Sum(v2)
        s2.generate_weights(5)
        p = spn.Product(s1, s2)
        init1 = s1.weights.node.initialize()
        assign = spn.assign_weights(p, 0.2)

        with tf.Session() as sess:
            sess.run([init1])
            val1i = sess.run(s1.weights.node.get_value())
            val1i_log = sess.run(tf.exp(s1.weights.node.get_log_value()))
            sess.run([assign])
            val1 = sess.run(s1.weights.node.get_value())
            val2 = sess.run(s2.weights.node.get_value())
            val1_log = sess.run(tf.exp(s1.weights.node.get_log_value()))
            val2_log = sess.run(tf.exp(s2.weights.node.get_log_value()))

        self.assertEqual(val1i.dtype, spn.conf.dtype.as_numpy_dtype())
        self.assertEqual(val1.dtype, spn.conf.dtype.as_numpy_dtype())
        self.assertEqual(val2.dtype, spn.conf.dtype.as_numpy_dtype())
        np.testing.assert_array_almost_equal(val1i, [0.4, 0.6])
        np.testing.assert_array_almost_equal(val1, [0.5, 0.5])
        np.testing.assert_array_almost_equal(val2, [0.25, 0.25, 0.25, 0.25])
        self.assertEqual(val1i_log.dtype, spn.conf.dtype.as_numpy_dtype())
        self.assertEqual(val1_log.dtype, spn.conf.dtype.as_numpy_dtype())
        self.assertEqual(val2_log.dtype, spn.conf.dtype.as_numpy_dtype())
        np.testing.assert_array_almost_equal(val1i_log, [0.4, 0.6])
        np.testing.assert_array_almost_equal(val1_log, [0.5, 0.5])
        np.testing.assert_array_almost_equal(val2_log, [0.25, 0.25, 0.25, 0.25])


if __name__ == '__main__':
    tf.test.main()
