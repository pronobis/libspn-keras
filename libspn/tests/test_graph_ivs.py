#!/usr/bin/env python3

from context import libspn as spn
from test import TestCase
import tensorflow as tf
import numpy as np


class TestGraphIndicatorLeaf(TestCase):

    def test_iv_value_feed_dict(self):
        """Calculating value of IndicatorLeaf based on inputs provided using feed_dict"""

        def test(num_vars, num_vals, rv_value, iv_value):
            with self.subTest(num_vars=num_vars, num_vals=num_vals,
                              rv_value=rv_value):
                n = spn.IndicatorLeaf(num_vars=num_vars, num_vals=num_vals)
                op = n.get_value()
                op_log = n.get_log_value()
                with self.test_session() as sess:
                    out = sess.run(op, feed_dict={n: rv_value})
                    out_log = sess.run(tf.exp(op_log), feed_dict={n: rv_value})
                np.testing.assert_array_almost_equal(
                    out,
                    np.array(iv_value, dtype=spn.conf.dtype.as_numpy_dtype()))
                np.testing.assert_array_almost_equal(
                    out_log,
                    np.array(iv_value, dtype=spn.conf.dtype.as_numpy_dtype()))

        # Run tests
        test(1, 3,
             [[1]],
             [[0, 1, 0]])
        test(1, 3,
             [[-1]],
             [[1, 1, 1]])
        test(1, 2,
             [[0],
              [1],
              [0],
              [-1]],
             [[1, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
        test(3, 4,
             [[0, 2, 1]],
             [[1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0]])
        test(3, 4,
             [[0, -1, -1]],
             [[1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]])
        test(2, 4,
             [[0, 2],
              [0, 0],
              [1, 3],
              [-1, 3],
              [2, -1]],
             [[1, 0, 0, 0, 0, 0, 1, 0],
              [1, 0, 0, 0, 1, 0, 0, 0],
              [0, 1, 0, 0, 0, 0, 0, 1],
              [1, 1, 1, 1, 0, 0, 0, 1],
              [0, 0, 1, 0, 1, 1, 1, 1]])

    def test_iv_value_tensor(self):
        """Calculating value of IndicatorLeaf based on inputs from a tensor"""

        def test(num_vars, num_vals, rv_value, iv_value):
            with self.subTest(num_vars=num_vars, num_vals=num_vals,
                              rv_value=rv_value):
                p = tf.placeholder(tf.int32, [None, num_vars])
                n = spn.IndicatorLeaf(feed=p, num_vars=num_vars,
                            num_vals=num_vals)
                op = n.get_value()
                op_log = n.get_log_value()
                with self.test_session() as sess:
                    out = sess.run(op, feed_dict={p: rv_value})
                    out_log = sess.run(tf.exp(op_log), feed_dict={p: rv_value})
                np.testing.assert_array_almost_equal(
                    out,
                    np.array(iv_value, dtype=spn.conf.dtype.as_numpy_dtype()))
                np.testing.assert_array_almost_equal(
                    out_log,
                    np.array(iv_value, dtype=spn.conf.dtype.as_numpy_dtype()))

        # Run tests
        test(1, 3,
             [[1]],
             [[0, 1, 0]])
        test(1, 3,
             [[-1]],
             [[1, 1, 1]])
        test(1, 2,
             [[0],
              [1],
              [0],
              [-1]],
             [[1, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
        test(3, 4,
             [[0, 2, 1]],
             [[1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0]])
        test(3, 4,
             [[0, -1, -1]],
             [[1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]])
        test(2, 4,
             [[0, 2],
              [0, 0],
              [1, 3],
              [-1, 3],
              [2, -1]],
             [[1, 0, 0, 0, 0, 0, 1, 0],
              [1, 0, 0, 0, 1, 0, 0, 0],
              [0, 1, 0, 0, 0, 0, 0, 1],
              [1, 1, 1, 1, 0, 0, 0, 1],
              [0, 0, 1, 0, 1, 1, 1, 1]])


if __name__ == '__main__':
    tf.test.main()
