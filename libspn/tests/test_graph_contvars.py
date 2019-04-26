#!/usr/bin/env python3

from context import libspn as spn
from test import TestCase
import tensorflow as tf
import numpy as np


class TestGraphRawLeaf(TestCase):

    def test_contvars_value_feed_dict(self):
        """Calculating value of RawLeaf based on inputs provided using feed_dict"""

        def test(num_vars, value):
            with self.subTest(num_vars=num_vars, value=value):
                n = spn.RawLeaf(num_vars=num_vars)
                op = n.get_value()
                op_log = n.get_log_value()
                with self.test_session() as sess:
                    out = sess.run(op, feed_dict={n: value})
                    out_log = sess.run(tf.exp(op_log), feed_dict={n: value})
                np.testing.assert_array_almost_equal(
                    out,
                    np.array(value, dtype=spn.conf.dtype.as_numpy_dtype()))
                np.testing.assert_array_almost_equal(
                    out_log,
                    np.array(value, dtype=spn.conf.dtype.as_numpy_dtype()))

        # Run tests
        test(1, [[1]])
        test(1, [[1],
                 [2]])
        test(2, [[1, 2]])
        test(2, [[1, 2],
                 [3, 4]])

    def test_contvars_value_tensor(self):
        """Calculating value of RawLeaf based on inputs from a tensor"""

        def test(num_vars, value):
            with self.subTest(num_vars=num_vars, value=value):
                p = tf.placeholder(spn.conf.dtype, [None, num_vars])
                n = spn.RawLeaf(feed=p, num_vars=num_vars)
                op = n.get_value()
                op_log = n.get_log_value()
                with self.test_session() as sess:
                    out = sess.run(op, feed_dict={p: value})
                    out_log = sess.run(tf.exp(op_log), feed_dict={p: value})
                np.testing.assert_array_almost_equal(
                    out,
                    np.array(value, dtype=spn.conf.dtype.as_numpy_dtype()))
                np.testing.assert_array_almost_equal(
                    out_log,
                    np.array(value, dtype=spn.conf.dtype.as_numpy_dtype()))

        # Run tests
        test(1, [[1]])
        test(1, [[1],
                 [2]])
        test(2, [[1, 2]])
        test(2, [[1, 2],
                 [3, 4]])


if __name__ == '__main__':
    tf.test.main()
