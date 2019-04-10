#!/usr/bin/env python3

from context import libspn as spn
from test import TestCase
import tensorflow as tf
import numpy as np


class TestGraphConcat(TestCase):

    def test_value(self):
        """Calculating value of Concat"""

        def test(inputs, feed, output):
            with self.subTest(inputs=inputs, feed=feed):
                n = spn.Concat(*inputs)
                op = n.get_value(spn.InferenceType.MARGINAL)
                op_log = n.get_log_value(spn.InferenceType.MARGINAL)
                op_mpe = n.get_value(spn.InferenceType.MPE)
                op_log_mpe = n.get_log_value(spn.InferenceType.MPE)
                with self.test_session() as sess:
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
        v1 = spn.RawLeaf(num_vars=3)
        v2 = spn.RawLeaf(num_vars=5)
        v3 = spn.RawLeaf(num_vars=1)

        # Multiple inputs, indices specified
        test([(v1, [0, 2]),
              (v2, [1])],
             {v1: [[1, 2, 3],
                   [4, 5, 6]],
              v2: [[7, 8, 9, 10, 11],
                   [12, 13, 14, 15, 16]]},
             [[1.0, 3.0, 8.0],
              [4.0, 6.0, 13.0]])

        # Single input, indices specified
        test([(v1, [0, 1, 2])],
             {v1: [[1, 2, 3],
                   [4, 5, 6]]},
             [[1.0, 2.0, 3.0],
              [4.0, 5.0, 6.0]])

        # Single input, no indices
        test([v1],
             {v1: [[1, 2, 3],
                   [4, 5, 6]]},
             [[1.0, 2.0, 3.0],
              [4.0, 5.0, 6.0]])

        # Single input with 1 value, no indices
        test([v3],
             {v3: [[17],
                   [18]]},
             [[17.0],
              [18.0]])

        # Multiple inputs, no indices
        test([v1, v2],
             {v1: [[1, 2, 3],
                   [4, 5, 6]],
              v2: [[7, 8, 9, 10, 11],
                   [12, 13, 14, 15, 16]]},
             [[1.0, 2.0, 3.0, 7.0, 8.0, 9.0, 10.0, 11.0],
              [4.0, 5.0, 6.0, 12.0, 13.0, 14.0, 15.0, 16.0]])

        # Mixed
        test([v1, (v2, [2, 3]), v3],
             {v1: [[1, 2, 3],
                   [4, 5, 6]],
              v2: [[7, 8, 9, 10, 11],
                   [12, 13, 14, 15, 16]],
              v3: [[17],
                   [18]]},
             [[1.0, 2.0, 3.0, 9.0, 10.0, 17.0],
              [4.0, 5.0, 6.0, 14.0, 15.0, 18.0]])

        # One-element batch
        test([(v1, [0, 2]),
              (v2, [1])],
             {v1: [[1, 2, 3]],
              v2: [[7, 8, 9, 10, 11]]},
             [[1.0, 3.0, 8.0]])

    def test_compute_mpe_path(self):
        v12 = spn.IndicatorLeaf(num_vars=2, num_vals=4)
        v34 = spn.RawLeaf(num_vars=2)
        v5 = spn.RawLeaf(num_vars=1)
        p = spn.Concat((v12, [0, 5]), v34, (v12, [3]), v5)
        counts = tf.placeholder(tf.float32, shape=(None, 6))
        op = p._compute_log_mpe_path(tf.identity(counts),
                                     v12.get_value(),
                                     v34.get_value(),
                                     v12.get_value(),
                                     v5.get_value())
        feed = np.r_[:18].reshape(-1, 6)
        with self.test_session() as sess:
            out = sess.run(op, feed_dict={counts: feed})
        np.testing.assert_array_almost_equal(
            out[0], np.array([[0., 0., 0., 0., 0., 1., 0., 0.],
                              [6., 0., 0., 0., 0., 7., 0., 0.],
                              [12., 0., 0., 0., 0., 13., 0., 0.]],
                             dtype=np.float32))
        np.testing.assert_array_almost_equal(
            out[1], np.array([[2., 3.],
                              [8., 9.],
                              [14., 15.]],
                             dtype=np.float32))
        np.testing.assert_array_almost_equal(
            out[2], np.array([[0., 0., 0., 4., 0., 0., 0., 0.],
                              [0., 0., 0., 10., 0., 0., 0., 0.],
                              [0., 0., 0., 16., 0., 0., 0., 0.]],
                             dtype=np.float32))
        np.testing.assert_array_almost_equal(
            out[3], np.array([[5.],
                              [11.],
                              [17.]],
                             dtype=np.float32))


if __name__ == '__main__':
    tf.test.main()
