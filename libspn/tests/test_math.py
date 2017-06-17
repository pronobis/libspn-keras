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
import collections


class TestMath(TestCase):

    def test_gather_cols_errors(self):
        # Should work
        spn.utils.gather_cols(tf.constant([10, 11, 12]),
                              [0, 1, 2])
        spn.utils.gather_cols(tf.constant([[10, 11, 12]]),
                              [0, 1, 2])
        spn.utils.gather_cols(tf.placeholder(tf.float32,
                                             shape=(None, 3)),
                              [0, 1, 2])

        # Param size defined
        with self.assertRaises(RuntimeError):
            spn.utils.gather_cols(tf.placeholder(tf.float32,
                                                 shape=(None, None)),
                                  [0, 1, 2])
        # Param dim number
        with self.assertRaises(ValueError):
            spn.utils.gather_cols(tf.constant(10),
                                  [0, 1, 2])
        with self.assertRaises(ValueError):
            spn.utils.gather_cols(tf.constant([[[10, 11, 12]]]),
                                  [0, 1, 2])
        # Index dims
        with self.assertRaises(ValueError):
            spn.utils.gather_cols(tf.constant([10, 11, 12]),
                                  [[0, -1, 2]])
        with self.assertRaises(ValueError):
            spn.utils.gather_cols(tf.constant([10, 11, 12]),
                                  1)
        # Indices empty
        with self.assertRaises(ValueError):
            spn.utils.gather_cols(tf.constant([10, 11, 12]),
                                  [])
        # Index values
        with self.assertRaises(ValueError):
            spn.utils.gather_cols(tf.constant([10, 11, 12]),
                                  [0, -1, 2])
        with self.assertRaises(ValueError):
            spn.utils.gather_cols(tf.constant([10, 11, 12]),
                                  [0, 3, 2])
        with self.assertRaises(ValueError):
            spn.utils.gather_cols(tf.constant([10, 11, 12]),
                                  [0.1, 3, 2])

    def test_gather_cols(self):
        def test(params, indices, true_output,
                 params_dtype, indices_dtype, on_gpu):
            with self.subTest(params=params, indices=indices,
                              params_dtype=params_dtype,
                              indices_dtype=indices_dtype,
                              on_gpu=on_gpu):
                tf.reset_default_graph()
                with self.test_session(force_gpu=on_gpu) as sess:
                    # Indices
                    indices = np.asarray(indices, dtype=indices_dtype)
                    # Params
                    p1d = tf.constant(params, dtype=params_dtype)
                    p2d1 = tf.constant(np.array([np.array(params)]),
                                       dtype=params_dtype)
                    p2d2 = tf.constant(np.array([np.array(params),
                                                 np.array(params) * 2,
                                                 np.array(params) * 3]),
                                       dtype=params_dtype)
                    # Define ops for different implementations
                    custom_gather_cols = spn.conf.custom_gather_cols
                    spn.conf.custom_gather_cols = False
                    op1dn = spn.utils.gather_cols(p1d, indices)
                    op2d1n = spn.utils.gather_cols(p2d1, indices)
                    op2d2n = spn.utils.gather_cols(p2d2, indices)
                    spn.conf.custom_gather_cols = True
                    op1dc = spn.utils.gather_cols(p1d, indices)
                    op2d1c = spn.utils.gather_cols(p2d1, indices)
                    op2d2c = spn.utils.gather_cols(p2d2, indices)
                    spn.conf.custom_gather_cols = custom_gather_cols
                    # Run
                    out1dn = sess.run(op1dn)
                    out1dc = sess.run(op1dc)
                    out2d1n = sess.run(op2d1n)
                    out2d1c = sess.run(op2d1c)
                    out2d2n = sess.run(op2d2n)
                    out2d2c = sess.run(op2d2c)
                # Compare
                np.testing.assert_array_almost_equal(out1dn, true_output)
                np.testing.assert_array_almost_equal(out1dc, true_output)
                self.assertEqual(params_dtype.as_numpy_dtype, out1dn.dtype)
                self.assertEqual(params_dtype.as_numpy_dtype, out1dc.dtype)
                true_output_2d1 = [np.array(true_output)]
                true_output_2d2 = [np.array(true_output),
                                   np.array(true_output) * 2,
                                   np.array(true_output) * 3]
                np.testing.assert_array_almost_equal(out2d1n, true_output_2d1)
                np.testing.assert_array_almost_equal(out2d1c, true_output_2d1)
                np.testing.assert_array_almost_equal(out2d2n, true_output_2d2)
                np.testing.assert_array_almost_equal(out2d2c, true_output_2d2)
                self.assertEqual(params_dtype.as_numpy_dtype, out2d1n.dtype)
                self.assertEqual(params_dtype.as_numpy_dtype, out2d1c.dtype)
                self.assertEqual(params_dtype.as_numpy_dtype, out2d2n.dtype)
                self.assertEqual(params_dtype.as_numpy_dtype, out2d2c.dtype)

        def test_all_dtypes(params, indices, true_output):
            # CPU
            test(params, indices, true_output, tf.float32, np.int32, False)
            test(params, indices, true_output, tf.float32, np.int64, False)
            test(params, indices, true_output, tf.float64, np.int32, False)
            test(params, indices, true_output, tf.float64, np.int64, False)
            # GPU
            test(params, indices, true_output, tf.float32, np.int32, True)
            test(params, indices, true_output, tf.float32, np.int64, True)
            test(params, indices, true_output, tf.float64, np.int32, True)
            test(params, indices, true_output, tf.float64, np.int64, True)

        # Single column input tensor
        test_all_dtypes([10],
                        [0],
                        [10.0])

        # Single index
        test_all_dtypes([10, 11, 12],
                        [1],
                        [11.0])

        # Multiple indices
        test_all_dtypes([10, 11, 12],
                        [2, 1, 0],
                        [12.0, 11.0, 10.0])
        test_all_dtypes([10, 11, 12],
                        [0, 2],
                        [10.0, 12.0])

        # Gathering single column tensor should return that tensor directly
        t = tf.constant([10])
        out = spn.utils.gather_cols(t, [0])
        self.assertIs(out, t)
        t = tf.constant([[10],
                         [11]])
        out = spn.utils.gather_cols(t, [0])
        self.assertIs(out, t)

        # Gathering all params in original order should return params tensor
        t = tf.constant([10, 11, 12])
        out = spn.utils.gather_cols(t, [0, 1, 2])
        self.assertIs(out, t)
        t = tf.constant([[10, 11, 12],
                         [13, 14, 15]])
        out = spn.utils.gather_cols(t, [0, 1, 2])
        self.assertIs(out, t)

    def test_scatter_cols_errors(self):
        # Should work
        spn.utils.scatter_cols(tf.constant([10, 11, 12]),
                               [0, 1, 2], 3)
        spn.utils.scatter_cols(tf.constant([[10, 11, 12]]),
                               [0, 1, 2], 3)
        spn.utils.scatter_cols(tf.placeholder(tf.float32,
                                              shape=(None, 3)),
                               [0, 1, 2], 3)

        # Param size defined
        with self.assertRaises(RuntimeError):
            spn.utils.scatter_cols(tf.placeholder(tf.float32,
                                                  shape=(None, None)),
                                   [0, 1, 2], 3)
        # Param dim number
        with self.assertRaises(ValueError):
            spn.utils.scatter_cols(tf.constant(10),
                                   [0, 1, 2], 3)
        with self.assertRaises(ValueError):
            spn.utils.scatter_cols(tf.constant([[[10, 11, 12]]]),
                                   [0, 1, 2], 3)
        # num_out_cols type
        with self.assertRaises(ValueError):
            spn.utils.scatter_cols(tf.constant([10, 11, 12]),
                                   [0, 1, 2], 3.1)
        # num_out_cols value
        with self.assertRaises(ValueError):
            spn.utils.scatter_cols(tf.constant([10, 11, 12]),
                                   [0, 1, 2], 2)
        # Indices dims
        with self.assertRaises(ValueError):
            spn.utils.scatter_cols(tf.constant([10, 11, 12]),
                                   [[0, 1, 2]], 3)
        with self.assertRaises(ValueError):
            spn.utils.scatter_cols(tf.constant([10, 11, 12]),
                                   1, 3)
        # Indices size
        with self.assertRaises(ValueError):
            spn.utils.scatter_cols(tf.constant([10, 11, 12]),
                                   [0, 1, 2, 3], 4)
        with self.assertRaises(ValueError):
            spn.utils.scatter_cols(tf.constant([10, 11, 12]),
                                   [0, 1], 4)
        with self.assertRaises(ValueError):
            spn.utils.scatter_cols(tf.constant([10, 11, 12]),
                                   [], 4)
        # Indices values
        with self.assertRaises(ValueError):
            spn.utils.scatter_cols(tf.constant([10, 11, 12]),
                                   [0.1, 1, 2], 3)
        with self.assertRaises(ValueError):
            spn.utils.scatter_cols(tf.constant([10, 11, 12]),
                                   [0, 1, 3], 3)
        with self.assertRaises(ValueError):
            spn.utils.scatter_cols(tf.constant([10, 11, 12]),
                                   [0, 1, 1], 3)

    def test_scatter_cols(self):
        def test(params, indices, num_out_cols, true_output,
                 params_dtype, indices_dtype, on_gpu):
            with self.subTest(params=params, indices=indices,
                              num_out_cols=num_out_cols,
                              params_dtype=params_dtype,
                              indices_dtype=indices_dtype,
                              on_gpu=on_gpu):
                tf.reset_default_graph()
                with self.test_session(force_gpu=on_gpu) as sess:
                    # Indices
                    indices = np.asarray(indices, dtype=indices_dtype)
                    # Params
                    p1d = tf.constant(params, dtype=params_dtype)
                    p2d1 = tf.constant(np.array([np.array(params)]),
                                       dtype=params_dtype)
                    p2d2 = tf.constant(np.array([np.array(params),
                                                 np.array(params) * 2,
                                                 np.array(params) * 3]),
                                       dtype=params_dtype)
                    # Define ops for different implementations
                    custom_scatter_cols = spn.conf.custom_scatter_cols
                    spn.conf.custom_scatter_cols = False
                    op1dn = spn.utils.scatter_cols(p1d, indices, num_out_cols)
                    op2d1n = spn.utils.scatter_cols(p2d1, indices, num_out_cols)
                    op2d2n = spn.utils.scatter_cols(p2d2, indices, num_out_cols)
                    spn.conf.custom_scatter_cols = True
                    op1dc = spn.utils.scatter_cols(p1d, indices, num_out_cols)
                    op2d1c = spn.utils.scatter_cols(p2d1, indices, num_out_cols)
                    op2d2c = spn.utils.scatter_cols(p2d2, indices, num_out_cols)
                    spn.conf.custom_scatter_cols = custom_scatter_cols
                    # Run
                    out1dn = sess.run(op1dn)
                    out1dc = sess.run(op1dc)
                    out2d1n = sess.run(op2d1n)
                    out2d1c = sess.run(op2d1c)
                    out2d2n = sess.run(op2d2n)
                    out2d2c = sess.run(op2d2c)
                # Compare
                np.testing.assert_array_almost_equal(out1dn, true_output)
                np.testing.assert_array_almost_equal(out1dc, true_output)
                self.assertEqual(params_dtype.as_numpy_dtype, out1dn.dtype)
                self.assertEqual(params_dtype.as_numpy_dtype, out1dc.dtype)
                true_output_2d1 = [np.array(true_output)]
                true_output_2d2 = [np.array(true_output),
                                   np.array(true_output) * 2,
                                   np.array(true_output) * 3]
                np.testing.assert_array_almost_equal(out2d1n, true_output_2d1)
                np.testing.assert_array_almost_equal(out2d1c, true_output_2d1)
                np.testing.assert_array_almost_equal(out2d2n, true_output_2d2)
                np.testing.assert_array_almost_equal(out2d2c, true_output_2d2)
                self.assertEqual(params_dtype.as_numpy_dtype, out2d1n.dtype)
                self.assertEqual(params_dtype.as_numpy_dtype, out2d1c.dtype)
                self.assertEqual(params_dtype.as_numpy_dtype, out2d2n.dtype)
                self.assertEqual(params_dtype.as_numpy_dtype, out2d2c.dtype)

        def test_all_dtypes(params, indices, num_out_cols, true_output):
            # CPU
            test(params, indices, num_out_cols, true_output,
                 tf.float32, np.int32, False)
            test(params, indices, num_out_cols, true_output,
                 tf.float32, np.int64, False)
            test(params, indices, num_out_cols, true_output,
                 tf.float64, np.int32, False)
            test(params, indices, num_out_cols, true_output,
                 tf.float64, np.int64, False)
            # GPU
            test(params, indices, num_out_cols, true_output,
                 tf.float32, np.int32, True)
            test(params, indices, num_out_cols, true_output,
                 tf.float32, np.int64, True)
            test(params, indices, num_out_cols, true_output,
                 tf.float64, np.int32, True)
            test(params, indices, num_out_cols, true_output,
                 tf.float64, np.int64, True)

        # Single column input, single column output
        test_all_dtypes([10],
                        [0],
                        1,
                        [10.0])

        # Multi-column output, single-column input
        test_all_dtypes([10],
                        [1],
                        4,
                        [0.0, 10.0, 0.0, 0.0])

        # Multi-column output, multi-column input
        test_all_dtypes([10, 11, 12],
                        [1, 3, 0],
                        4,
                        [12.0, 10.0, 0.0, 11.0])

        # Pass through if scattering to a single column
        t = tf.constant([10])
        out = spn.utils.scatter_cols(t, [0], 1)
        self.assertIs(out, t)
        t = tf.constant([[10],
                         [11]])
        out = spn.utils.scatter_cols(t, [0], 1)
        self.assertIs(out, t)

        # Pass through if scattering to the output of same size
        # in original index order
        t = tf.constant([10, 11, 12])
        out = spn.utils.scatter_cols(t, [0, 1, 2], 3)
        self.assertIs(out, t)
        t = tf.constant([[10, 11, 12],
                         [13, 14, 15]])
        out = spn.utils.scatter_cols(t, [0, 1, 2], 3)
        self.assertIs(out, t)

    def test_scatter_values(self):
        def test(params, indices, num_out_cols, param_dtype, ind_dtype,
                 true_output, use_gpu=False):

            if use_gpu:
                device = [False, True]
            else:
                device = [False]

            for p_dt in param_dtype:
                for i_dt in ind_dtype:
                    for dev in device:
                        with self.test_session(use_gpu=dev) as sess:
                            row1 = 1
                            row2 = -1
                            row3 = 2

                            # Convert params and output to appropriate data-types
                            if p_dt == tf.float32 or p_dt == tf.float64:
                                par = list(map(float, params))
                                if isinstance(true_output[0], collections.Iterable):
                                    t_out = [list(map(float, to)) for to in
                                             true_output]
                                else:
                                    t_out = list(map(float, true_output))
                            else:
                                par = list(map(int, params))
                                if isinstance(true_output[0], collections.Iterable):
                                    t_out = [list(map(int, to)) for to in
                                             true_output]
                                else:
                                    t_out = list(map(int, true_output))

                            p1d = tf.constant(np.array(par), dtype=p_dt)
                            p2d1 = tf.constant(np.array([np.array(par)]),
                                                        dtype=p_dt)
                            p2d2 = tf.constant(np.array([np.array(par) * row1,
                                                         np.array(par) * row2,
                                                         np.array(par) * row3]),
                                               dtype=p_dt)

                            ind1d = tf.constant(np.array(indices), dtype=i_dt)
                            ind2d1 = tf.constant(np.array([np.array(indices)]),
                                                 dtype=i_dt)
                            ind2d2 = tf.constant(np.array([np.array(indices),
                                                           np.array(indices),
                                                           np.array(indices)]),
                                                 dtype=i_dt)

                            op1d = spn.utils.scatter_values(p1d, ind1d,
                                                            num_out_cols)
                            op2d1 = spn.utils.scatter_values(p2d1, ind2d1,
                                                             num_out_cols)
                            op2d2 = spn.utils.scatter_values(p2d2, ind2d2,
                                                             num_out_cols)

                            out1d = sess.run(op1d)
                            out2d1 = sess.run(op2d1)
                            out2d2 = sess.run(op2d2)

                            # Test outputs
                            np.testing.assert_array_almost_equal(out1d,
                                                       np.array(t_out))
                            self.assertEqual(p_dt.as_numpy_dtype, out1d.dtype)
                            np.testing.assert_array_equal(op1d.get_shape(),
                                                    list(np.array(t_out).shape))

                            t_out_2d1 = [np.array(t_out)]
                            np.testing.assert_array_almost_equal(out2d1,
                                                                 t_out_2d1)
                            self.assertEqual(p_dt.as_numpy_dtype, out2d1.dtype)
                            np.testing.assert_array_equal(op2d1.get_shape(),
                                                list(np.array(t_out_2d1).shape))

                            t_out_2d2 = [np.array(t_out) * row1,
                                         np.array(t_out) * row2,
                                         np.array(t_out) * row3]
                            np.testing.assert_array_almost_equal(out2d2,
                                                      np.array(t_out_2d2))
                            self.assertEqual(p_dt.as_numpy_dtype, out2d2.dtype)
                            np.testing.assert_array_equal(op2d2.get_shape(),
                                                list(np.array(t_out_2d2).shape))

        # Single param, single index
        # Without padding - Only scatter
        test([12.34],
             [0],
             1,
             [tf.float32, tf.float64, tf.int32, tf.int64],
             [tf.int32, tf.int64],
             [[12.34]],
             use_gpu=True)

        # With padding
        test([12.34],
             [1],
             4,
             [tf.float32, tf.float64, tf.int32, tf.int64],
             [tf.int32, tf.int64],
             [[0.0, 12.34, 0.0, 0.0]],
             use_gpu=True)

        # Multiple params, multiple indices
        # Without padding - Only scatter
        test([12.34, 12.34*2, 12.34*3, 12.34*4],
             [0, 0, 0, 0],
             1,
             [tf.float32, tf.float64, tf.int32, tf.int64],
             [tf.int32, tf.int64],
             [[12.34],
              [12.34*2],
              [12.34*3],
              [12.34*4]],
             use_gpu=True)

        # With padding
        test([12.34, 12.34*2, 12.34*3, 12.34*4],
             [1, 4, 2, 0],
             5,
             [tf.float32, tf.float64, tf.int32, tf.int64],
             [tf.int32, tf.int64],
             [[0.0, 12.34, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 12.34*2],
              [0.0, 0.0, 12.34*3, 0.0, 0.0],
              [12.34*4, 0.0, 0.0, 0.0, 0.0]],
             use_gpu=True)

    def test_broadcast_value(self):
        """broadcast_value for various value types"""

        v1 = spn.utils.broadcast_value(spn.ValueType.RANDOM_UNIFORM(0, 1),
                                       (2, 3), dtype=tf.float64)

        v2 = spn.utils.broadcast_value(1,
                                       (2, 3), dtype=tf.float64)

        v3 = spn.utils.broadcast_value(1.0,
                                       (2, 3), dtype=tf.float64)

        v4 = spn.utils.broadcast_value([1],
                                       (2, 3), dtype=tf.float64)

        with self.test_session() as sess:
            out1 = sess.run(v1)
            out2 = sess.run(v2)
            out3 = sess.run(v3)
            out4 = sess.run(v4)
            self.assertEqual(out1.dtype, np.float64)
            self.assertEqual(out2.dtype, np.float64)
            self.assertEqual(out3.dtype, np.float64)
            self.assertEqual(out4.dtype, np.float64)
            self.assertGreaterEqual(out1.min(), 0)
            self.assertLessEqual(out1.max(), 1)
            np.testing.assert_array_almost_equal(out2, [[1.0, 1.0, 1.0],
                                                        [1.0, 1.0, 1.0]])
            np.testing.assert_array_almost_equal(out3, [[1.0, 1.0, 1.0],
                                                        [1.0, 1.0, 1.0]])
            np.testing.assert_array_almost_equal(out4, [1.0])

    def test_normalize_tensor(self):
        """normalize_tensor"""

        v1 = spn.utils.normalize_tensor([1])
        v2 = spn.utils.normalize_tensor(tf.constant([1], dtype=tf.float32))
        v3 = spn.utils.normalize_tensor(tf.constant([1], dtype=tf.float64))
        v4 = spn.utils.normalize_tensor([1.0])
        v5 = spn.utils.normalize_tensor([0.1])
        v6 = spn.utils.normalize_tensor([1, 0.5, 1])
        v7 = spn.utils.normalize_tensor([[1, 1],
                                         [1, 2]])
        v8 = spn.utils.normalize_tensor([0.25, 0.25, 0.25, 0.25])
        v9 = spn.utils.normalize_tensor([[0.25, 0.25],
                                         [0.25, 0.25]])

        with self.test_session() as sess:
            out1 = sess.run(v1)
            out2 = sess.run(v2)
            out3 = sess.run(v3)
            out4 = sess.run(v4)
            out5 = sess.run(v5)
            out6 = sess.run(v6)
            out7 = sess.run(v7)
            out8 = sess.run(v8)
            out9 = sess.run(v9)
            self.assertEqual(out1.dtype, np.float64)
            self.assertEqual(out2.dtype, np.float32)
            self.assertEqual(out3.dtype, np.float64)
            self.assertEqual(out4.dtype, np.float32)
            self.assertEqual(out5.dtype, np.float32)
            self.assertEqual(out6.dtype, np.float32)
            self.assertEqual(out7.dtype, np.float64)
            self.assertEqual(out8.dtype, np.float32)
            self.assertEqual(out9.dtype, np.float32)

            np.testing.assert_array_almost_equal(out1, [1.0])
            np.testing.assert_array_almost_equal(out2, [1.0])
            np.testing.assert_array_almost_equal(out3, [1.0])
            np.testing.assert_array_almost_equal(out4, [1.0])
            np.testing.assert_array_almost_equal(out5, [1.0])
            np.testing.assert_array_almost_equal(out6, [0.4, 0.2, 0.4])
            np.testing.assert_array_almost_equal(out7, [[0.2, 0.2],
                                                        [0.2, 0.4]])
            np.testing.assert_array_almost_equal(out8, [0.25, 0.25, 0.25, 0.25])
            np.testing.assert_array_almost_equal(out9, [[0.25, 0.25],
                                                        [0.25, 0.25]])

    def test_reduce_log_sum(self):

        def test(dtype):
            with self.subTest(dtype=dtype):
                inpt = [[0.0, 0.0, 0.0],
                        [0.0, 0.1, 0.0],
                        [0.1, 0.2, 0.3],
                        [1.0, 0.0, 2.0],
                        [0.0000001, 0.0000002, 0.0000003],
                        [1e-10, 2e-10, 3e-10]]
                inpt_array = np.array(inpt, dtype=dtype.as_numpy_dtype())
                inpt_tensor = tf.constant(inpt, dtype=dtype)
                log_inpt_tensor = tf.log(inpt_tensor)
                op_log = spn.utils.reduce_log_sum(log_inpt_tensor)
                op = tf.exp(op_log)

                with self.test_session() as sess:
                    out = sess.run(op)

                np.testing.assert_array_almost_equal(out,
                                                     np.sum(inpt_array, axis=1,
                                                            keepdims=True))
                self.assertEqual(out.dtype, dtype.as_numpy_dtype())

        test(tf.float32)
        test(tf.float64)

    def test_split_maybe(self):
        value1 = tf.constant(np.r_[:7])
        value2 = tf.constant(np.r_[:21].reshape(-1, 7))
        op1 = spn.utils.split_maybe(value=value1, split_sizes=(1, 3, 1, 2),
                                    axis=0)
        op2 = spn.utils.split_maybe(value=value2, split_sizes=(1, 3, 1, 2),
                                    axis=1)
        op3 = spn.utils.split_maybe(value=value1, split_sizes=(7,), axis=0)
        op4 = spn.utils.split_maybe(value=value2, split_sizes=(7,), axis=1)
        with self.test_session() as sess:
            out1 = sess.run(op1)
            out2 = sess.run(op2)
            out3 = sess.run(op3)
            out4 = sess.run(op4)

        # Test values
        np.testing.assert_array_equal(out1[0], np.array([0]))
        np.testing.assert_array_equal(out1[1], np.array([1, 2, 3]))
        np.testing.assert_array_equal(out1[2], np.array([4]))
        np.testing.assert_array_equal(out1[3], np.array([5, 6]))

        np.testing.assert_array_equal(out2[0], np.array([[0],
                                                         [7],
                                                         [14]]))
        np.testing.assert_array_equal(out2[1], np.array([[1, 2, 3],
                                                         [8, 9, 10],
                                                         [15, 16, 17]]))
        np.testing.assert_array_equal(out2[2], np.array([[4],
                                                         [11],
                                                         [18]]))
        np.testing.assert_array_equal(out2[3], np.array([[5, 6],
                                                         [12, 13],
                                                         [19, 20]]))

        np.testing.assert_array_equal(out3[0], np.array([0, 1, 2, 3, 4, 5, 6]))
        np.testing.assert_array_equal(out4[0], np.array([[0, 1, 2, 3, 4, 5, 6],
                                                         [7, 8, 9, 10, 11, 12, 13],
                                                         [14, 15, 16, 17, 18, 19, 20]]))
        # Test if original tensor returned for 1 split
        self.assertIs(op3[0], value1)
        self.assertIs(op4[0], value2)


if __name__ == '__main__':
    tf.test.main()
