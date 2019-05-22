#!/usr/bin/env python3

import libspn as spn
import tensorflow as tf
import numpy as np
import collections
from random import shuffle


def _broadcast_to_2D(test_inputs, subset_indices=None, n_stack=2):
    # Subset indices is either specified or set to [0, ..., len(inputs)-1]
    subset_indices = subset_indices or list(range(len(test_inputs[0])))
    ret = []
    for test_input in test_inputs:
        # Append a tuple with n_stack repetitions if the index of the original element index is in
        # subset_indices
        ret.append(tuple(np.asarray(n_stack*[elem]) if ind in subset_indices else elem
                         for ind, elem in enumerate(test_input)))
        ret.append(test_input)
    return ret


class TestMath(tf.test.TestCase):


    def test_logmatmul(self):
        a = tf.random_uniform(shape=(8, 150))
        b = tf.random_uniform(shape=(150, 9))

        ab_linear = tf.matmul(a, b)
        ab_log = tf.exp(spn.utils.logmatmul(tf.log(a), tf.log(b)))

        with self.test_session() as sess:
            ab_linear_out, ab_log_out = sess.run([ab_linear, ab_log])

        self.assertAllClose(ab_linear_out, ab_log_out)

    def test_gather_columns_3d_not_padded(self):
        def assert_output(params, indices, params_dtype, output, output_shape):
            # Assert Output values, shape and dtype
            true_output = (params[indices] if len(params.shape) == 1
                           else params[:, indices])

            np.testing.assert_array_almost_equal(output,
                                                 np.array(true_output))
            self.assertEqual(params_dtype.as_numpy_dtype, output.dtype)
            np.testing.assert_array_equal(output_shape,
                                          list(np.array(true_output).shape))

        def test(params_shape, indices_shape, param_dtype, ind_dtype, use_gpu=False):

            if use_gpu:
                device = [False, True]
            else:
                device = [False]

            if len(params_shape) == 1:
                params_cols = params_shape[0]
            else:
                params_cols = params_shape[1]

            for p_dt in param_dtype:
                for i_dt in ind_dtype:
                    for dev in device:
                        with self.test_session(use_gpu=dev) as sess:
                            # Generate random params array
                            params = np.random.randint(100, size=params_shape)
                            # Convert params to appropriate data-types
                            params = np.array(params, dtype=p_dt.as_numpy_dtype)
                            # Create params tensor
                            params_tensor = tf.constant(params, dtype=p_dt)

                            # Random indices
                            random_indices = np.random.randint(params_cols,
                                                               size=indices_shape,
                                                               dtype=i_dt)
                            # Arange indices
                            if len(indices_shape) == 1:
                                arange_indices = np.arange(0, params_cols, dtype=i_dt)
                            else:
                                arange_indices = np.array([np.arange(0, params_cols) for
                                                           _ in range(indices_shape[0])],
                                                          dtype=i_dt)

                            # Create Ops
                            op_rand_ind = spn.utils.gather_cols_3d(params_tensor,
                                                                   random_indices)
                            op_arange_ind = spn.utils.gather_cols_3d(params_tensor,
                                                                     arange_indices)

                            # Execute Sessions
                            output_rand_ind = sess.run(op_rand_ind)
                            output_arange_ind = sess.run(op_arange_ind)

                            # Test Output
                            assert_output(params, random_indices, p_dt, output_rand_ind,
                                          op_rand_ind.get_shape())
                            assert_output(params, arange_indices, p_dt, output_arange_ind,
                                          op_arange_ind.get_shape())

        # List of params shapes
        params_shapes = [(1, ),   # Single params
                         (1, 1),  # 2D params with single row and column
                         (6, ),   # 1D params
                         (3, 1),  # 2D params with single column
                         (1, 6),  # 2D params with single row
                         (3, 6)]  # 2D params with multiple rows and columns

        # List of indices shapes
        indices_shapes = [(1, ),   # Single index
                          (1, 1),  # 2D indices with single row and column
                          (4, ),   # 1D indices
                          (4, 1),  # 2D indices with single column
                          (1, 5),  # 2D indices with single row
                          (4, 5)]  # 2D indices with multiple rows and columns

        # All combination of test cases for gather_cols_3d without padding
        for p_shape in params_shapes:
            for i_shape in indices_shapes:
                test(params_shape=p_shape, indices_shape=i_shape,
                     param_dtype=[tf.float32, tf.float64, tf.int32, tf.int64],
                     ind_dtype=[np.int32, np.int64],
                     use_gpu=True)

    def test_gather_columns_3d_padded(self):
        def test(params_shape, indices_shape, param_dtype, ind_dtype,
                 pad_elem=0, use_gpu=False):

            if use_gpu:
                device = [False, True]
            else:
                device = [False]

            if len(params_shape) == 1:
                params_rows = 1
                params_cols = params_shape[0]
            else:
                params_rows = params_shape[0]
                params_cols = params_shape[1]

            if len(indices_shape) == 1:
                indices_rows = 1
                indices_cols = indices_shape[0]
            else:
                indices_rows = indices_shape[0]
                indices_cols = indices_shape[1]

            for p_dt in param_dtype:
                for i_dt in ind_dtype:
                    for dev in device:
                        with self.test_session(use_gpu=dev) as sess:

                            # Generate random params array
                            params = np.random.randint(100, size=params_shape)
                            # Convert params to appropriate data-types
                            params = np.array(params, dtype=p_dt.as_numpy_dtype)
                            # Create params tensor
                            params_tensor = tf.constant(params, dtype=p_dt)

                            # Generate a list of 1D indices arrays, with random
                            # length ranging between [1, indices-column-size)
                            indices = []
                            ind_length = indices_cols
                            for i in range(indices_rows):
                                indices.append(np.random.randint(params_cols,
                                                                 size=ind_length,
                                                                 dtype=i_dt))
                                ind_length = np.random.randint(1, indices_cols)
                            # Shuffle indices list
                            shuffle(indices)

                            # Create Ops
                            op = spn.utils.gather_cols_3d(params_tensor, indices,
                                                          pad_elem=pad_elem)

                            # Execute session
                            output = sess.run(op)

                            # Insert a column of zeros to the last column of params
                            params_with_zero = \
                                np.insert(params, params_cols,
                                          np.ones(params_rows,
                                                  dtype=p_dt.as_numpy_dtype)*pad_elem,
                                          axis=-1)

                            # Fill indices of padded columns with index of the
                            # last-column of params
                            indices = [np.insert(ind, ind.size,
                                                 np.full((indices_cols-ind.size),
                                                         params_cols, dtype=i_dt))
                                       for ind in indices]
                            # Convert list of indices to a np.array
                            indices = np.array(indices)

                            # Compute true output
                            true_output = (params_with_zero[indices] if
                                           len(params_with_zero.shape) == 1
                                           else params_with_zero[:, indices])

                            # Test Output values, shape and dtype
                            np.testing.assert_array_almost_equal(output,
                                                                 np.array(true_output))
                            self.assertEqual(p_dt.as_numpy_dtype, output.dtype)
                            np.testing.assert_array_equal(op.get_shape(),
                                                          list(np.array(true_output).shape))

        # List of params shapes
        params_shapes = [(6, ),   # 1D params
                         (1, 6),  # 2D params with single row
                         (3, 6)]  # 2D params with multiple rows and columns

        # List of padding elements
        pad_elems = [-float('inf'), -1.0, 0.0, 1.0, 1.23456789, float('inf'),  # float
                     -1, 0, 1, 12345678]  # int

        # All combination of test cases for gather_cols_3d without padding
        for p_shape in params_shapes:
            for p_elem in pad_elems:
                test(params_shape=p_shape, indices_shape=(4, 5),
                     param_dtype=[tf.float32, tf.float64, tf.int32, tf.int64],
                     ind_dtype=[np.int32, np.int64], pad_elem=p_elem,
                     use_gpu=True)

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
                    op1dn = spn.utils.scatter_cols(p1d, indices, num_out_cols)
                    op2d1n = spn.utils.scatter_cols(p2d1, indices, num_out_cols)
                    op2d2n = spn.utils.scatter_cols(p2d2, indices, num_out_cols)
                    # Run
                    out1dn = sess.run(op1dn)
                    out2d1n = sess.run(op2d1n)
                    out2d2n = sess.run(op2d2n)
                # Compare
                np.testing.assert_array_almost_equal(out1dn, true_output)
                self.assertEqual(params_dtype.as_numpy_dtype, out1dn.dtype)
                true_output_2d1 = [np.array(true_output)]
                true_output_2d2 = [np.array(true_output),
                                   np.array(true_output) * 2,
                                   np.array(true_output) * 3]
                np.testing.assert_array_almost_equal(out2d1n, true_output_2d1)
                np.testing.assert_array_almost_equal(out2d2n, true_output_2d2)
                self.assertEqual(params_dtype.as_numpy_dtype, out2d1n.dtype)
                self.assertEqual(params_dtype.as_numpy_dtype, out2d2n.dtype)

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
                                                          list(np.array(
                                                               t_out).shape))

                            t_out_2d1 = [np.array(t_out)]
                            np.testing.assert_array_almost_equal(out2d1,
                                                                 t_out_2d1)
                            self.assertEqual(p_dt.as_numpy_dtype, out2d1.dtype)
                            np.testing.assert_array_equal(op2d1.get_shape(),
                                                          list(np.array(
                                                               t_out_2d1).shape))

                            t_out_2d2 = [np.array(t_out) * row1,
                                         np.array(t_out) * row2,
                                         np.array(t_out) * row3]
                            np.testing.assert_array_almost_equal(out2d2,
                                                                 np.array(t_out_2d2))
                            self.assertEqual(p_dt.as_numpy_dtype, out2d2.dtype)
                            np.testing.assert_array_equal(op2d2.get_shape(),
                                                          list(np.array(
                                                               t_out_2d2).shape))

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


if __name__ == '__main__':
    tf.test.main()
