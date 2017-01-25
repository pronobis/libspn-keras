import unittest
import tensorflow as tf
import numpy as np

from tensorflow.python.framework import ops
from tensorflow.python.framework import common_shapes

class TestMath(tf.test.TestCase):
    gather_columns_module = tf.load_op_library('./gather_columns.so')
    num_cols = 1000
    num_rows = 30000

    def tearDown(self):
        tf.reset_default_graph()

    def testEmptyParams(self):
      with self.test_session(use_gpu=False) as sess:
        params = tf.constant([], dtype=tf.int32)
        indices = [1, 2, 3]
        gather = self.gather_columns_module.gather_columns(params, indices)
        with self.assertRaisesOpError("Params cannot be empty."):
            sess.run(gather)

    def testEmptyIndices(self):
      with self.test_session(use_gpu=False) as sess:
        params = [0, 1, 2]
        indices = tf.constant([], dtype=tf.int32)
        gather = self.gather_columns_module.gather_columns(params, indices)
        with self.assertRaisesOpError("Indices cannot be empty."):
            sess.run(gather)

    def testScalarParams(self):
      with self.test_session(use_gpu=False) as sess:
        params = 10
        indices = [1, 2, 3]
        gather = self.gather_columns_module.gather_columns(params, indices)
        with self.assertRaisesOpError("Params must be at least a vector."):
            sess.run(gather)

    def testScalarIndices(self):
      with self.test_session(use_gpu=False) as sess:
        params = [1, 2, 3]
        indices = 1
        gather = self.gather_columns_module.gather_columns(params, indices)
        with self.assertRaisesOpError("Indices must be a vector, but it is a: 0D Tensor."):
            sess.run(gather)

    def test3DParams(self):
      with self.test_session(use_gpu=False) as sess:
        params = [[[0, 1, 2]]]
        indices = [1, 2, 3]
        gather = self.gather_columns_module.gather_columns(params, indices)
        with self.assertRaisesOpError("Params must be 1D or 2D but it is: 3D."):
            sess.run(gather)

    def test2DIndices(self):
      with self.test_session(use_gpu=False) as sess:
        params = [[0, 1, 2]]
        indices = [[1, 2, 3]]
        gather = self.gather_columns_module.gather_columns(params, indices)
        with self.assertRaisesOpError("Indices must be a vector, but it is a: 2D Tensor."):
            sess.run(gather)

    def testNegativeIndices_CPU(self):
      with self.test_session(use_gpu=False) as sess:
        params = tf.constant([1, 2, 3], dtype=tf.float32)
        indices = [-1]
        gather = self.gather_columns_module.gather_columns(params, indices)
        with self.assertRaisesOpError("Indices\(0\) is not in range \(0, 3\]."):
            sess.run(gather)

    def testNegativeIndices_GPU(self):
      with self.test_session(use_gpu=True) as sess:
        params = tf.constant([1, 2, 3], dtype=tf.float32)
        indices = [-1]
        gather = self.gather_columns_module.gather_columns(params, indices)
        with self.assertRaisesOpError("Indices\(0\) is not in range \(0, 3\]."):
            sess.run(gather)

    def testBadIndices_CPU(self):
      with self.test_session(use_gpu=False) as sess:
        params = tf.constant([[1, 2, 3, 4, 5]], dtype=tf.float64)
        indices = tf.constant([2, 1, 10, 1, 2], dtype=tf.int32)
        gather = self.gather_columns_module.gather_columns(params, indices)
        with self.assertRaisesOpError("Indices\(2\) is not in range \(0, 5\]."):
            sess.run(gather)

    def testBadIndices_GPU(self):
      with self.test_session(use_gpu=True) as sess:
        params = tf.constant([[1, 2, 3, 4, 5]], dtype=tf.float64)
        indices = tf.constant([2, 1, 10, 1, 2], dtype=tf.int32)
        gather = self.gather_columns_module.gather_columns(params, indices)
        with self.assertRaisesOpError("Indices\(2\) is not in range \(0, 5\]."):
            sess.run(gather)

    def test_gather_columns(self):
        ops.RegisterShape("GatherColumns")(common_shapes.call_cpp_shape_fn)

        def test(params, indices, dtype, true_output, use_gpu=False, large_case=False):

            with self.test_session(use_gpu=use_gpu) as sess:
                if dtype == bool:
                    row1 = row2 = row3 = 1
                else:
                    row1 = 1
                    row2 = 0
                    row3 = -1

                if dtype == tf.bool:
                    npdtype = np.bool
                elif dtype == tf.float32:
                    npdtype = np.float32
                elif dtype == tf.float64:
                    npdtype = np.float64
                elif dtype == tf.int32:
                    npdtype = np.int32
                elif dtype == tf.int64:
                    npdtype = np.int64

                p1d = tf.constant(params, dtype=dtype)
                p2d1 = tf.constant(np.array([np.array(params)]), dtype=dtype)

                if not large_case:
                    p2d2 = tf.constant(np.array([np.array(params) * row1,
                                                 np.array(params) * row2,
                                                 np.array(params) * row3]), dtype=dtype)
                else:
                    params_matrix = np.empty([self.num_rows, self.num_cols], dtype=npdtype)
                    params_row = np.array(params, dtype=npdtype)
                    for i in range(0, self.num_rows):
                        params_matrix[i,:] = params_row * (i+1)
                    p2d2 = tf.constant(params_matrix, dtype=dtype)

                    # For testing only the overhead time
                    #p2d2 = tf.constant(params, dtype=dtype)

                ind_32 = tf.constant(indices, dtype=tf.int32)
                ind_64 = tf.constant(indices, dtype=tf.int64)

                op1d = self.gather_columns_module.gather_columns(p1d, ind_64)
                op2d1 = self.gather_columns_module.gather_columns(p2d1, ind_32)
                op2d2 = self.gather_columns_module.gather_columns(p2d2, ind_64)

                out1d = sess.run(op1d)
                out2d1 = sess.run(op2d1)
                out2d2 = sess.run(op2d2)

                np.testing.assert_array_almost_equal(out1d, true_output)
                self.assertEqual(dtype.as_numpy_dtype, out1d.dtype)
                np.testing.assert_array_equal(op1d.get_shape(), np.array([len(indices)]))

                true_output_2d1 = [np.array(true_output)]
                np.testing.assert_array_almost_equal(out2d1, true_output_2d1)
                self.assertEqual(dtype.as_numpy_dtype, out2d1.dtype)
                np.testing.assert_array_equal(op2d1.get_shape(), np.array([1, len(indices)]))

                if not large_case:
                    true_output_2d2 = [np.array(true_output) * row1,
                                       np.array(true_output) * row2,
                                       np.array(true_output) * row3]
                    true_shape = np.array([3, len(indices)])
                else:
                    true_output_row = np.array(true_output, dtype=npdtype)
                    for i in range(0, self.num_rows):
                        params_matrix[i,:] = true_output_row * (i+1)
                    true_output_2d2 = params_matrix
                    true_shape = np.array([self.num_rows, len(indices)])

                    # For testing only the overhead time
                    #true_output_2d2 = true_output

                np.testing.assert_array_almost_equal(out2d2, true_output_2d2)
                self.assertEqual(dtype.as_numpy_dtype, out2d2.dtype)
                np.testing.assert_array_equal(op2d2.get_shape(), true_shape)


        float_val = 1.23456789
        int_val = 123456789
        int_32_upper = 2147483647
        int_64_upper = 9223372036854775807


        # Single column input tensor
        # float
        test([float_val],
             [0],
             tf.float32,
             [float_val],
             use_gpu=False)
        test([float_val],
             [0],
             tf.float64,
             [float_val],
             use_gpu=False)

        test([float_val],
             [0],
             tf.float32,
             [float_val],
             use_gpu=True)
        test([float_val],
             [0],
             tf.float64,
             [float_val],
             use_gpu=True)

        # int
        test([int_32_upper],
             [0],
             tf.int32,
             [int_32_upper],
             use_gpu=False)
        test([int_64_upper],
             [0],
             tf.int64,
             [int_64_upper],
             use_gpu=False)

        test([int_32_upper],
             [0],
             tf.int32,
             [int_32_upper],
             use_gpu=True)
        test([int_64_upper],
             [0],
             tf.int64,
             [int_64_upper],
             use_gpu=True)

        # bool
        test([True],
             [0],
             tf.bool,
             [True],
             use_gpu=False)

        test([True],
             [0],
             tf.bool,
             [True],
             use_gpu=True)

        # Single index
        # float
        test([float_val, float_val*2, float_val*3],
             [1],
             tf.float32,
             [float_val*2],
             use_gpu=False)
        test([float_val, float_val*2, float_val*3],
             [1],
             tf.float64,
             [float_val*2],
             use_gpu=False)

        test([float_val, float_val*2, float_val*3],
             [1],
             tf.float32,
             [float_val*2],
             use_gpu=True)
        test([float_val, float_val*2, float_val*3],
             [1],
             tf.float64,
             [float_val*2],
             use_gpu=True)

        # int
        test([int_val, int_val*2, int_val*3],
             [0],
             tf.int32,
             [int_val],
             use_gpu=False)
        test([int_val, int_val*2, int_val*3],
             [2],
             tf.int64,
             [int_val*3],
             use_gpu=False)

        test([int_val, int_val*2, int_val*3],
             [0],
             tf.int32,
             [int_val],
             use_gpu=True)
        test([int_val, int_val*2, int_val*3],
             [2],
             tf.int64,
             [int_val*3],
             use_gpu=True)

        # bool
        test([False, True, False],
             [2],
             tf.bool,
             [False],
             use_gpu=False)

        test([False, True, False],
             [2],
             tf.bool,
             [False],
             use_gpu=True)

        # Multiple indices
        # float
        test([float_val, float_val*2, float_val*3, float_val*4],
             [1, 3, 2],
             tf.float32,
             [float_val*2, float_val*4, float_val*3],
             use_gpu=False)
        test([float_val, float_val*2, float_val*3, float_val*4],
             [1, 3, 2, 0, 2, 3, 1],
             tf.float64,
             [float_val*2, float_val*4, float_val*3, float_val*1, float_val*3, float_val*4, float_val*2],
             use_gpu=False)

        test([float_val, float_val*2, float_val*3, float_val*4],
             [1, 3, 2],
             tf.float32,
             [float_val*2, float_val*4, float_val*3],
             use_gpu=True)
        test([float_val, float_val*2, float_val*3, float_val*4],
             [1, 3, 2, 0, 2, 3, 1],
             tf.float64,
             [float_val*2, float_val*4, float_val*3, float_val*1, float_val*3, float_val*4, float_val*2],
             use_gpu=True)

        # int
        test([int_val, int_val*2, int_val*3, int_val*4],
             [3, 2, 1],
             tf.int32,
             [int_val*4, int_val*3, int_val*2],
             use_gpu=False)
        test([int_val, int_val*2, int_val*3, int_val*4],
             [3, 2, 1, 0, 1, 2, 3],
             tf.int64,
             [int_val*4, int_val*3, int_val*2, int_val*1, int_val*2, int_val*3, int_val*4],
             use_gpu=False)

        test([int_val, int_val*2, int_val*3, int_val*4],
             [3, 2, 1],
             tf.int32,
             [int_val*4, int_val*3, int_val*2],
             use_gpu=True)
        test([int_val, int_val*2, int_val*3, int_val*4],
             [3, 2, 1, 0, 1, 2, 3],
             tf.int64,
             [int_val*4, int_val*3, int_val*2, int_val*1, int_val*2, int_val*3, int_val*4],
             use_gpu=True)

        # bool
        test([True, True, False, True, False],
             [0, 1, 2, 3, 4],
             tf.bool,
             [True, True, False, True, False],
             use_gpu=False)
        test([False, False, True, True, False, True],
             [5, 4, 3, 2, 1, 0],
             tf.bool,
             [True, False, True, True, False, False],
             use_gpu=False)

        test([True, True, False, True, False],
             [0, 1, 2, 3, 4],
             tf.bool,
             [True, True, False, True, False],
             use_gpu=True)
        test([False, False, True, True, False, True],
             [5, 4, 3, 2, 1, 0],
             tf.bool,
             [True, False, True, True, False, False],
             use_gpu=True)

        # Indices with consecutive columns
        # Begining
        test([float_val*1, float_val*2, float_val*3, float_val*4, float_val*5, float_val*6, float_val*7, float_val*8, float_val*9],
             [4, 5, 6, 8, 2, 0, 3, 1, 7],
             tf.float32,
             [float_val*5, float_val*6, float_val*7, float_val*9, float_val*3, float_val*1, float_val*4, float_val*2, float_val*8],
             use_gpu=False)
        test([float_val*1, float_val*2, float_val*3, float_val*4, float_val*5, float_val*6, float_val*7, float_val*8, float_val*9],
             [4, 5, 6, 8, 2, 0, 3, 1, 7],
             tf.float32,
             [float_val*5, float_val*6, float_val*7, float_val*9, float_val*3, float_val*1, float_val*4, float_val*2, float_val*8],
             use_gpu=True)

        # Middle
        test([int_val*1, int_val*2, int_val*3, int_val*4, int_val*5, int_val*6, int_val*7, int_val*8, int_val*9],
             [3, 5, 6, 7, 4, 0, 1, 2, 8],
             tf.int32,
             [int_val*4, int_val*6, int_val*7, int_val*8, int_val*5, int_val*1, int_val*2, int_val*3, int_val*9],
             use_gpu=False)
        test([int_val*1, int_val*2, int_val*3, int_val*4, int_val*5, int_val*6, int_val*7, int_val*8, int_val*9],
             [3, 5, 6, 7, 4, 0, 1, 2, 8],
             tf.int32,
             [int_val*4, int_val*6, int_val*7, int_val*8, int_val*5, int_val*1, int_val*2, int_val*3, int_val*9],
             use_gpu=True)

        # End
        test([int_val*1, int_val*2, int_val*3, int_val*4, int_val*5, int_val*6, int_val*7, int_val*8, int_val*9],
             [6, 5, 0, 7, 4, 8, 1, 2, 3],
             tf.int64,
             [int_val*7, int_val*6, int_val*1, int_val*8, int_val*5, int_val*9, int_val*2, int_val*3, int_val*4],
             use_gpu=False)
        test([int_val*1, int_val*2, int_val*3, int_val*4, int_val*5, int_val*6, int_val*7, int_val*8, int_val*9],
             [6, 5, 0, 7, 4, 8, 1, 2, 3],
             tf.int64,
             [int_val*7, int_val*6, int_val*1, int_val*8, int_val*5, int_val*9, int_val*2, int_val*3, int_val*4],
             use_gpu=True)

        # Beginning, middle and end
        test([int_val*1, int_val*2, int_val*3, int_val*4, int_val*5, int_val*6, int_val*7, int_val*8, int_val*9, int_val*10, int_val*11, int_val*12],
             [5, 6, 7, 11, 1, 2, 3, 0, 4, 8, 9, 10],
             tf.int64,
             [int_val*6, int_val*7, int_val*8, int_val*12, int_val*2, int_val*3, int_val*4, int_val*1, int_val*5, int_val*9, int_val*10, int_val*11],
             use_gpu=False)
        test([int_val*1, int_val*2, int_val*3, int_val*4, int_val*5, int_val*6, int_val*7, int_val*8, int_val*9, int_val*10, int_val*11, int_val*12],
             [5, 6, 7, 11, 1, 2, 3, 0, 4, 8, 9, 10],
             tf.int64,
             [int_val*6, int_val*7, int_val*8, int_val*12, int_val*2, int_val*3, int_val*4, int_val*1, int_val*5, int_val*9, int_val*10, int_val*11],
             use_gpu=True)

        # Large case for performance test
        test(list(range(1, self.num_cols+1)), # [1, 2, 3, ..., n-1, n]
             list(range(self.num_cols-1, -1, -1)), # [n-1, n-2, n-3, ..., 1, 0]
             tf.float64,
             list(range(self.num_cols, 0, -1)), # [n, n-1, n-2, ..., 2, 1]
             use_gpu=False,
             large_case=True)
        test(list(range(1, self.num_cols+1)), # [1, 2, 3, ..., n-1, n]
             list(range(self.num_cols-1, -1, -1)), # [n-1, n-2, n-3, ..., 1, 0]
             tf.float64,
             list(range(self.num_cols, 0, -1)), # [n, n-1, n-2, ..., 2, 1]
             use_gpu=True,
             large_case=True)

if __name__ == '__main__':
    unittest.main()
