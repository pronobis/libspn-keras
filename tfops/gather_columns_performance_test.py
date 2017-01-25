import unittest
import tensorflow as tf
import numpy as np

class TestMath(tf.test.TestCase):
    gather_columns_module = tf.load_op_library('./gather_columns.so')
    num_cols = 1000
    num_rows = 20000
    an_odd_number = 101

    def tearDown(self):
        tf.reset_default_graph()

    def test_gather_columns(self):

        def test(params, indices, dtype, true_output, use_gpu=False):

            with self.test_session(use_gpu=use_gpu) as sess:
                if self.an_odd_number % 2 == 0:
                    self.an_odd_number = self.an_odd_number+1

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

                params_matrix = np.empty([self.num_rows, self.num_cols], dtype=npdtype)
                params_row = np.array(params, dtype=npdtype)
                for i in range(0, self.num_rows):
                    params_matrix[i,:] = params_row * (i+1)
                op2d2 = tf.constant(params_matrix, dtype=dtype)

                # For testing only the overhead time
                #op2d2 = tf.constant(params, dtype=dtype)

                ind = tf.constant(indices, dtype=tf.int32)

                # Create an Op Stack
                for i in range(0, self.an_odd_number):
                    op2d2 = self.gather_columns_module.gather_columns(op2d2, ind)

                out2d2 = sess.run(op2d2)

                true_output_row = np.array(true_output, dtype=npdtype)
                for i in range(0, self.num_rows):
                    params_matrix[i,:] = true_output_row * (i+1)
                true_output_2d2 = params_matrix

                # For testing only the overhead time
                #true_output_2d2 = true_output

                np.testing.assert_array_almost_equal(out2d2, true_output_2d2)

        # Large case for performance test
        test(list(range(1, self.num_cols+1)), # [1, 2, 3, ..., n-1, n]
             list(range(self.num_cols-1, -1, -1)), # [n-1, n-2, n-3, ..., 1, 0]
             tf.float64,
             list(range(self.num_cols, 0, -1)), # [n, n-1, n-2, ..., 2, 1]
             use_gpu=False)

        test(list(range(1, self.num_cols+1)), # [1, 2, 3, ..., n-1, n]
             list(range(self.num_cols-1, -1, -1)), # [n-1, n-2, n-3, ..., 1, 0]
             tf.float64,
             list(range(self.num_cols, 0, -1)), # [n, n-1, n-2, ..., 2, 1]
             use_gpu=True)

if __name__ == '__main__':
    unittest.main()
