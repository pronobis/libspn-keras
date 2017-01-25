import unittest
import tensorflow as tf
import numpy as np

class TestMath(tf.test.TestCase):
    scatter_columns_module = tf.load_op_library('./scatter_columns.so')
    num_cols = 1000
    num_rows = 25000
    an_odd_number = 100

    def tearDown(self):
        tf.reset_default_graph()

    def test_scatter_cols(self):

        def test(params, indices, out_num_cols, pad_elem, dtype, true_output, use_gpu=False):
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


                params_matrix = np.empty([self.num_rows, self.num_cols], npdtype)
                params_row = np.array(params, npdtype)
                for i in range(0, self.num_rows):
                    params_matrix[i,:] = params_row * (i+1)
                op2d2 = tf.constant(params_matrix, dtype=dtype)
                # For testing only the overhead time
                #op2d2 = tf.constant(params, dtype=dtype)

                ind_32 = tf.constant(indices, dtype=tf.int32)
                ind_64 = tf.constant(indices, dtype=tf.int64)

                # Create an Op Stack
                for i in range(0, self.an_odd_number):
                    op2d2 = self.scatter_columns_module.scatter_columns(op2d2, ind_32, pad_elem, out_num_cols)

                out2d2 = sess.run(op2d2)

                true_output_matrix = np.empty([self.num_rows, self.num_cols], npdtype)
                true_output_row = np.array(true_output, npdtype)
                ind = np.array(indices)
                for i in range(0, self.num_rows):
                    true_output_matrix[i,:] = true_output_row
                    true_output_matrix[i,ind] = true_output_row[ind] * (i+1)
                true_output_2d2 = true_output_matrix
                # For testing only the overhead time
                #true_output_2d2 = true_output

                np.testing.assert_array_almost_equal(out2d2, true_output_2d2)

        pad_elem = 333

        # Large case for performance test
        test(list(range(1, self.num_cols+1)), # [1, 2, 3, ..., n-1, n]
             list(range(self.num_cols-1, -1, -1)), # [n-1, n-2, n-3, ..., 1, 0]
             self.num_cols,
             pad_elem,
             tf.float64,
             list(range(self.num_cols, 0, -1)),
             use_gpu=False) # [n, n-1, n-2, ..., 2, 1]

        test(list(range(1, self.num_cols+1)), # [1, 2, 3, ..., n-1, n]
             list(range(self.num_cols-1, -1, -1)), # [n-1, n-2, n-3, ..., 1, 0]
             self.num_cols,
             pad_elem,
             tf.float64,
             list(range(self.num_cols, 0, -1)),
             use_gpu=True) # [n, n-1, n-2, ..., 2, 1]

if __name__ == '__main__':
    unittest.main()
