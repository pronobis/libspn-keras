#!/usr/bin/env python3

import unittest
import tensorflow as tf
import numpy as np
import time


class TestGatherColumnsPerformance(tf.test.TestCase):
    gather_columns_module = tf.load_op_library('./gather_columns.so')
    num_cols = 1000  # This should always be a multiple of 10
    num_rows = 10000
    num_stacked_ops = 6

    def tearDown(self):
        tf.reset_default_graph()

    def test_gather_columns(self):

        def test(params, indices, dtype, true_output, case='best', use_gpu=False):

            with self.test_session(use_gpu=use_gpu) as sess:
                # Make num_stacked_ops an even number to ensure that output of
                # the final op in the stacked operations matches the true_output
                if self.num_stacked_ops % 2 == 1:
                    self.num_stacked_ops = self.num_stacked_ops + 1

                npdtype = dtype.as_numpy_dtype()

                # Based on the params vector, create a params matrix of size
                # (num_rows, num_cols).
                params_matrix = np.empty([self.num_rows, self.num_cols],
                                         dtype=npdtype)
                params_row = np.array(params, dtype=npdtype)
                for i in range(0, self.num_rows):
                    params_matrix[i, :] = params_row * (i + 1)
                op2d2 = tf.constant(params_matrix, dtype=dtype)

                ind = tf.constant(indices, dtype=tf.int32)

                # Create an Op Stack
                for i in range(0, self.num_stacked_ops):
                    op2d2 = self.gather_columns_module.gather_columns(op2d2, ind)

                start_time = time.time()
                out2d2 = sess.run(op2d2)
                total_time = time.time() - start_time

                # Calclate and print total time taken to process the op stack.
                # To print processing time of each individual op, use 'Make debug'
                # instead, which enables the EXEC_TIME_CALC debug flag.
                print("%s case - Total time for %s: %.5f s" % (case, "GPU" if
                      use_gpu else "CPU", total_time))

                # Create a large output matrix, based on the true output
                # parameter, to compare the final op's output against.
                true_output_row = np.array(true_output, dtype=npdtype)
                for i in range(0, self.num_rows):
                    params_matrix[i, :] = true_output_row * (i + 1)
                true_output_2d2 = params_matrix

                np.testing.assert_array_almost_equal(out2d2, true_output_2d2)

        # Large case for performance test
        # Worst-case (0% - In-op optimization not used)
        # CPU
        test(list(range(1, self.num_cols + 1)),  # params:  [1, 2, 3, ..., n-1, n],
                                                 # where n = num_cols
             list(range(self.num_cols - 1, -1, -1)),  # indices: [n-1, n-2, n-3, ..., 1, 0]
             tf.float64,
             list(range(1, self.num_cols + 1)),  # Expected op output: [n, n-1, n-2, ..., 2, 1]
             case='Worst', use_gpu=False)

        # GPU
        test(list(range(1, self.num_cols + 1)),  # params:  [1, 2, 3, ..., n-1, n],
                                                 # where n = num_cols
             list(range(self.num_cols - 1, -1, -1)),  # indices: [n-1, n-2, n-3, ..., 1, 0]
             tf.float64,
             list(range(1, self.num_cols + 1)),  # Expected op output: [n, n-1, n-2, ..., 2, 1]
             case='Worst', use_gpu=True)

        # Intermediate-case (10% optimal)
        rows = self.num_cols/10
        cols = 10
        shuffled_ind = [0, 1, 9, 8, 7, 6, 5, 4, 3, 2]
        ind = np.ones((rows, cols), dtype=np.int) * \
              (np.arange(0, self.num_cols, cols)[np.newaxis]).T
        ind = (ind + shuffled_ind).flatten()
        # CPU
        test(list(range(1, self.num_cols + 1)),  # params:  [1, 2, 3, ..., n-1, n],
                                                 # where n = num_cols
             ind,  # indices: [(0, 1), (9), (8), (7), (6), (5), (4), (3), (2), ...]
             tf.float64,
             list(range(1, self.num_cols + 1)),  # Expected op output: [1, 2, 3, ..., n-1, n]
             case='Intermediate(10%)', use_gpu=False)

        # GPU
        test(list(range(1, self.num_cols + 1)),  # params:  [1, 2, 3, ..., n-1, n],
                                                 # where n = num_cols
             ind,  # indices: [(0, 1), (9), (8), (7), (6), (5), (4), (3), (2), ...]
             tf.float64,
             list(range(1, self.num_cols + 1)),  # Expected op output: [1, 2, 3, ..., n-1, n]
             case='Intermediate(10%)', use_gpu=True)

        # Intermediate-case (40% optimal)
        rows = self.num_cols/10
        cols = 10
        shuffled_ind = [0, 1, 2, 9, 5, 4, 6, 7, 8, 3]
        ind = np.ones((rows, cols), dtype=np.int) * \
              (np.arange(0, self.num_cols, cols)[np.newaxis]).T
        ind = (ind + shuffled_ind).flatten()
        # CPU
        test(list(range(1, self.num_cols + 1)),  # params:  [1, 2, 3, ..., n-1, n],
                                                 # where n = num_cols
             ind,  # indices: [(0, 1, 2), (9), (5), (4), (6, 7, 8), (3), ...]
             tf.float64,
             list(range(1, self.num_cols + 1)),  # Expected op output: [1, 2, 3, ..., n-1, n]
             case='Intermediate(40%)', use_gpu=False)

        # GPU
        test(list(range(1, self.num_cols + 1)),  # params:  [1, 2, 3, ..., n-1, n],
                                                 # where n = num_cols
             ind,  # indices: [(0, 1, 2), (9), (5), (4), (6, 7, 8), (3), ...]
             tf.float64,
             list(range(1, self.num_cols + 1)),  # Expected op output: [1, 2, 3, ..., n-1, n]
             case='Intermediate(40%)', use_gpu=True)

        # Intermediate-case (70% optimal)
        rows = self.num_cols/10
        cols = 10
        shuffled_ind = [9, 1, 2, 3, 4, 5, 6, 7, 8, 0]
        ind = np.ones((rows, cols), dtype=np.int) * \
              (np.arange(0, self.num_cols, cols)[np.newaxis]).T
        ind = (ind + shuffled_ind).flatten()
        # CPU
        test(list(range(1, self.num_cols + 1)),  # params:  [1, 2, 3, ..., n-1, n],
                                                 # where n = num_cols
             ind,  # indices: [(9), (1, 2, 3, 4, 5, 6, 7, 8), (0), ...]
             tf.float64,
             list(range(1, self.num_cols + 1)),  # Expected op output: [1, 2, 3, ..., n-1, n]
             case='Intermediate(70%)', use_gpu=False)

        # GPU
        test(list(range(1, self.num_cols + 1)),  # params:  [1, 2, 3, ..., n-1, n],
                                                 # where n = num_cols
             ind,  # indices: [(9), (1, 2, 3, 4, 5, 6, 7, 8), (0), ...]
             tf.float64,
             list(range(1, self.num_cols + 1)),  # Expected op output: [1, 2, 3, ..., n-1, n]
             case='Intermediate(70%)', use_gpu=True)

        # Best-case (100% optimal - In-op optimization completely used)
        # CPU
        test(list(range(1, self.num_cols + 1)),  # params:  [1, 2, 3, ..., n-1, n],
                                                 # where n = num_cols
             list(range(0, self.num_cols)),  # indices: [0, 1, 2, ..., n-2, n-1]
             tf.float64,
             list(range(1, self.num_cols + 1)),  # Expected op output: [1, 2, 3, ..., n-1, n]
             case='Best', use_gpu=False)

        # GPU
        test(list(range(1, self.num_cols + 1)),  # params:  [1, 2, 3, ..., n-1, n],
                                                 # where n = num_cols
             list(range(0, self.num_cols)),  # indices: [0, 1, 2, ..., n-2, n-1]
             tf.float64,
             list(range(1, self.num_cols + 1)),  # Expected op output: [1, 2, 3, ..., n-1, n]
             case='Best', use_gpu=True)


if __name__ == '__main__':
    unittest.main()
