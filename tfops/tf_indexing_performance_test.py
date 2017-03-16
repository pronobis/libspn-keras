#!/usr/bin/env python3

import unittest
import tensorflow as tf
import numpy as np
import time


class TestGatherColumnsPerformance(tf.test.TestCase):
    num_cols = 1000
    num_rows = 19000
    num_stacked_ops = 5

    def tearDown(self):
        tf.reset_default_graph()

    def test_gather_columns(self):

        def test(params, indices, dtype, true_output, device_name):

            with self.test_session() as sess:
                with tf.device(device_name):
                    # Make the num_stacked_ops an odd number to ensure that the
                    # output of the final op in the stacked operations matches
                    # the true_output
                    if self.num_stacked_ops % 2 == 0:
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

                    ind = np.array(indices, dtype=np.int32)

                    # Create an Op Stack
                    for i in range(0, self.num_stacked_ops):
                        op2d2 = tf.stack([op2d2[:, c] for c in ind], -1)

                    # Create a large output matrix, based on the true output
                    # parameter, to compare the final op's output against.
                    true_output_row = np.array(true_output, dtype=npdtype)
                    for i in range(0, self.num_rows):
                        params_matrix[i, :] = true_output_row * (i + 1)
                    true_output_2d2 = params_matrix

                start_time = time.time()
                out2d2 = sess.run(op2d2)
                total_time = time.time() - start_time

                # Calclate and print total time taken to process the op stack.
                # To print processing time of each individual op, use 'Make debug'
                # instead, which enables the EXEC_TIME_CALC debug flag.
                print("Total time for %s: %.5f s" % (
                      "CPU" if device_name == '/cpu:0' else "GPU", total_time))

                np.testing.assert_array_almost_equal(out2d2, true_output_2d2)

        # Large case for performance test
        # CPU
        test(list(range(1, self.num_cols + 1)),  # params:  [1, 2, 3, ..., n-1, n],
                                                 # where n = num_cols
             list(range(self.num_cols - 1, -1, -1)),  # indices: [n-1, n-2, n-3, ..., 1, 0]
             tf.float64,
             list(range(self.num_cols, 0, -1)),  # Expected op output: [n, n-1, n-2, ..., 2, 1]
             device_name='/cpu:0')

        # GPU
        test(list(range(1, self.num_cols + 1)),  # params:  [1, 2, 3, ..., n-1, n],
                                                 # where n = num_cols
             list(range(self.num_cols - 1, -1, -1)),  # indices: [n-1, n-2, n-3, ..., 1, 0]
             tf.float64,
             list(range(self.num_cols, 0, -1)),  # Expected op output: [n, n-1, n-2, ..., 2, 1]
             device_name='/gpu:0')


if __name__ == '__main__':
    unittest.main()
