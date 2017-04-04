#!/usr/bin/env python3

import tensorflow as tf
import math
import numpy as np
from context import libspn as spn
import time

gather_columns_module = tf.load_op_library('./gather_columns.so')


def fun_custom(params, indices):
    return gather_columns_module.gather_columns(params, indices)


def fun_tfindexing(params, indices):
    return tf.stack([params[:, c] for c in indices], -1)


COLOR = '\033[1m\033[93m'
ENDC = '\033[0m'


class TestGatherColumnsPerformance(tf.test.TestCase):

    @classmethod
    def setUpClass(self):
        # Params
        self.num_cols = 100  # This should always be a multiple of 10
        self.num_rows = 1000
        self.num_stacked_ops = 600
        self.log_device_placement = False  # Change to True to see if GPU is really used
        self.dtype = tf.float32
        self.npdtype = self.dtype.as_numpy_dtype()

    def run_test(self, fun, params, indices, true_output,
                 device_name):

        with self.test_session(config=tf.ConfigProto(
                log_device_placement=self.log_device_placement)) as sess:
            with tf.device(device_name):
                # Make the num_stacked_ops an even number to ensure that the
                # output of the final op in the stacked operations matches
                # the true_output
                if self.num_stacked_ops % 2 == 1:
                    self.num_stacked_ops = self.num_stacked_ops + 1

                # Based on the params vector, create a params matrix of size
                # (num_rows, num_cols).
                params_matrix = np.empty([self.num_rows, self.num_cols],
                                         dtype=self.npdtype)
                params_row = np.array(params, dtype=self.npdtype)
                for i in range(0, self.num_rows):
                    params_matrix[i, :] = params_row * (i + 1)
                op2d2 = tf.constant(params_matrix, dtype=self.dtype)

                ind = np.array(indices, dtype=np.int32)

                # Create an Op Stack
                for i in range(0, self.num_stacked_ops):
                    op2d2 = fun(op2d2, ind)

                # Create a large output matrix, based on the true output
                # parameter, to compare the final op's output against.
                true_output_row = np.array(true_output, dtype=self.npdtype)
                for i in range(0, self.num_rows):
                    params_matrix[i, :] = true_output_row * (i + 1)
                true_output_2d2 = params_matrix

            # Run
            start_time = time.time()
            out2d2 = sess.run(op2d2)
            total_time = time.time() - start_time

            # Calclate and print total time taken to process the op stack.
            # To print processing time of each individual op, use 'Make debug'
            # instead, which enables the EXEC_TIME_CALC debug flag.
            print((COLOR + "Total time for case %s on %s: %.5f s" + ENDC) %
                  (self.id().split('.')[2].upper(), device_name, total_time))

            # Test generated output
            np.testing.assert_array_almost_equal(out2d2, true_output_2d2)

    def run_test_opt0(self, fun, device_name):
        """Case: Worst-case (in-op optimization not used (0%))"""

        self.run_test(
            fun,
            list(range(1, self.num_cols + 1)),  # params:  [1, 2, 3, ..., n-1, n], n = num_cols
            list(range(self.num_cols - 1, -1, -1)),  # indices: [n-1, n-2, n-3, ..., 1, 0]
            list(range(1, self.num_cols + 1)),  # Expected op output: [n, n-1, n-2, ..., 2, 1]
            device_name=device_name)

    def run_test_tfindexing_opt0(self, device_name):
        """Method: TF Indexing
           Case: Worst-case (in-op optimization not used (0%))"""

        self.run_test_opt0(fun=fun_tfindexing, device_name=device_name)

    def test_tfindexing_cpu_opt0(self):
        """Method: TF Indexing
           Device: CPU
           Case: Worst-case (in-op optimization not used (0%))"""

        self.run_test_tfindexing_opt0('/cpu:0')

    def test_tfindexing_gpu_opt0(self):
        """Method: TF Indexing
           Device: GPU
           Case: Worst-case (in-op optimization not used (0%))"""

        self.run_test_tfindexing_opt0('/gpu:0')

    def run_test_gathernd_opt0(self, device_name):
        """Method: gather_nd and transpose
           Case: Worst-case (in-op optimization not used (0%))"""

        self.run_test_opt0(fun=lambda p, i: spn.utils.gather_cols(p, i, use_gather_nd=True),
                           device_name=device_name)

    def test_gathernd_cpu_opt0(self):
        """Method: gather_nd and transpose
           Device: CPU
           Case: Worst-case (in-op optimization not used (0%))"""

        self.run_test_gathernd_opt0('/cpu:0')

    def test_gathernd_gpu_opt0(self):
        """Method: gather_nd and transpose
           Device: GPU
           Case: Worst-case (in-op optimization not used (0%))"""

        self.run_test_gathernd_opt0('/gpu:0')

    def run_test_custom_opt0(self, device_name):
        """Method: custom gather_cols op
           Case: Worst-case (in-op optimization not used (0%))"""

        self.run_test_opt0(fun=fun_custom,
                           device_name=device_name)

    def test_custom_cpu_opt0(self):
        """Method: custom gather_cols op
           Device: CPU
           Case: Worst-case (in-op optimization not used (0%))"""

        self.run_test_custom_opt0('/cpu:0')

    def test_custom_gpu_opt0(self):
        """Method: custom gather_cols op
           Device: GPU
           Case: Worst-case (in-op optimization not used (0%))"""

        self.run_test_custom_opt0('/gpu:0')


if __name__ == '__main__':
    tf.test.main()
