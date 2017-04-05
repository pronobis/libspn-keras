#!/usr/bin/env python3

import tensorflow as tf
import sys
import numpy as np
from context import libspn as spn
import time
import argparse

gather_columns_module = tf.load_op_library('./gather_columns.so')


def fun_custom(params, indices):
    return gather_columns_module.gather_columns(params, indices)


def fun_tfindexing(params, indices):
    return tf.stack([params[:, c] for c in indices], -1)


def printc(string):
    COLOR = '\033[1m\033[93m'
    ENDC = '\033[0m'
    print(COLOR + string + ENDC)


class TestGatherColumnsPerformance(tf.test.TestCase):

    @classmethod
    def setUpClass(self):
        # Params
        self.dtype = tf.float32
        printc("Info:")
        printc("- num_cols: %s" % self.num_cols)
        printc("- num_rows: %s" % self.num_rows)
        printc("- num_stacked_ops: %s" % self.num_stacked_ops)
        printc("- log_device_placement: %s" % self.log_device_placement)
        printc("- dtype: %s" % self.dtype)
        # Generate params matrix
        self.params = np.random.rand(self.num_rows, self.num_cols)
        self.params = np.asarray(self.params,
                                 dtype=self.dtype.as_numpy_dtype())

    def run_test(self, fun, indices, device_name):
        with self.test_session(config=tf.ConfigProto(
                log_device_placement=self.log_device_placement)) as sess:
            with tf.device(device_name):
                indices = np.asarray(indices, dtype=np.int32)

                # Create an op stack
                op = tf.constant(self.params, dtype=self.dtype)
                for i in range(self.num_stacked_ops):
                    op = fun(op, indices)

                # Compute true output with numpy
                true_out = self.params
                for i in range(self.num_stacked_ops):
                    true_out = true_out[:, indices]

            # Run
            start_time = time.time()
            op_out = sess.run(op)
            total_time = time.time() - start_time

            # Print stats
            # To print processing time of each individual op, use 'make debug'
            # instead, which enables the EXEC_TIME_CALC debug flag.
            printc("Total time for case %s on %s: %.5f s" %
                   (self.id().split('.')[2].upper(), device_name, total_time))

            # Test generated output
            np.testing.assert_array_almost_equal(op_out, true_out)

    def run_test_opt0(self, fun, device_name):
        """Case: Worst-case (in-op optimization not used (0%))"""

        self.run_test(
            fun,
            list(range(self.num_cols - 1, -1, -1)),  # indices: [n-1, n-2, n-3, ..., 1, 0]
            device_name=device_name)

    def run_test_opt10(self, fun, device_name):
        """Case: In-op optimization 10%"""
        shuffled_ind = [0, 1, 9, 8, 7, 6, 5, 4, 3, 2]
        ind = np.empty((self.num_cols // 10, 10))
        ind[:] = shuffled_ind
        self.run_test(fun, ind.ravel(), device_name=device_name)

    def run_test_opt40(self, fun, device_name):
        """Case: In-op optimization 40%"""
        shuffled_ind = [0, 1, 2, 9, 5, 4, 6, 7, 8, 3]
        ind = np.empty((self.num_cols // 10, 10))
        ind[:] = shuffled_ind
        self.run_test(fun, ind.ravel(), device_name=device_name)

    def run_test_opt70(self, fun, device_name):
        """Case: In-op optimization 70%"""
        shuffled_ind = [9, 1, 2, 3, 4, 5, 6, 7, 8, 0]
        ind = np.empty((self.num_cols // 10, 10))
        ind[:] = shuffled_ind
        self.run_test(fun, ind.ravel(), device_name=device_name)

    def run_test_opt100(self, fun, device_name):
        """Case: Best-case (in-op optimization 100%)"""

        self.run_test(
            fun,
            list(range(0, self.num_cols)),  # indices: [0, 1, 2, ..., n-2, n-1]
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

    def run_test_tfindexing_opt10(self, device_name):
        """Method: TF Indexing
           Case: In-op optimization 10%"""

        self.run_test_opt10(fun=fun_tfindexing, device_name=device_name)

    def test_tfindexing_cpu_opt10(self):
        """Method: TF Indexing
           Device: CPU
           Case: In-op optimization 10%"""

        self.run_test_tfindexing_opt10('/cpu:0')

    def test_tfindexing_gpu_opt10(self):
        """Method: TF Indexing
           Device: GPU
           Case: In-op optimization 10%"""

        self.run_test_tfindexing_opt10('/gpu:0')

    def run_test_gathernd_opt10(self, device_name):
        """Method: gather_nd and transpose
           Case: In-op optimization 10%"""

        self.run_test_opt10(fun=lambda p, i: spn.utils.gather_cols(p, i, use_gather_nd=True),
                            device_name=device_name)

    def test_gathernd_cpu_opt10(self):
        """Method: gather_nd and transpose
           Device: CPU
           Case: In-op optimization 10%"""

        self.run_test_gathernd_opt10('/cpu:0')

    def test_gathernd_gpu_opt10(self):
        """Method: gather_nd and transpose
           Device: GPU
           Case: In-op optimization 10%"""

        self.run_test_gathernd_opt10('/gpu:0')

    def run_test_custom_opt10(self, device_name):
        """Method: custom gather_cols op
           Case: In-op optimization 10%"""

        self.run_test_opt10(fun=fun_custom,
                            device_name=device_name)

    def test_custom_cpu_opt10(self):
        """Method: custom gather_cols op
           Device: CPU
           Case: In-op optimization 10%"""

        self.run_test_custom_opt10('/cpu:0')

    def test_custom_gpu_opt10(self):
        """Method: custom gather_cols op
           Device: GPU
           Case: In-op optimization 10%"""

        self.run_test_custom_opt10('/gpu:0')

    def run_test_tfindexing_opt40(self, device_name):
        """Method: TF Indexing
           Case: In-op optimization 40%"""

        self.run_test_opt40(fun=fun_tfindexing, device_name=device_name)

    def test_tfindexing_cpu_opt40(self):
        """Method: TF Indexing
           Device: CPU
           Case: In-op optimization 40%"""

        self.run_test_tfindexing_opt40('/cpu:0')

    def test_tfindexing_gpu_opt40(self):
        """Method: TF Indexing
           Device: GPU
           Case: In-op optimization 40%"""

        self.run_test_tfindexing_opt40('/gpu:0')

    def run_test_gathernd_opt40(self, device_name):
        """Method: gather_nd and transpose
           Case: In-op optimization 40%"""

        self.run_test_opt40(fun=lambda p, i: spn.utils.gather_cols(p, i, use_gather_nd=True),
                            device_name=device_name)

    def test_gathernd_cpu_opt40(self):
        """Method: gather_nd and transpose
           Device: CPU
           Case: In-op optimization 40%"""

        self.run_test_gathernd_opt40('/cpu:0')

    def test_gathernd_gpu_opt40(self):
        """Method: gather_nd and transpose
           Device: GPU
           Case: In-op optimization 40%"""

        self.run_test_gathernd_opt40('/gpu:0')

    def run_test_custom_opt40(self, device_name):
        """Method: custom gather_cols op
           Case: In-op optimization 40%"""

        self.run_test_opt40(fun=fun_custom,
                            device_name=device_name)

    def test_custom_cpu_opt40(self):
        """Method: custom gather_cols op
           Device: CPU
           Case: In-op optimization 40%"""

        self.run_test_custom_opt40('/cpu:0')

    def test_custom_gpu_opt40(self):
        """Method: custom gather_cols op
           Device: GPU
           Case: In-op optimization 40%"""

        self.run_test_custom_opt40('/gpu:0')

    def run_test_tfindexing_opt70(self, device_name):
        """Method: TF Indexing
           Case: In-op optimization 70%"""

        self.run_test_opt70(fun=fun_tfindexing, device_name=device_name)

    def test_tfindexing_cpu_opt70(self):
        """Method: TF Indexing
           Device: CPU
           Case: In-op optimization 70%"""

        self.run_test_tfindexing_opt70('/cpu:0')

    def test_tfindexing_gpu_opt70(self):
        """Method: TF Indexing
           Device: GPU
           Case: In-op optimization 70%"""

        self.run_test_tfindexing_opt70('/gpu:0')

    def run_test_gathernd_opt70(self, device_name):
        """Method: gather_nd and transpose
           Case: In-op optimization 70%"""

        self.run_test_opt70(fun=lambda p, i: spn.utils.gather_cols(p, i, use_gather_nd=True),
                            device_name=device_name)

    def test_gathernd_cpu_opt70(self):
        """Method: gather_nd and transpose
           Device: CPU
           Case: In-op optimization 70%"""

        self.run_test_gathernd_opt70('/cpu:0')

    def test_gathernd_gpu_opt70(self):
        """Method: gather_nd and transpose
           Device: GPU
           Case: In-op optimization 70%"""

        self.run_test_gathernd_opt70('/gpu:0')

    def run_test_custom_opt70(self, device_name):
        """Method: custom gather_cols op
           Case: In-op optimization 70%"""

        self.run_test_opt70(fun=fun_custom,
                            device_name=device_name)

    def test_custom_cpu_opt70(self):
        """Method: custom gather_cols op
           Device: CPU
           Case: In-op optimization 70%"""

        self.run_test_custom_opt70('/cpu:0')

    def test_custom_gpu_opt70(self):
        """Method: custom gather_cols op
           Device: GPU
           Case: In-op optimization 70%"""

        self.run_test_custom_opt70('/gpu:0')

    def run_test_tfindexing_opt100(self, device_name):
        """Method: TF Indexing
           Case: Best-case (in-op optimization 100%)"""

        self.run_test_opt100(fun=fun_tfindexing, device_name=device_name)

    def test_tfindexing_cpu_opt100(self):
        """Method: TF Indexing
           Device: CPU
           Case: Best-case (in-op optimization 100%)"""

        self.run_test_tfindexing_opt100('/cpu:0')

    def test_tfindexing_gpu_opt100(self):
        """Method: TF Indexing
           Device: GPU
           Case: Best-case (in-op optimization 100%)"""

        self.run_test_tfindexing_opt100('/gpu:0')

    def run_test_gathernd_opt100(self, device_name):
        """Method: gather_nd and transpose
           Case: Best-case (in-op optimization 100%)"""

        self.run_test_opt100(fun=lambda p, i: spn.utils.gather_cols(p, i, use_gather_nd=True),
                             device_name=device_name)

    def test_gathernd_cpu_opt100(self):
        """Method: gather_nd and transpose
           Device: CPU
           Case: Best-case (in-op optimization 100%)"""

        self.run_test_gathernd_opt100('/cpu:0')

    def test_gathernd_gpu_opt100(self):
        """Method: gather_nd and transpose
           Device: GPU
           Case: Best-case (in-op optimization 100%)"""

        self.run_test_gathernd_opt100('/gpu:0')

    def run_test_custom_opt100(self, device_name):
        """Method: custom gather_cols op
           Case: Best-case (in-op optimization 100%)"""

        self.run_test_opt100(fun=fun_custom,
                             device_name=device_name)

    def test_custom_cpu_opt100(self):
        """Method: custom gather_cols op
           Device: CPU
           Case: Best-case (in-op optimization 100%)"""

        self.run_test_custom_opt100('/cpu:0')

    def test_custom_gpu_opt100(self):
        """Method: custom gather_cols op
           Device: GPU
           Case: Best-case (in-op optimization 100%)"""

        self.run_test_custom_opt100('/gpu:0')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--num-cols', default=100, type=int)
    parser.add_argument('--num-rows', default=1000, type=int)
    parser.add_argument('--num-stacked-ops', default=600, type=int)
    parser.add_argument('--log-device', default=False, type=bool)
    parser.add_argument('unittest_args', nargs='*')

    args = parser.parse_args()

    # Verify args
    if args.num_cols % 10:
        args.num_cols = (args.num_cols // 10) * 10 + 10

    TestGatherColumnsPerformance.num_cols = args.num_cols
    TestGatherColumnsPerformance.num_rows = args.num_rows
    TestGatherColumnsPerformance.num_stacked_ops = args.num_stacked_ops
    TestGatherColumnsPerformance.log_device_placement = args.log_device
    sys.argv[1:] = args.unittest_args

    tf.test.main()
