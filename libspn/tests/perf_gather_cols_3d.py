#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
from random import shuffle
from context import libspn as spn
from itertools import chain
import time
import argparse
import sys
import colorama as col
col.init()

red = col.Fore.RED
blue = col.Fore.BLUE
green = col.Fore.GREEN
yellow = col.Fore.YELLOW
magenta = col.Fore.MAGENTA


def print1(str, file, color=yellow):
    if file:
        print(str, file=file)
    print(color + str + col.Style.RESET_ALL)


def print2(str, file):
    if file:
        print(str, file=file)
    print(blue + str + col.Style.RESET_ALL)


class Ops:

    def custom_gather_3d(params, indices, padding):
        if padding:
            # Fill indices of padded columns with '-1'
            indices_cols = max([ind.size for ind in indices])
            indices = [np.append(ind, np.ones(indices_cols-ind.size,
                       dtype=ind.dtype)*-1) for ind in indices]
            # Convert the list of indices arrays to an indices matrix
            indices = np.vstack(indices)
        return spn.ops.gather_cols_3d(params, indices, padding)

    def custom_gather(params, indices, padding):
        if padding:
            indices_cols = max([ind.size for ind in indices])
            return tf.stack([tf.pad(spn.ops.gather_cols(params, ind),
                                    [[0, 0], [0, indices_cols-ind.size]])
                             if ind.size < indices_cols else
                             spn.ops.gather_cols(params, ind)
                             for ind in indices], 1)
        else:
            return tf.reshape(spn.ops.gather_cols(params, list(chain.from_iterable(
                              indices))), [-1, len(indices), len(indices[0])])

    def tf_gather(params, indices, padding):
        if padding:
            indices_cols = max([ind.size for ind in indices])
            return tf.stack([tf.pad(tf.gather(params, ind, axis=1),
                             [[0, 0], [0, indices_cols-ind.size]])
                             if ind.size < indices_cols else
                             tf.gather(params, ind, axis=1)
                            for ind in indices], 1)
        else:
            return tf.reshape(tf.gather(params, list(chain.from_iterable(indices)),
                                        axis=1), [-1, len(indices), len(indices[0])])


class OpTestResult:
    """Result of a single test of a single op."""

    def __init__(self, op_name, on_gpu, index_dtype, graph_size, setup_time,
                 run_times, output_correct):
        self.op_name = op_name
        self.on_gpu = on_gpu
        self.index_dtype = index_dtype
        self.graph_size = graph_size
        self.setup_time = setup_time
        self.run_times = run_times
        self.output_correct = output_correct


class TestResults:
    """Results for a single test for multiple ops and devices."""

    def __init__(self, test_name, cpu_results, gpu_results):
        self.test_name = test_name
        self.cpu_results = cpu_results
        self.gpu_results = gpu_results

    def print(self, file):
        def get_header(dev):
            return ("%3s %14s %5s: %5s %11s %15s %14s %10s" %
                    (dev, 'op', 'dt', 'size', 'setup_time',
                     'first_run_time', 'rest_run_time', 'correct'))

        def get_res(res):
            """Helper function printing a single result."""
            return ("%18s %5s: %5d %11.2f %15.2f %14.2f %10s" %
                    (res.op_name, str(res.index_dtype)[14:-2], res.graph_size,
                     res.setup_time * 1000, res.run_times[0] * 1000,
                     np.mean(res.run_times[1:]) * 1000,
                     res.output_correct))

        # Print results
        print1("\n-----------------------", file)
        print1("%s" % self.test_name, file)
        print1("-----------------------", file)
        print1(get_header("CPU"), file)
        for res in sorted(self.cpu_results, key=lambda x: x.op_name):
            print1(get_res(res), file, (red if res.op_name is "custom_gather" else
                   green if res.op_name is "custom_gather_3d" else magenta))
        print1(get_header("GPU"), file)
        for res in sorted(self.gpu_results, key=lambda x: x.op_name):
            print1(get_res(res), file, (red if res.op_name is "custom_gather" else
                   green if res.op_name is "custom_gather_3d" else magenta))


class PerformanceTest:

    def __init__(self, num_param_rows, num_param_cols, num_indices_rows,
                 num_indices_cols, num_ops, num_runs, dtype, without_cpu,
                 without_gpu, log_devs, file):
        self.num_param_rows = num_param_rows
        self.num_param_cols = num_param_cols
        self.num_indices_rows = num_indices_rows
        self.num_indices_cols = num_indices_cols
        self.num_ops = num_ops
        self.num_runs = num_runs
        self.dtype = dtype
        self.without_cpu = without_cpu
        self.without_gpu = without_gpu
        self.log_devs = log_devs
        self.file = file
        self.test_failed = False

        print1("Params:", file)
        print1("- num_param_rows=%s" % num_param_rows, file)
        print1("- num_param_cols=%s" % num_param_cols, file)
        print1("- num_indices_rows=%s" % num_indices_rows, file)
        print1("- num_indices_cols=%s" % num_indices_cols, file)
        print1("- num_ops=%s" % num_ops, file)
        print1("- num_runs=%s" % num_runs, file)
        print1("- dtype=%s" % dtype, file)
        print1("", file=file)

    def _run_op_test(self, op_fun, params, indices, padding,
                     on_gpu, index_dtype):
        """Run a single test for a single op."""
        # Preparations
        op_name = op_fun.__name__
        device_name = '/gpu:0' if on_gpu else '/cpu:0'
        # Convert params and indices to array like
        try:
            indices = np.asarray(indices, dtype=index_dtype)
        except ValueError:
            indices = [np.asarray(ind, dtype=index_dtype) for ind in indices]
        params = np.asarray(params, dtype=self.dtype.as_numpy_dtype())
        # Print
        print2("--> %s: on_gpu=%s, padded=%s, index_dtype=%s, params_shape=%s"
               % (op_name, on_gpu, padding, index_dtype, params.shape),
               self.file)
        # Compute true output with numpy
        if padding:
            # Insert a column of zeros to the last column of params
            params_with_zero = np.insert(params, self.num_param_cols,
                                         np.zeros(self.num_param_rows,
                                                  dtype=self.dtype.as_numpy_dtype()),
                                         axis=-1)
            # Fill indices of padded columns with index of the last-column of params
            indices_filled = [np.insert(ind, ind.size, np.full((self.num_indices_cols-ind.size),
                              self.num_param_cols, dtype=ind.dtype)) for ind in indices]
            indices_filled = np.array(indices_filled)
            true_out = params_with_zero[:, indices_filled]
        else:
            true_out = params[:, indices]

        # Create graph
        tf.reset_default_graph()
        with tf.device(device_name):
            # Create input
            # We cannot use a constant here, since the operation will be pre-computed
            # on a CPU for some cases (e.g. for int64 indices)
            # To ensure that data is copied only once, we add an identity op
            # which is served the input data and connected to all ops
            params_pl = tf.placeholder(dtype=self.dtype)
            params_op = tf.identity(params_pl)
            # Create ops
            start_time = time.time()
            ops = op_fun(params_op, indices, padding)
            for _ in range(self.num_ops - 1):
                # The tuple ensures that the next op waits for the output
                # of the previous op, effectively stacking the ops
                # but using the original input every time
                ops = op_fun(tf.tuple([params_op, ops])[0], indices, padding)
            setup_time = time.time() - start_time
        # Get num of TF graph ops
        graph_size = len(tf.get_default_graph().get_operations())
        # Run multiple times
        output_correct = True
        with tf.Session(config=tf.ConfigProto(
                allow_soft_placement=False,
                log_device_placement=self.log_devs)) as sess:
            run_times = []
            for n in range(self.num_runs):
                # Run
                start_time = time.time()
                out = sess.run(ops, feed_dict={params_pl: params})
                run_times.append(time.time() - start_time)
                # Test value
                try:
                    np.testing.assert_array_almost_equal(out, true_out)
                except AssertionError:
                    output_correct = False
                    self.test_failed = True
        # Return stats
        return OpTestResult(op_name, on_gpu, index_dtype, graph_size, setup_time,
                            run_times, output_correct)

    def _run_test(self, test_name, op_funs, params, indices, padding):
        """Run a single test for multiple ops, devices and dtypes."""
        cpu_results = []
        gpu_results = []
        for op_fun in op_funs:
            if not self.without_cpu:
                cpu_results.append(
                    self._run_op_test(op_fun, params, indices, padding,
                                      on_gpu=False, index_dtype=np.int32))
                cpu_results.append(
                    self._run_op_test(op_fun, params, indices, padding,
                                      on_gpu=False, index_dtype=np.int64))
            if not self.without_gpu:
                gpu_results.append(
                    self._run_op_test(op_fun, params, indices, padding,
                                      on_gpu=True, index_dtype=np.int32))
                gpu_results.append(
                    self._run_op_test(op_fun, params, indices, padding,
                                      on_gpu=True, index_dtype=np.int64))
        return TestResults(test_name, cpu_results, gpu_results)

    def run(self):
        """Run all tests."""
        print1("Running tests:", self.file)

        results = []
        params = np.random.rand(self.num_param_rows, self.num_param_cols)

        # Non-padded case
        # A 2D indices as np.array
        indices = np.random.randint(low=0, high=self.num_param_cols,
                                    size=(self.num_indices_rows, self.num_indices_cols))

        r = self._run_test('Non-padded',
                           [Ops.custom_gather_3d, Ops.custom_gather, Ops.tf_gather],
                           params, indices, padding=False)
        results.append(r)

        # Padded case
        # A list of 1D indices as np.arrays
        indices = []

        # # Indices list with indices size in [1, ind_cols]
        # ind_length = self.num_indices_cols
        # for i in range(self.num_indices_rows):
        #     indices.append(np.random.randint(self.num_param_cols, size=ind_length))
        #     ind_length = np.random.randint(1, self.num_indices_cols+1)
        # shuffle(indices)

        # Lit of indices arrays, each with size: [ind_cols, ind_cols-1, ind_cols-2, ...
        #                                         ... ind_cols-ind_rows-1]
        for i in range(self.num_indices_rows):
            indices.append(np.random.randint(self.num_param_cols,
                                             size=self.num_indices_cols-i))
        shuffle(indices)

        # indices = [np.random.randint(self.num_param_cols, size=self.num_indices_cols)
        #            for _ in range(self.num_indices_rows)]

        r = self._run_test('Padded',
                           [Ops.custom_gather_3d, Ops.custom_gather, Ops.tf_gather],
                           params, indices, padding=True)
        results.append(r)

        # Print results
        for res in results:
            res.print(self.file)

        if self.test_failed:
            print("\n ATLEAST ONE TEST FAILED!")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--num-param-rows', default=200, type=int,
                        help="Num of rows of params")
    parser.add_argument('--num-param-cols', default=100, type=int,
                        help="Num of cols of params")
    parser.add_argument('--num-indices-rows', default=50, type=int,
                        help="Num of rows of indices used for SOME tests")
    parser.add_argument('--num-indices-cols', default=100, type=int,
                        help="Num of cols of indices used for SOME tests")
    parser.add_argument('--num-ops', default=10, type=int,
                        help="Num of ops used for tests")
    parser.add_argument('--num-runs', default=50, type=int,
                        help="Number of times each test is run")
    parser.add_argument('--log-devices', action='store_true',
                        help="Log on which device op is run. Affects run time!")
    parser.add_argument('--without-cpu', action='store_true',
                        help="Do not run CPU tests")
    parser.add_argument('--without-gpu', action='store_true',
                        help="Do not run GPU tests")
    parser.add_argument('--save-to', default='', type=str,
                        help="Save results to file")
    dtype = tf.float32
    args = parser.parse_args()

    # Needed to generate indices for partially optimized cases
    if args.num_indices_rows > args.num_indices_cols:
        sys.exit('ERROR: num_indices_rows must be <= num_indices_cols')

    # Open a file
    f = None
    if args.save_to:
        f = open(args.save_to, 'w')

    try:
        t = PerformanceTest(args.num_param_rows, args.num_param_cols,
                            args.num_indices_rows, args.num_indices_cols,
                            args.num_ops, args.num_runs, dtype, args.without_cpu,
                            args.without_gpu, args.log_devices, f)
        t.run()
    finally:
        if f is not None:
            f.close()


if __name__ == '__main__':
    main()
