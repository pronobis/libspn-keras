#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import time
import argparse
import colorama as col
import sys
col.init()


def print1(str, file):
    if file:
        print(str, file=file)
    print(col.Fore.YELLOW + str + col.Style.RESET_ALL)


def print2(str, file):
    if file:
        print(str, file=file)
    print(col.Fore.BLUE + str + col.Style.RESET_ALL)


class Ops:

    def indexing(value, split_sizes):
        slice_indices = np.cumsum(split_sizes)
        return [value[:, start:stop] for start, stop in
                zip(np.r_[0, slice_indices], slice_indices)]

    def tf_split(value, split_sizes):
        return tf.split(value, split_sizes, axis=1)


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
            return ("%3s %11s %5s: %5s %11s %15s %14s %10s" %
                    (dev, 'op', 'dt', 'size', 'setup_time',
                     'first_run_time', 'rest_run_time', 'correct'))

        def get_res(res):
            """Helper function printing a single result."""
            return ("%15s %5s: %5d %11.2f %15.2f %14.2f %10s" %
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
            print1(get_res(res), file)
        print1(get_header("GPU"), file)
        for res in sorted(self.gpu_results, key=lambda x: x.op_name):
            print1(get_res(res), file)


class PerformanceTest:

    def __init__(self, num_value_rows, num_value_cols,
                 num_ops, num_runs, dtype,
                 without_cpu, without_gpu, log_devs, file):
        self.num_value_rows = num_value_rows
        self.num_value_cols = num_value_cols
        self.num_ops = num_ops
        self.num_runs = num_runs
        self.dtype = dtype
        self.without_cpu = without_cpu
        self.without_gpu = without_gpu
        self.log_devs = log_devs
        self.file = file

        print1("Params:", file)
        print1("- num_value_rows=%s" % num_value_rows, file)
        print1("- num_value_cols=%s" % num_value_cols, file)
        print1("- num_ops=%s" % num_ops, file)
        print1("- num_runs=%s" % num_runs, file)
        print1("- dtype=%s" % dtype, file)
        print1("", file=file)

    def _run_op_test(self, op_fun, value, split_sizes,
                     on_gpu, split_sizes_dtype):
        """Run a single test for a single op."""
        # Preparations
        op_name = op_fun.__name__
        device_name = '/gpu:0' if on_gpu else '/cpu:0'
        split_sizes = np.asarray(split_sizes, dtype=split_sizes_dtype)
        value = np.asarray(value, dtype=self.dtype.as_numpy_dtype())
        # Print
        print2("--> %s: on_gpu=%s, split_sizes_dtype=%s, value_shape=%s, split_sizes_shape=%s"
               % (op_name, on_gpu, split_sizes_dtype, value.shape, split_sizes.shape),
               self.file)
        # Compute true output with numpy
        true_slice_indices = np.cumsum(split_sizes)
        true_out = [value[:, start:stop] for start, stop in
                    zip(np.r_[0, true_slice_indices], true_slice_indices)]
        # Create graph
        tf.reset_default_graph()
        with tf.device(device_name):
            # Create input
            # We cannot use a constant here, since the operation will be
            # pre-computed on a CPU for some cases (e.g. for int64 indices)
            # To ensure that data is copied only once, we add an identity op
            # which is served the input data and connected to all ops
            value_pl = tf.placeholder(dtype=self.dtype)
            value_op = tf.identity(value_pl)
            # Create ops
            start_time = time.time()
            ops = op_fun(value_op, split_sizes)
            for _ in range(self.num_ops - 1):
                # The tuple ensures that the next op waits for the output
                # of the previous op, effectively stacking the ops
                # but using the original input every time
                ops = op_fun(tf.tuple([value_op] + ops)[0], split_sizes)
            setup_time = time.time() - start_time
        # Get num of graph ops
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
                out = sess.run(ops, feed_dict={value_pl: value})
                run_times.append(time.time() - start_time)
                # Test value
                try:
                    assert len(out) == len(true_out)
                    for o, to in zip(out, true_out):
                        np.testing.assert_array_almost_equal(o, to)
                except AssertionError:
                    output_correct = False
        # Return stats
        return OpTestResult(op_name, on_gpu, split_sizes_dtype, graph_size,
                            setup_time, run_times, output_correct)

    def _run_test(self, test_name, op_funs, value, split_sizes):
        """Run a single test for multiple ops and devices."""
        cpu_results = []
        gpu_results = []
        for op_fun in op_funs:
            if not self.without_cpu:
                cpu_results.append(
                    self._run_op_test(op_fun, value, split_sizes,
                                      on_gpu=False, split_sizes_dtype=np.int32))
                cpu_results.append(
                    self._run_op_test(op_fun, value, split_sizes,
                                      on_gpu=False, split_sizes_dtype=np.int64))
            if not self.without_gpu:
                gpu_results.append(
                    self._run_op_test(op_fun, value, split_sizes,
                                      on_gpu=True, split_sizes_dtype=np.int32))
                gpu_results.append(
                    self._run_op_test(op_fun, value, split_sizes,
                                      on_gpu=True, split_sizes_dtype=np.int64))
        return TestResults(test_name, cpu_results, gpu_results)

    def run(self):
        """Run all tests."""
        print1("Running tests:", self.file)
        results = []
        value = np.random.rand(self.num_value_rows, self.num_value_cols)

        # 1-element splits
        split_sizes = np.full(self.num_value_cols, 1)
        r = self._run_test('1elem_splits',
                           [Ops.tf_split, Ops.indexing],
                           value, split_sizes)
        results.append(r)

        # 10 splits
        split_sizes = np.full(10, self.num_value_cols / 10)
        r = self._run_test('10_splits',
                           [Ops.tf_split, Ops.indexing],
                           value, split_sizes)
        results.append(r)

        # Print results
        for res in results:
            res.print(self.file)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--num-value-rows', default=200, type=int,
                        help="Num of rows of value")
    parser.add_argument('--num-value-cols', default=100, type=int,
                        help="Num of cols of value")
    parser.add_argument('--num-ops', default=200, type=int,
                        help="Num of ops used for tests")
    parser.add_argument('--num-runs', default=10, type=int,
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

    if args.num_value_cols % 10:
        sys.exit('ERROR: num_value_cols must be divisible by 10')

    # Open a file
    f = None
    if args.save_to:
        f = open(args.save_to, 'w')

    try:
        t = PerformanceTest(args.num_value_rows, args.num_value_cols,
                            args.num_ops, args.num_runs, dtype,
                            args.without_cpu, args.without_gpu,
                            args.log_devices, f)
        t.run()
    finally:
        if f is not None:
            f.close()


if __name__ == '__main__':
    main()
