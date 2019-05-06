#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import time
import argparse
import colorama as col
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

    def tensordot(a, b):
        return tf.tensordot(a, b, axes=[[1], [1]])

    def reduction(a, b):
        return tf.reduce_sum(tf.expand_dims(a, axis=1) * b, axis=2)

    def matmul(a, b):
        if len(b.shape) == 2:
            return tf.matmul(a, b, transpose_b=True)
        else:
            return tf.squeeze(tf.matmul(tf.expand_dims(a, axis=1), b,
                                        transpose_b=True))

    def reduction_by_matmul(a, b):
        bcasted = tf.expand_dims(a, axis=1) * b
        ones = tf.ones(shape=(tf.shape(a)[0], 1, a.shape.as_list()[1]),
                       dtype=bcasted.dtype)
        return tf.squeeze(tf.matmul(ones, bcasted, transpose_b=True))


class OpTestResult:
    """Result of a single test of a single op."""

    def __init__(self, op_name, on_gpu, graph_size, setup_time,
                 run_times, output_correct):
        self.op_name = op_name
        self.on_gpu = on_gpu
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
            return ("%3s %15s: %5s %11s %15s %14s %10s" %
                    (dev, 'op', 'size', 'setup_time',
                     'first_run_time', 'rest_run_time', 'correct'))

        def get_res(res):
            """Helper function printing a single result."""
            return ("%19s: %5d %11.2f %15.2f %14.2f %10s" %
                    (res.op_name, res.graph_size,
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

    def __init__(self, num_a_rows, num_a_cols, num_b_rows,
                 num_ops, num_runs, dtype,
                 without_cpu, without_gpu, log_devs, file):
        self.num_a_rows = num_a_rows
        self.num_a_cols = num_a_cols
        self.num_b_rows = num_b_rows
        self.num_ops = num_ops
        self.num_runs = num_runs
        self.dtype = dtype
        self.without_cpu = without_cpu
        self.without_gpu = without_gpu
        self.log_devs = log_devs
        self.file = file

        print1("Params:", file)
        print1("- num_a_rows=%s" % num_a_rows, file)
        print1("- num_a_cols=%s" % num_a_cols, file)
        print1("- num_b_rows=%s" % num_b_rows, file)
        print1("- num_ops=%s" % num_ops, file)
        print1("- num_runs=%s" % num_runs, file)
        print1("- dtype=%s" % dtype, file)
        print1("", file=file)

    def _run_op_test(self, op_fun, a, b, on_gpu):
        """Run a single test for a single op."""
        # Preparations
        op_name = op_fun.__name__
        device_name = '/gpu:0' if on_gpu else '/cpu:0'
        a = np.asarray(a, dtype=self.dtype.as_numpy_dtype())
        b = np.asarray(b, dtype=self.dtype.as_numpy_dtype())
        # Print
        print2("--> %s: on_gpu=%s, a_shape=%s, b_shape=%s"
               % (op_name, on_gpu, a.shape, b.shape),
               self.file)
        # true_out = a@b.transpose()
        true_out = np.sum(a.reshape(a.shape[0], 1, -1) * b, axis=2)
        # Create graph
        tf.reset_default_graph()
        with tf.device(device_name):
            # Create input
            # We cannot use a constant here, since the operation will be
            # pre-computed on a CPU for some cases (e.g. for int64 indices)
            # To ensure that data is copied only once, we add an identity op
            # which is served the input data and connected to all ops
            a_pl = tf.placeholder(dtype=self.dtype, shape=(None, a.shape[1]))
            a_op = tf.identity(a_pl)
            if b.ndim == 2:
                b_pl = tf.placeholder(dtype=self.dtype, shape=(None, b.shape[1]))
            else:
                b_pl = tf.placeholder(dtype=self.dtype, shape=(None, b.shape[1],
                                      b.shape[2]))
            b_op = tf.identity(b_pl)
            # Create ops
            start_time = time.time()
            ops = op_fun(a_op, b_op)
            for _ in range(self.num_ops - 1):
                # The tuple ensures that the next op waits for the output
                # of the previous op, effectively stacking the ops
                # but using the original input every time
                ops = op_fun(tf.tuple([a_op, ops])[0], b_op)
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
                out = sess.run(ops, feed_dict={a_pl: a, b_pl: b})
                run_times.append(time.time() - start_time)
                # Test value
                try:
                    np.testing.assert_array_almost_equal(out, true_out, decimal=5)
                except AssertionError:
                    output_correct = False
        # Return stats
        return OpTestResult(op_name, on_gpu, graph_size, setup_time,
                            run_times, output_correct)

    def _run_test(self, test_name, op_funs, a, b):
        """Run a single test for multiple ops and devices."""
        cpu_results = []
        gpu_results = []
        for op_fun in op_funs:
            if not self.without_cpu:
                cpu_results.append(self._run_op_test(op_fun, a, b, on_gpu=False))
            if not self.without_gpu:
                gpu_results.append(self._run_op_test(op_fun, a, b, on_gpu=True))
        return TestResults(test_name, cpu_results, gpu_results)

    def run(self):
        """Run all tests."""
        print1("Running tests:", self.file)
        results = []
        a = np.random.rand(self.num_a_rows, self.num_a_cols)
        b = np.random.rand(self.num_b_rows, self.num_a_cols)
        r = self._run_test('case1_2d (simulate multiplication with weights)',
                           [Ops.reduction, Ops.matmul, Ops.reduction_by_matmul],
                           a, b)
        results.append(r)

        a = np.random.rand(self.num_a_rows, self.num_a_cols)
        b = np.random.rand(self.num_a_rows, self.num_b_rows, self.num_a_cols)
        r = self._run_test('case2_3d (simulate multiplication with IndicatorLeaf)',
                           [Ops.reduction, Ops.matmul, Ops.reduction_by_matmul],
                           a, b)
        results.append(r)

        # Print results
        for res in results:
            res.print(self.file)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--num-a-rows', default=200, type=int,
                        help="Num of rows of matrix A")
    parser.add_argument('--num-a-cols', default=30, type=int,
                        help="Num of cols of matrix A")
    parser.add_argument('--num-b-rows', default=5, type=int,
                        help="Num of rows of matrix B")
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

    # Open a file
    f = None
    if args.save_to:
        f = open(args.save_to, 'w')

    try:
        t = PerformanceTest(args.num_a_rows, args.num_a_cols,
                            args.num_b_rows, args.num_ops,
                            args.num_runs, dtype,
                            args.without_cpu, args.without_gpu,
                            args.log_devices, f)
        t.run()
    finally:
        if f is not None:
            f.close()


if __name__ == '__main__':
    main()
