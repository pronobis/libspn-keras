#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
from context import libspn as spn
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

    def custom(params, indices, out_size):
        return spn.ops.scatter_cols(params, indices,
                                    num_out_col=out_size, pad_elem=0)

    def noop(params, indices, out_size):
        return params

    def pad_1d(params, indices, out_size):
        return tf.pad(params, [[indices[0], out_size - indices[0] - 1]])

    def pad_2d(params, indices, out_size):
        return tf.pad(params, [[0, 0],
                               [indices[0], out_size - indices[0] - 1]])

    def gather_1d(params, indices, out_size):
        with_zeros = tf.concat(values=([0], params), axis=0)
        gather_indices = np.zeros(out_size, dtype=indices.dtype)
        gather_indices[indices] = np.arange(indices.size) + 1
        return tf.gather(with_zeros, gather_indices)

    def gather_nd(params, indices, out_size):
        zero_col = tf.zeros((tf.shape(params)[0], 1), dtype=params.dtype)
        with_zeros = tf.concat(values=(zero_col, params), axis=1)
        gather_indices = np.zeros(out_size, dtype=indices.dtype)
        gather_indices[indices] = np.arange(indices.size) + 1
        return tf.transpose(tf.gather_nd(tf.transpose(with_zeros),
                                         np.expand_dims(gather_indices, 1)))

    def custom_gather_cols(params, indices, out_size):
        zero_col = tf.zeros((tf.shape(params)[0], 1), dtype=params.dtype)
        with_zeros = tf.concat(values=(zero_col, params), axis=1)
        gather_indices = np.zeros(out_size, dtype=indices.dtype)
        gather_indices[indices] = np.arange(indices.size) + 1
        return spn.ops.gather_cols(with_zeros, gather_indices)


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
            print1(get_res(res), file)
        print1(get_header("GPU"), file)
        for res in sorted(self.gpu_results, key=lambda x: x.op_name):
            print1(get_res(res), file)


class PerformanceTest:

    def __init__(self, num_param_rows, num_param_cols, out_size,
                 num_ops, num_runs, dtype,
                 without_cpu, without_gpu, log_devs, file):
        self.num_param_rows = num_param_rows
        self.num_param_cols = num_param_cols
        self.out_size = out_size
        self.num_ops = num_ops
        self.num_runs = num_runs
        self.dtype = dtype
        self.without_cpu = without_cpu
        self.without_gpu = without_gpu
        self.log_devs = log_devs
        self.file = file

        print1("Params:", file)
        print1("- num_param_rows=%s" % num_param_rows, file)
        print1("- num_param_cols=%s" % num_param_cols, file)
        print1("- out_size=%s" % out_size, file)
        print1("- num_ops=%s" % num_ops, file)
        print1("- num_runs=%s" % num_runs, file)
        print1("- dtype=%s" % dtype, file)
        print1("", file=file)

    def _run_op_test(self, op_fun, params, indices, out_size,
                     on_gpu, index_dtype):
        """Run a single test for a single op."""
        # Preparations
        op_name = op_fun.__name__
        device_name = '/gpu:0' if on_gpu else '/cpu:0'
        indices = np.asarray(indices, dtype=index_dtype)
        params = np.asarray(params, dtype=self.dtype.as_numpy_dtype())
        # Print
        print2("--> %s: on_gpu=%s, index_dtype=%s, params_shape=%s, indices_shape=%s, out_size=%s"
               % (op_name, on_gpu, index_dtype, params.shape, indices.shape, out_size),
               self.file)
        # Compute true output with numpy
        if params.ndim == 1:
            true_out = np.zeros(out_size, dtype=params.dtype)
            true_out[indices] = params
        else:
            true_out = np.zeros((params.shape[0], out_size), dtype=params.dtype)
            true_out[:, indices] = params
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
            ops = op_fun(params_op, indices, out_size)
            for _ in range(self.num_ops - 1):
                # The tuple ensures that the next op waits for the output
                # of the previous op, effectively stacking the ops
                # but using the original input every time
                ops = op_fun(tf.tuple([params_op, ops])[0], indices, out_size)
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
                out = sess.run(ops, feed_dict={params_pl: params})
                run_times.append(time.time() - start_time)
                # Test value
                try:
                    np.testing.assert_array_almost_equal(out, true_out)
                except AssertionError:
                    output_correct = False
        # Return stats
        return OpTestResult(op_name, on_gpu, index_dtype, graph_size, setup_time,
                            run_times, output_correct)

    def _run_test(self, test_name, op_funs, params, indices, out_size):
        """Run a single test for multiple ops and devices."""
        cpu_results = []
        gpu_results = []
        for op_fun in op_funs:
            if not self.without_cpu:
                cpu_results.append(
                    self._run_op_test(op_fun, params, indices, out_size,
                                      on_gpu=False, index_dtype=np.int32))
                cpu_results.append(
                    self._run_op_test(op_fun, params, indices, out_size,
                                      on_gpu=False, index_dtype=np.int64))
            if not self.without_gpu:
                gpu_results.append(
                    self._run_op_test(op_fun, params, indices, out_size,
                                      on_gpu=True, index_dtype=np.int32))
                gpu_results.append(
                    self._run_op_test(op_fun, params, indices, out_size,
                                      on_gpu=True, index_dtype=np.int64))
        return TestResults(test_name, cpu_results, gpu_results)

    def _run_1d(self):
        """Run all 1D tests."""
        results = []

        # 1 index
        params = np.random.rand(1)
        indices = np.random.randint(low=0, high=self.out_size, size=1)
        r = self._run_test('1d_1index',
                           [Ops.custom, Ops.pad_1d, Ops.gather_1d],
                           params, indices, self.out_size)
        results.append(r)

        # Passthrough
        params = np.random.rand(self.out_size)
        indices = range(self.out_size)
        r = self._run_test('1d_passthrough_%sindices' % self.out_size,
                           [Ops.custom, Ops.noop, Ops.gather_1d],
                           params, indices, self.out_size)
        results.append(r)

        # Reverse params
        params = np.random.rand(self.out_size)
        indices = range(self.out_size - 1, -1, -1)
        r = self._run_test('1d_reverse_%sindices' % self.out_size,
                           [Ops.custom, Ops.gather_1d],
                           params, indices, self.out_size)
        results.append(r)

        # Random
        params = np.random.rand(self.num_param_cols)
        # Random, integers without repetitions
        indices = np.random.choice(self.out_size, size=self.num_param_cols,
                                   replace=False)
        r = self._run_test('1d_random_%dindices' % self.num_param_cols,
                           [Ops.custom, Ops.gather_1d],
                           params, indices, self.out_size)
        results.append(r)

        return results

    def _run_2d(self):
        """Run all 2D tests."""
        results = []

        # 1 index
        params = np.random.rand(self.num_param_rows, 1)
        indices = np.random.randint(low=0, high=self.out_size, size=1)
        r = self._run_test('2d_1index',
                           [Ops.custom, Ops.pad_2d, Ops.gather_nd,
                            Ops.custom_gather_cols],
                           params, indices, self.out_size)
        results.append(r)

        # Passthrough
        params = np.random.rand(self.num_param_rows, self.out_size)
        indices = range(self.out_size)
        r = self._run_test('2d_passthrough_%sindices' % self.out_size,
                           [Ops.custom, Ops.noop, Ops.gather_nd,
                            Ops.custom_gather_cols],
                           params, indices, self.out_size)
        results.append(r)

        # Reverse params
        params = np.random.rand(self.num_param_rows, self.out_size)
        indices = range(self.out_size - 1, -1, -1)
        r = self._run_test('2d_reverse_%sindices' % self.out_size,
                           [Ops.custom, Ops.gather_nd, Ops.custom_gather_cols],
                           params, indices, self.out_size)
        results.append(r)

        # Random
        params = np.random.rand(self.num_param_rows, self.num_param_cols)
        # Random, integers without repetitions
        indices = np.random.choice(self.out_size, size=self.num_param_cols,
                                   replace=False)
        r = self._run_test('2d_random_%dindices' % self.num_param_cols,
                           [Ops.custom, Ops.gather_nd, Ops.custom_gather_cols],
                           params, indices, self.out_size)
        results.append(r)

        return results

    def run(self):
        """Run all tests."""
        print1("Running tests:", self.file)
        results = []
        results += self._run_1d()
        results += self._run_2d()

        # Print results
        for res in results:
            res.print(self.file)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--num-param-rows', default=200, type=int,
                        help="Num of rows of params")
    parser.add_argument('--num-param-cols', default=50, type=int,
                        help="Num of cols of params")
    parser.add_argument('--out-size', default=100, type=int,
                        help="Size of the output")
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

    # Needed to generate indices for partially optimized cases
    if args.num_param_cols % 10:
        sys.exit('ERROR: num_param_cols must be divisible by 10')

    # Open a file
    f = None
    if args.save_to:
        f = open(args.save_to, 'w')

    try:
        t = PerformanceTest(args.num_param_rows, args.num_param_cols,
                            args.out_size, args.num_ops,
                            args.num_runs, dtype,
                            args.without_cpu, args.without_gpu,
                            args.log_devices, f)
        t.run()
    finally:
        if f is not None:
            f.close()


if __name__ == '__main__':
    main()
