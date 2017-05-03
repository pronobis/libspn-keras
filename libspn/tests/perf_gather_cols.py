#!/usr/bin/env python3

# ------------------------------------------------------------------------
# Copyright (C) 2016 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

import tensorflow as tf
import numpy as np
from context import libspn as spn
import time
import argparse
import colorama as col
col.init()


def print1(str):
    print(col.Fore.YELLOW + str + col.Style.RESET_ALL)


def print2(str):
    print(col.Fore.BLUE + str + col.Style.RESET_ALL)


class Ops:

    def custom(params, indices):
        return spn.ops.gather_cols(params, indices)

    def gather_nd(params, indices):
        return tf.transpose(tf.gather_nd(tf.transpose(params),
                                         np.expand_dims(indices, 1)))

    def indexing(params, indices):
        return tf.stack([params[:, c] for c in indices], -1)

    def noop(params, indices):
        return params

    def gather_1d(params, indices):
        return tf.gather(params, indices)

    def gather_2d(params, indices):
        p_shape = tf.shape(params)
        p_flat = tf.reshape(params, [-1])
        i_flat = tf.reshape(tf.reshape(tf.range(0, p_shape[0]) * p_shape[1],
                                       [-1, 1]) + indices, [-1])
        return tf.reshape(tf.gather(p_flat, i_flat),
                          [p_shape[0], -1])

    def slice_1d(params, indices):
        index = indices[0]
        return tf.slice(params, [index], [1])

    def slice_2d(params, indices):
        index = indices[0]
        return tf.slice(params, [0, index], [-1, 1])


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

    def print(self):
        def print_header(dev):
            print1("%3s %11s: %5s %11s %15s %14s %10s" %
                   (dev, 'op', 'size', 'setup_time',
                    'first_run_time', 'rest_run_time', 'correct'))

        def print_res(res):
            """Helper function printing a single result."""
            print1("%15s: %5d %11.2f %15.2f %14.2f %10s" %
                   (res.op_name, res.graph_size,
                    res.setup_time * 1000, res.run_times[0] * 1000,
                    np.mean(res.run_times[1:]) * 1000,
                    res.output_correct))

        # Print results
        print()
        print1("-----------------------")
        print1("%s" % self.test_name)
        print1("-----------------------")
        print_header("CPU")
        for res in sorted(self.cpu_results, key=lambda x: x.op_name):
            print_res(res)
        print_header("GPU")
        for res in sorted(self.gpu_results, key=lambda x: x.op_name):
            print_res(res)


class PerformanceTest:

    def __init__(self, num_param_rows, num_param_cols, num_indices,
                 num_ops, num_runs, dtype, with_indexing,
                 without_cpu, without_gpu, log_devs):
        self.num_param_rows = num_param_rows
        self.num_param_cols = num_param_cols
        self.num_indices = num_indices
        self.num_ops = num_ops
        self.num_runs = num_runs
        self.dtype = dtype
        self.with_indexing = with_indexing
        self.without_cpu = without_cpu
        self.without_gpu = without_gpu
        self.log_devs = log_devs

        print1("Params:")
        print1("- num_param_rows=%s" % num_param_rows)
        print1("- num_param_cols=%s" % num_param_cols)
        print1("- num_indices=%s" % num_indices)
        print1("- num_ops=%s" % num_ops)
        print1("- num_runs=%s" % num_runs)
        print1("- dtype=%s" % dtype)

    def _run_op_test(self, op_fun, params, indices, on_gpu):
        """Run a single test for a single op."""
        # Preparations
        op_name = op_fun.__name__
        device_name = '/gpu:0' if on_gpu else '/cpu:0'
        indices = np.asarray(indices, dtype=np.int32)
        params = np.asarray(params, dtype=self.dtype.as_numpy_dtype())
        # Print
        print2("--> %s: on_gpu=%s, params_shape=%s, indices_shape=%s" %
               (op_name, on_gpu, params.shape, indices.shape))
        # Compute true output with numpy
        if params.ndim == 1:
            true_out = params[indices]
        else:
            true_out = params[:, indices]
        # Create graph
        tf.reset_default_graph()
        with tf.device(device_name):
            start_time = time.time()
            # Create input constants
            params_pl = tf.placeholder(dtype=self.dtype)
            # Create ops
            ops = [op_fun(params_pl, indices)
                   for i in range(self.num_ops)]
            # Group all
            tup = tf.tuple(ops)
            grp = tf.group(*ops)
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
                sess.run(grp, feed_dict={params_pl: params})
                run_times.append(time.time() - start_time)
            # Run the tuple to get and check the output
            out = sess.run(tup, feed_dict={params_pl: params})
            for o in out:
                try:
                    np.testing.assert_array_almost_equal(o, true_out)
                except AssertionError:
                    output_correct = False
        # Return stats
        return OpTestResult(op_name, on_gpu, graph_size, setup_time,
                            run_times, output_correct)

    def _run_test(self, test_name, op_funs, params, indices):
        """Run a single test for multiple ops and devices."""
        cpu_results = []
        gpu_results = []
        for op_fun in op_funs:
            if not self.without_cpu:
                cpu_results.append(
                    self._run_op_test(op_fun, params, indices, on_gpu=False))
            if not self.without_gpu:
                gpu_results.append(
                    self._run_op_test(op_fun, params, indices, on_gpu=True))
        return TestResults(test_name, cpu_results, gpu_results)

    def _run_1d(self):
        """Run all 1D tests."""

        results = []
        params = np.random.rand(self.num_param_cols)

        # 1 index
        indices = np.random.randint(low=0, high=self.num_param_cols,
                                    size=1)
        r = self._run_test('1d_1index',
                           [Ops.custom, Ops.gather_1d, Ops.slice_1d],
                           params, indices)
        results.append(r)

        # Passthrough
        indices = range(self.num_param_cols)
        r = self._run_test('1d_passthrough_%sindices' % self.num_param_cols,
                           [Ops.custom, Ops.gather_1d, Ops.noop],
                           params, indices)
        results.append(r)

        # Partially optimized
        shuffled_ind = [0, 1, 2, 9, 5, 4, 6, 7, 8, 3]
        indices = np.empty((self.num_param_cols // 10, 10))
        indices[:] = shuffled_ind
        indices = indices.ravel()
        r = self._run_test('1d_opt_%sindices' % self.num_param_cols,
                           [Ops.custom, Ops.gather_1d],
                           params, indices)
        results.append(r)

        # Worst case
        indices = range(self.num_param_cols - 1, -1, -1)
        r = self._run_test('1d_worst_%sindices' % self.num_param_cols,
                           [Ops.custom, Ops.gather_1d],
                           params, indices)
        results.append(r)

        # Random
        indices = np.random.randint(low=0, high=self.num_param_cols,
                                    size=self.num_indices)
        r = self._run_test('1d_random_%dindices' % self.num_indices,
                           [Ops.custom, Ops.gather_1d],
                           params, indices)
        results.append(r)

        return results

    def _run_2d(self):
        """Run all 2D tests."""

        results = []
        params = np.random.rand(self.num_param_rows, self.num_param_cols)

        # 1 index
        indices = np.random.randint(low=0, high=self.num_param_cols,
                                    size=1)
        r = self._run_test('2d_1index',
                           [Ops.custom, Ops.gather_nd, Ops.slice_2d],
                           params, indices)
        results.append(r)

        # Passthrough
        indices = range(self.num_param_cols)
        r = self._run_test('2d_passthrough_%sindices' % self.num_param_cols,
                           [Ops.custom, Ops.gather_nd, Ops.noop],
                           params, indices)
        results.append(r)

        # Partially optimized
        shuffled_ind = [0, 1, 2, 9, 5, 4, 6, 7, 8, 3]
        indices = np.empty((self.num_param_cols // 10, 10))
        indices[:] = shuffled_ind
        indices = indices.ravel()
        r = self._run_test('2d_opt_%sindices' % self.num_param_cols,
                           [Ops.custom, Ops.gather_nd] +
                           ([Ops.indexing] if self.with_indexing else []),
                           params, indices)
        results.append(r)

        # Worst case
        indices = range(self.num_param_cols - 1, -1, -1)
        r = self._run_test('2d_worst_%sindices' % self.num_param_cols,
                           [Ops.custom, Ops.gather_nd] +
                           ([Ops.indexing] if self.with_indexing else []),
                           params, indices)
        results.append(r)

        # Random
        indices = np.random.randint(low=0, high=self.num_param_cols,
                                    size=self.num_indices)
        r = self._run_test('2d_random_%dindices' % self.num_indices,
                           [Ops.custom, Ops.gather_nd] +
                           ([Ops.indexing] if self.with_indexing else []),
                           params, indices)
        results.append(r)

        return results

    def run(self):
        """Run all tests."""

        results = []
        results += self._run_1d()
        results += self._run_2d()

        for res in results:
            res.print()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_param_rows', default=200, type=int)
    parser.add_argument('--num_param_cols', default=100, type=int)
    parser.add_argument('--num_indices', default=50, type=int)
    parser.add_argument('--num_ops', default=100, type=int)
    parser.add_argument('--num_runs', default=10, type=int)
    parser.add_argument('--log-devices', action='store_true')
    parser.add_argument('--with-indexing', action='store_true')
    parser.add_argument('--without-cpu', action='store_true')
    parser.add_argument('--without-gpu', action='store_true')
    dtype = tf.float32
    args = parser.parse_args()

    print(args.with_indexing)

    t = PerformanceTest(args.num_param_rows, args.num_param_cols,
                        args.num_indices, args.num_ops, args.num_runs,
                        dtype, args.with_indexing,
                        args.without_cpu, args.without_gpu,
                        args.log_devices)
    t.run()


if __name__ == '__main__':
    main()
