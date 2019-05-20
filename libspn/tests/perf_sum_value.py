#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
from context import libspn as spn
import time
import argparse
import colorama as col
import sys
from tensorflow.python.client import timeline
import os
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

    def sum(inputs, indices, latent_indicators, num_sums, inf_type, log=False, output=None):
        if indices is None:
            inputs = [inputs]
        else:
            inputs = [(inputs, indices)]

        # Generate 'num_sums' Sum nodes, connecting each to inputs and latent_indicators
        s = []
        for iv in latent_indicators:
            s = s + [spn.Sum(*inputs, latent_indicators=iv)]
            # Generate weights for each Sum node
            s[-1].generate_weights()

        # Connect all sum nodes to a single root Sum node and generate its weights
        root = spn.Sum(*s)
        root.generate_weights()

        if log:
            value_op = root.get_log_value(inference_type=inf_type)
        else:
            value_op = root.get_value(inference_type=inf_type)

        return spn.initialize_weights(root), value_op

    def par_sums(inputs, indices, latent_indicators, num_sums, inf_type, log=True, output=None):
        if indices is None:
            inputs = [inputs]
        else:
            inputs = [(inputs, indices)]

        # Generate a single ParallelSums node, modeling 'num_sums' sum nodes
        # within, connecting it to inputs and latent_indicators
        s = spn.ParallelSums(*inputs, num_sums=num_sums, latent_indicators=latent_indicators[0])
        # Generate weights of the ParallelSums node
        s.generate_weights()

        # Connect the ParallelSums nodes to a single root Sum node and generate
        # its weights
        root = spn.Sum(s)
        root.generate_weights()

        if log:
            value_op = root.get_log_value(inference_type=inf_type)
        else:
            value_op = root.get_value(inference_type=inf_type)

        return spn.initialize_weights(root), value_op

    def sums(inputs, indices, latent_indicators, num_sums, inf_type, log=True, output=None):
        if indices is None:
            inputs = [inputs for _ in range(num_sums)]
        else:
            inputs = [(inputs, indices) for _ in range(num_sums)]

        # Generate a single Sums node, modeling 'num_sums' sum nodes within,
        # connecting it to inputs and latent_indicators
        s = spn.Sums(*inputs, num_sums=num_sums, latent_indicators=latent_indicators[0])
        # Generate weights of the Sums node
        s.generate_weights()

        # Connect the Sums nodes to a single root Sum node and generate its weights
        root = spn.Sum(s)
        root.generate_weights()

        if log:
            value_op = root.get_log_value(inference_type=inf_type)
        else:
            value_op = root.get_value(inference_type=inf_type)

        return spn.initialize_weights(root), value_op


class OpTestResult:
    """Result of a single test of a single op."""

    def __init__(self, op_name, on_gpu, graph_size, indices, latent_indicators, setup_time,
                 weights_init_time, run_times, output_correct):
        self.op_name = op_name
        self.on_gpu = on_gpu
        self.graph_size = graph_size
        self.indices = indices
        self.latent_indicators = latent_indicators
        self.setup_time = setup_time
        self.weights_init_time = weights_init_time
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
            return ("%3s %11s %5s %5s %5s %11s %15s %15s %14s %10s" %
                    (dev, 'op', 'size', 'indices', 'latent_indicators', 'setup_time',
                     'weights_init_time', 'first_run_time', 'rest_run_time',
                     'correct'))

        def get_res(res):
            """Helper function printing a single result."""
            return ("%15s %5d %5s %7s %11.2f %15.2f %15.2f %14.2f %10s" %
                    (res.op_name, res.graph_size, res.indices, res.latent_indicators,
                     res.setup_time * 1000, res.weights_init_time * 1000,
                     res.run_times[0] * 1000,
                     np.mean(res.run_times[1:]) * 1000,
                     res.output_correct))

        # Print results
        print1("\n-----------------------", file)
        print1("%s" % self.test_name, file)
        print1("-----------------------", file)
        print1(get_header("CPU"), file)
        for res in sorted(self.cpu_results, key=lambda x: len(x.op_name)):
            print1(get_res(res), file, (red if res.indices is "No" else green if
            res.latent_indicators is "No" else magenta))
        print1(get_header("GPU"), file)
        for res in sorted(self.gpu_results, key=lambda x: len(x.op_name)):
            print1(get_res(res), file, (red if res.indices is "No" else green if
            res.latent_indicators is "No" else magenta))


class PerformanceTest:

    def __init__(self, num_input_rows, num_input_cols, num_sums, num_ops,
                 num_runs,  without_cpu, without_gpu, log_devs, profile,
                 profiles_dir, file):
        self.num_input_rows = num_input_rows
        self.num_input_cols = num_input_cols
        self.num_sums = num_sums
        self.num_ops = num_ops
        self.num_runs = num_runs
        self.without_cpu = without_cpu
        self.without_gpu = without_gpu
        self.log_devs = log_devs
        self.profile = profile
        self.profiles_dir = profiles_dir
        self.file = file
        self.test_failed = False

        print1("Params:", file)
        print1("- num_input_rows=%s" % num_input_rows, file)
        print1("- num_input_cols=%s" % num_input_cols, file)
        print1("- num_sums=%s" % num_sums, file)
        print1("- num_ops=%s" % num_ops, file)
        print1("- num_runs=%s" % num_runs, file)
        print1("", file=file)

    def _true_output(self, op_fun, inputs, indices, latent_indicators=None, inf_type=None):
        input_size = inputs.shape[1]

        if indices is not None:
            inputs = inputs[:, indices]

        if inf_type == spn.InferenceType.MARGINAL:
            np_op = np.sum
        elif inf_type == spn.InferenceType.MPE:
            np_op = np.amax
        else:
            sys.exit('ERROR: Incorrect inference type: ', inf_type)

        if op_fun is Ops.sum:
            input_size = inputs.shape[1]
            weight = 1.0 / input_size
            input_slice = inputs
            latent_indicators_slice = latent_indicators
        elif op_fun is Ops.par_sums or Ops.sums:
            weight = 1.0 / input_size
            input_slice = inputs
            if latent_indicators is not None:
                latent_indicators_slice = np.split(latent_indicators, self.num_sums, axis=1)[0]

        root_weight = 1.0 / self.num_sums
        inputs_array = np.stack([input_slice for _ in range(self.num_sums)], axis=0)

        # Compute true output with numpy
        if latent_indicators is None:
            return np_op(np.transpose(np_op((inputs_array * weight), axis=-1))
                         * root_weight, axis=-1, keepdims=True)
        else:
            latent_indicators_oh = np.eye(input_size)[np.squeeze(latent_indicators_slice)]
            return np_op(np.transpose(np_op((inputs_array * latent_indicators_oh * weight),
                         axis=-1)) * root_weight, axis=-1, keepdims=True)

    def _run_op_test(self, op_fun, inputs, indices=None, latent_indicators=None,
                     inf_type=spn.InferenceType.MARGINAL, log=False, on_gpu=True):
        """Run a single test for a single op."""
        # Preparations
        op_name = op_fun.__name__
        device_name = '/gpu:0' if on_gpu else '/cpu:0'

        # Print
        print2("--> %s: on_gpu=%s, inputs_shape=%s, indices=%s, latent_indicators=%s, inference=%s, log=%s"
               % (op_name, on_gpu, inputs.shape, ("No" if indices is None else "Yes"),
                  ("No" if latent_indicators is None else "Yes"), ("MPE" if inf_type ==
                  spn.InferenceType.MPE else "MARGINAL"), log), self.file)

        input_size = inputs.shape[1]

        # Compute true output
        true_out = self._true_output(op_fun, inputs, indices, latent_indicators, inf_type)

        # Create graph
        tf.reset_default_graph()
        with tf.device(device_name):
            # Create input
            inputs_pl = spn.RawLeaf(num_vars=input_size)
            # Create IndicatorLeaf
            if latent_indicators is None:
                latent_indicators_pl = [None for _ in range(self.num_sums)]
            else:
                if op_fun is Ops.sum:
                    latent_indicators_pl = [spn.IndicatorLeaf(num_vars=1, num_vals=input_size)
                              for _ in range(self.num_sums)]
                elif op_fun is Ops.par_sums or Ops.sums:
                    latent_indicators_pl = [spn.IndicatorLeaf(num_vars=self.num_sums, num_vals=input_size)]
            # Create ops
            start_time = time.time()
            init_ops, ops = op_fun(inputs_pl, indices, latent_indicators_pl, self.num_sums,
                                   inf_type, log)
            for _ in range(self.num_ops - 1):
                # The tuple ensures that the next op waits for the output
                # of the previous op, effectively stacking the ops
                # but using the original input every time
                init_ops, ops = op_fun(inputs_pl, indices, latent_indicators_pl, self.num_sums,
                                       inf_type, log, tf.tuple([ops])[0])
            setup_time = time.time() - start_time
        # Get num of graph ops
        graph_size = len(tf.get_default_graph().get_operations())
        # Run op multiple times
        output_correct = True
        with tf.Session(config=tf.ConfigProto(
                allow_soft_placement=False,
                log_device_placement=self.log_devs)) as sess:
            # Initialize weights of all the sum nodes in the graph
            start_time = time.time()
            init_ops.run()
            weights_init_time = time.time() - start_time

            run_times = []
            # Create feed dictionary
            feed = {inputs_pl: inputs}
            if latent_indicators is not None:
                for iv_pl in latent_indicators_pl:
                    feed[iv_pl] = latent_indicators
            for n in range(self.num_runs):
                # Run
                start_time = time.time()
                out = sess.run(ops, feed_dict=feed)
                run_times.append(time.time() - start_time)
                # Test value
                try:
                    np.testing.assert_array_almost_equal(out, (np.log(true_out)
                                                               if log else true_out))
                except AssertionError:
                    output_correct = False
                    self.test_failed = True

            if self.profile:
                # Add additional options to trace the session execution
                options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

                out = sess.run(ops, feed_dict=feed, options=options,
                               run_metadata=run_metadata)

                # Create the Timeline object, and write it to a json file
                fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                chrome_trace = fetched_timeline.generate_chrome_trace_format()
                if not os.path.exists(self.profiles_dir):
                    os.makedirs(self.profiles_dir)

                file_name = op_name
                file_name += ("_GPU" if on_gpu else "_CPU")
                file_name += ("_MPE-LOG" if log else "_MPE") if inf_type == \
                    spn.InferenceType.MPE else ("_MARGINAL-LOG" if log else
                                                "_MARGINAL")
                if indices is not None:
                    file_name += "_Indices"
                if latent_indicators is not None:
                    file_name += "_IVS"

                with open('%s/timeline_value_%s.json' % (self.profiles_dir,
                          file_name), 'w') as f:
                    f.write(chrome_trace)

        # Return stats
        return OpTestResult(op_name, on_gpu, graph_size, ("No" if indices is
                                                          None else "Yes"),
                            ("No" if latent_indicators is None else "Yes"), setup_time,
                            weights_init_time, run_times, output_correct)

    def _run_test(self, test_name, op_funs, inputs, indices, latent_indicators, inf_type, log):
        """Run a single test for multiple ops and devices."""
        cpu_results = []
        gpu_results = []
        for op_fun, inp, ind, iv in zip(op_funs, inputs, indices, latent_indicators):
            if not self.without_cpu:
                cpu_results.append(  # Indices = No, IndicatorLeaf = No
                    self._run_op_test(op_fun, inp, indices=None, latent_indicators=None,
                                      inf_type=inf_type, log=log, on_gpu=False))
                cpu_results.append(  # Indices = Yes, IndicatorLeaf = No
                    self._run_op_test(op_fun, inp, indices=ind, latent_indicators=None,
                                      inf_type=inf_type, log=log, on_gpu=False))
                cpu_results.append(  # Indices = Yes, IndicatorLeaf = Yes
                    self._run_op_test(op_fun, inp, indices=ind, latent_indicators=iv,
                                      inf_type=inf_type, log=log, on_gpu=False))
            if not self.without_gpu:
                gpu_results.append(  # Indices = No, IndicatorLeaf = No
                    self._run_op_test(op_fun, inp, indices=None, latent_indicators=None,
                                      inf_type=inf_type, log=log, on_gpu=True))
                gpu_results.append(  # Indices = Yes, IndicatorLeaf = No
                    self._run_op_test(op_fun, inp, indices=ind, latent_indicators=None,
                                      inf_type=inf_type, log=log, on_gpu=True))
                gpu_results.append(  # Indices = Yes, IndicatorLeaf = Yes
                    self._run_op_test(op_fun, inp, indices=ind, latent_indicators=iv,
                                      inf_type=inf_type, log=log, on_gpu=True))
        return TestResults(test_name, cpu_results, gpu_results)

    def run(self):
        """Run all tests."""
        print1("Running tests:", self.file)
        results = []

        # Sum
        sum_inputs = np.random.rand(self.num_input_rows, self.num_input_cols)
        sum_indices = list(range(self.num_input_cols-1, -1, -1))
        sum_latent_indicators = np.expand_dims(np.random.randint(self.num_input_cols,
                                                   size=self.num_input_rows),
                                 axis=1)

        # ParallelSums
        par_sums_inputs = np.random.rand(self.num_input_rows,
                                         self.num_input_cols)
        par_sums_indices = list(range(self.num_input_cols-1, -1, -1))
        par_sums_latent_indicators = np.tile(np.expand_dims(np.random.randint(self.num_input_cols,
                               size=self.num_input_rows), axis=1), (1, self.num_sums))

        # Sums
        sums_inputs = np.random.rand(self.num_input_rows, self.num_input_cols)
        sums_indices = list(range(self.num_input_cols-1, -1, -1))
        sums_latent_indicators = np.tile(np.expand_dims(np.random.randint(self.num_input_cols,
                           size=self.num_input_rows), axis=1), (1, self.num_sums))

        r = self._run_test('InferenceType: MARGINAL',
                           [Ops.sum, Ops.par_sums, Ops.sums],
                           [sum_inputs, par_sums_inputs, sums_inputs],
                           [sum_indices, par_sums_indices, sums_indices],
                           [sum_latent_indicators, par_sums_latent_indicators, sums_latent_indicators],
                           inf_type=spn.InferenceType.MARGINAL, log=False)
        results.append(r)

        r = self._run_test('InferenceType: MARGINAL-LOG',
                           [Ops.sum, Ops.par_sums, Ops.sums],
                           [sum_inputs, par_sums_inputs, sums_inputs],
                           [sum_indices, par_sums_indices, sums_indices],
                           [sum_latent_indicators, par_sums_latent_indicators, sums_latent_indicators],
                           inf_type=spn.InferenceType.MARGINAL, log=True)
        results.append(r)

        r = self._run_test('InferenceType: MPE',
                           [Ops.sum, Ops.par_sums, Ops.sums],
                           [sum_inputs, par_sums_inputs, sums_inputs],
                           [sum_indices, par_sums_indices, sums_indices],
                           [sum_latent_indicators, par_sums_latent_indicators, sums_latent_indicators],
                           inf_type=spn.InferenceType.MPE, log=False)
        results.append(r)

        r = self._run_test('InferenceType: MPE-LOG',
                           [Ops.sum, Ops.par_sums, Ops.sums],
                           [sum_inputs, par_sums_inputs, sums_inputs],
                           [sum_indices, par_sums_indices, sums_indices],
                           [sum_latent_indicators, par_sums_latent_indicators, sums_latent_indicators],
                           inf_type=spn.InferenceType.MPE, log=True)
        results.append(r)

        # Print results
        for res in results:
            res.print(self.file)

        if self.test_failed:
            print("\n ATLEAST ONE TEST FAILED!")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--num-input-rows', default=200, type=int,
                        help="Num of rows of inputs")
    parser.add_argument('--num-input-cols', default=101, type=int,
                        help="Num of cols of inputs")
    parser.add_argument('--num-sums', default=100, type=int,
                        help="Num of sums modelled in a single layer")
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
    parser.add_argument('--profile', default=False, action='store_true',
                        help="Run test one more time and profile")
    parser.add_argument('--profiles-dir', default='profiles', type=str,
                        help="Run test one more time and profile")
    parser.add_argument('--save-to', default='', type=str,
                        help="Save results to file")
    args = parser.parse_args()

    # Open a file
    f = None
    if args.save_to:
        f = open(args.save_to, 'w')

    try:
        t = PerformanceTest(args.num_input_rows, args.num_input_cols,
                            args.num_sums, args.num_ops, args.num_runs,
                            args.without_cpu, args.without_gpu, args.log_devices,
                            args.profile, args.profiles_dir, f)
        t.run()
    finally:
        if f is not None:
            f.close()


if __name__ == '__main__':
    main()
