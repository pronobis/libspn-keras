#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
from context import libspn as spn
import time
import argparse
import colorama as col
from tensorflow.python.client import timeline
import os
from itertools import product
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

    def poon_single(inputs, num_vals, num_mixtures, num_subsets, inf_type,
                    log=False, output=None):

        # Build a POON-like network with single-op nodes
        subsets = [[spn.Sum((inputs, list(range(i*num_vals, (i+1)*num_vals))))
                   for _ in range(num_mixtures)] for i in range(num_subsets)]
        products = [spn.Product(*list(inp)) for inp in list(product(*[s for s in
                                                                      subsets]))]
        root = spn.Sum(*products, name="root")

        # Generate dense SPN and all weights in the network
        spn.generate_weights(root)

        # Generate path ops based on inf_type and log
        if log:
            mpe_path_gen = spn.MPEPath(value_inference_type=inf_type, log=True)
        else:
            mpe_path_gen = spn.MPEPath(value_inference_type=inf_type, log=False)

        mpe_path_gen.get_mpe_path(root)
        path_ops = [mpe_path_gen.counts[inp] for inp in (inputs if isinstance(inputs, list)
                                                         else [inputs])]
        return root, spn.initialize_weights(root), path_ops

    def poons_multi(inputs, num_vals, num_mixtures, num_subsets, inf_type,
                    log=False, output=None):

        # Build a POON-like network with multi-op nodes
        subsets = [spn.ParallelSums((inputs, list(range(i*num_vals, (i+1)*num_vals))),
                                    num_sums=num_mixtures) for i in range(num_subsets)]
        products = spn.PermuteProducts(*subsets)
        root = spn.Sum(products, name="root")

        # Generate dense SPN and all weights in the network
        spn.generate_weights(root)

        # Generate path ops based on inf_type and log
        if log:
            mpe_path_gen = spn.MPEPath(value_inference_type=inf_type, log=True)
        else:
            mpe_path_gen = spn.MPEPath(value_inference_type=inf_type, log=False)

        mpe_path_gen.get_mpe_path(root)
        path_ops = [mpe_path_gen.counts[inp] for inp in (inputs if isinstance(inputs, list)
                                                         else [inputs])]
        return root, spn.initialize_weights(root), path_ops


class OpTestResult:
    """Result of a single test of a single op."""

    def __init__(self, op_name, on_gpu, spn_size, tf_size, setup_time,
                 weights_init_time, run_times, output_correct):
        self.op_name = op_name
        self.on_gpu = on_gpu
        self.spn_size = spn_size
        self.tf_size = tf_size
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
            return ("%4s %11s %9s %8s %11s %17s %15s %14s %10s" %
                    (dev, 'op', 'SPN_size', 'TF_size', 'setup_time',
                     'weights_init_time', 'first_run_time', 'rest_run_time',
                     'correct'))

        def get_res(res):
            """Helper function printing a single result."""
            return ("%16s %7d %7d %11.2f %15.2f %15.2f %14.2f %12s" %
                    (res.op_name, res.spn_size, res.tf_size, res.setup_time * 1000,
                     res.weights_init_time * 1000, res.run_times[0] * 1000,
                     np.mean(res.run_times[1:]) * 1000,
                     res.output_correct))

        # Print results
        print1("\n-----------------------", file)
        print1("%s" % self.test_name, file)
        print1("-----------------------", file)
        print1(get_header("CPU"), file)
        for res in sorted(self.cpu_results, key=lambda x: len(x.op_name)):
            print1(get_res(res), file, (red if res.op_name is "poon_single" else
                                        green))
        print1(get_header("GPU"), file)
        for res in sorted(self.gpu_results, key=lambda x: len(x.op_name)):
            print1(get_res(res), file, (red if res.op_name is "poon_single" else
                                        green))


class PerformanceTest:

    def __init__(self, num_input_rows, num_input_vars, num_input_vals, num_mixtures,
                 num_networks, num_runs, without_cpu, without_gpu, log_devs, profile,
                 profiles_dir, file):
        self.num_input_rows = num_input_rows
        self.num_input_vars = num_input_vars
        self.num_input_vals = num_input_vals
        self.num_mixtures = num_mixtures
        self.num_subsets = num_input_vars
        self.num_networks = num_networks
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
        print1("- num_input_vars=%s" % num_input_vars, file)
        print1("- num_input_vals=%s" % num_input_vals, file)
        print1("- num_mixtures=%s" % num_mixtures, file)
        print1("- num_subsets=%s" % num_input_vars, file)
        print1("- num_networks=%s" % num_networks, file)
        print1("- num_runs=%s" % num_runs, file)
        print1("", file=file)

    def _true_output(self):
        true_out = np.zeros((1, self.num_input_rows,
                             self.num_input_vars*self.num_input_vals))
        true_out[:, :, list(range(0, self.num_input_vars*self.num_input_vals,
                                  self.num_input_vals))] = 1
        return true_out

    def _run_network_test(self, network_fun, inputs, inf_type=spn.InferenceType.MARGINAL,
                          log=False, on_gpu=True):
        """Run a single test for a single op."""
        # Preparations
        op_name = network_fun.__name__
        device_name = '/gpu:0' if on_gpu else '/cpu:0'

        # Print
        print2("--> %s: on_gpu=%s, inputs_shape=%s, inference=%s, log=%s"
               % (op_name, on_gpu, inputs.shape, ("MPE" if inf_type ==
                  spn.InferenceType.MPE else "MARGINAL"), log), self.file)

        # Compute true output
        true_out = self._true_output()

        # Create graph
        tf.reset_default_graph()
        with tf.device(device_name):
            # Create input
            inputs_pl = spn.IndicatorLeaf(num_vars=self.num_input_vars,
                                num_vals=self.num_input_vals, name="iv_x")
            # Create networks, stacking one on top of the other, although each
            # network remains unconnected and independent of each other.
            start_time = time.time()
            root, init_network, network = \
                network_fun(inputs_pl, self.num_input_vals, self.num_mixtures,
                            self.num_subsets, inf_type, log)
            for _ in range(self.num_networks - 1):
                # The tuple ensures that the next network waits for the output
                # of the previous network, effectively stacking the networks
                # but using the original input every time
                root, init_network, network = \
                    network_fun(inputs_pl, self.num_input_vals, self.num_mixtures,
                                self.num_subsets, inf_type, log, tf.tuple([network])[0])
            setup_time = time.time() - start_time
        # Get num of SPN ops
        spn_size = root.get_num_nodes() * self.num_networks
        # Get num of graph ops
        tf_size = len(tf.get_default_graph().get_operations())
        # Run op multiple times
        output_correct = True
        with tf.Session(config=tf.ConfigProto(
                allow_soft_placement=False,
                log_device_placement=self.log_devs)) as sess:
            # Initialize weights of all the sum node types in the graph
            start_time = time.time()
            init_network.run()
            weights_init_time = time.time() - start_time

            run_times = []
            # Create feed dictionary
            feed = {inputs_pl: inputs}
            for n in range(self.num_runs):
                # Run
                start_time = time.time()
                out = sess.run(network, feed_dict=feed)
                run_times.append(time.time() - start_time)
                # Test value
                try:
                    np.testing.assert_array_almost_equal(out, true_out)
                except AssertionError:
                    output_correct = False
                    self.test_failed = True

            if self.profile:
                # Add additional options to trace the session execution
                options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

                out = sess.run(network, feed_dict=feed, options=options,
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

                with open('%s/timeline_path_%s.json' % (self.profiles_dir,
                          file_name), 'w') as f:
                    f.write(chrome_trace)

        # Return stats
        return OpTestResult(op_name, on_gpu, spn_size, tf_size, setup_time,
                            weights_init_time, run_times, output_correct)

    def _run_test(self, test_name, network_funs, inputs, inf_type, log):
        """Run a single test for multiple ops and devices."""
        cpu_results = []
        gpu_results = []
        for network_fun in network_funs:
            if not self.without_cpu:
                cpu_results.append(
                    self._run_network_test(network_fun, inputs, inf_type=inf_type,
                                           log=log, on_gpu=False))
            if not self.without_gpu:
                gpu_results.append(
                    self._run_network_test(network_fun, inputs, inf_type=inf_type,
                                           log=log, on_gpu=True))
        return TestResults(test_name, cpu_results, gpu_results)

    def run(self):
        """Run all tests."""
        print1("Running tests:", self.file)
        results = []

        inputs = np.zeros((self.num_input_rows, self.num_input_vars))

        r = self._run_test('InferenceType: MARGINAL',
                           [Ops.poon_single, Ops.poons_multi], inputs,
                           inf_type=spn.InferenceType.MARGINAL, log=False)
        results.append(r)

        r = self._run_test('InferenceType: MARGINAL-LOG',
                           [Ops.poon_single, Ops.poons_multi], inputs,
                           inf_type=spn.InferenceType.MARGINAL, log=True)
        results.append(r)

        r = self._run_test('InferenceType: MPE',
                           [Ops.poon_single, Ops.poons_multi], inputs,
                           inf_type=spn.InferenceType.MPE, log=False)
        results.append(r)

        r = self._run_test('InferenceType: MPE-LOG',
                           [Ops.poon_single, Ops.poons_multi], inputs,
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
    parser.add_argument('--num-input-vars', default=5, type=int,
                        help="Num of input variables")
    parser.add_argument('--num-input-vals', default=5, type=int,
                        help="Num of input values per variable")
    parser.add_argument('--num-mixtures', default=5, type=int,
                        help="Num of mixtures per subset")
    parser.add_argument('--num-networks', default=1, type=int,
                        help="Num of networks used for tests")
    parser.add_argument('--num-runs', default=50, type=int,
                        help="Num of times each test is run")
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
        t = PerformanceTest(args.num_input_rows, args.num_input_vars, args.num_input_vals,
                            args.num_mixtures, args.num_networks, args.num_runs,
                            args.without_cpu, args.without_gpu, args.log_devices,
                            args.profile, args.profiles_dir, f)
        t.run()
    finally:
        if f is not None:
            f.close()


if __name__ == '__main__':
    main()
