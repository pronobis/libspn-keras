#!/usr/bin/env python3

import argparse
import itertools
import random
import time

import colorama as col
import numpy as np
import tensorflow as tf
from context import libspn as spn

from libspn.tests.profiler import profile_report

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


def sums_layer_value_numpy(inputs, sums_sizes, weights, latent_indicators=None,
                           inference_type=spn.InferenceType.MARGINAL):
    """ Computes value of SumsLayer using numpy """
    inputs_to_reduce, iv_mask_per_sum, weights_per_sum = sums_layer_numpy_common(
        inputs, latent_indicators, sums_sizes, weights)

    # Finally, we reduce each sum node and concatenate the results
    reduce_fn = np.sum if inference_type == spn.InferenceType.MARGINAL else np.max
    return reduce_fn(np.concatenate(
        [reduce_fn(x * np.reshape(w/np.sum(w), (1, -1)) * iv, axis=1, keepdims=True)
         for x, w, iv in zip(inputs_to_reduce, weights_per_sum, iv_mask_per_sum)], axis=1
    ), axis=1, keepdims=True) * 1 / len(sums_sizes)


def sums_layer_numpy_common(inputs, latent_indicators, sums_sizes, weights):
    inputs_selected = []
    # Taking care of indices
    for x, indices in inputs:
        if indices:
            inputs_selected.append(x[:, indices])
        else:
            inputs_selected.append(x)
    # Concatenate and then split based on sums_sizes
    splits = np.cumsum(sums_sizes)
    inputs_concatenated = np.concatenate(inputs_selected, axis=1)
    inputs_to_reduce = np.split(inputs_concatenated, splits, axis=1)[:-1]
    weights_per_sum = np.split(weights, splits)[:-1]
    iv_mask = build_iv_mask(inputs_selected, inputs_to_reduce, latent_indicators, sums_sizes)
    iv_mask_per_sum = np.split(iv_mask, splits, axis=1)[:-1]

    return inputs_to_reduce, iv_mask_per_sum, weights_per_sum


def build_iv_mask(inputs_selected, inputs_to_reduce, latent_indicators, sums_sizes):
    """
    Creates concatenated IV matrix with boolean values that can be multiplied with the
    reducible values for masking
    """
    if latent_indicators is not None:
        latent_indicators_ = np.ones((inputs_selected[0].shape[0], sum(sums_sizes)))
        for row in range(latent_indicators_.shape[0]):
            offset = 0
            for iv, s in zip(latent_indicators, sums_sizes):
                if 0 <= iv[row] < s:
                    latent_indicators_[row, offset:offset + s] = 0
                    latent_indicators_[row, offset + iv[row]] = 1
                offset += s
    else:
        latent_indicators_ = np.concatenate([np.ones_like(x) for x in inputs_to_reduce], 1)
    return latent_indicators_


class Ops:
    @staticmethod
    def sum(inputs, sum_indices, repetitions, inf_type, log=False, latent_indicators=None):
        """ Creates the graph using only Sum nodes """
        sum_nodes = []
        for ind in sum_indices:
            sum_nodes.extend([spn.Sum((inputs, ind)) for _ in range(repetitions)])
        [s.generate_weights() for s in sum_nodes]
        if latent_indicators:
            [s.set_latent_indicators(iv) for s, iv in zip(sum_nodes, latent_indicators)]

        root, value_op = Ops._build_root_and_value(inf_type, log, sum_nodes)

        return spn.initialize_weights(root), value_op

    @staticmethod
    def _build_root_and_value(inf_type, log, sum_nodes):
        """ Connects the sum node outputs to a single root as a way of grouping Ops """
        root = spn.Sum(*sum_nodes)
        root.generate_weights()
        if log:
            value_op = root.get_log_value(inference_type=inf_type)
        else:
            value_op = root.get_value(inference_type=inf_type)
        return root, value_op

    @staticmethod
    def par_sums(inputs, sum_indices, repetitions, inf_type, log=False, latent_indicators=None):
        """ Creates the graph using only ParSum nodes """
        parallel_sum_nodes = []
        for ind in sum_indices:
            parallel_sum_nodes.append(spn.ParallelSums((inputs, ind), num_sums=repetitions))
        [s.generate_weights() for s in parallel_sum_nodes]
        if latent_indicators:
            [s.set_latent_indicators(iv) for s, iv in zip(parallel_sum_nodes, latent_indicators)]

        root, value_op = Ops._build_root_and_value(inf_type, log, parallel_sum_nodes)

        return spn.initialize_weights(root), value_op

    @staticmethod
    def sums_layer(inputs, sum_indices, repetitions, inf_type, log=False, latent_indicators=None):
        """ Creates the graph using a SumsLayer node """
        repeated_inputs = []
        repeated_sum_sizes = []
        for ind in sum_indices:
            repeated_inputs.extend([(inputs, ind) for _ in range(repetitions)])
            repeated_sum_sizes.extend([len(ind) for _ in range(repetitions)])

        sums_layer = spn.SumsLayer(*repeated_inputs, n_sums_or_sizes=repeated_sum_sizes)
        sums_layer.generate_weights()
        if latent_indicators:
            sums_layer.set_latent_indicators(*latent_indicators)

        root, value_op = Ops._build_root_and_value(inf_type, log, [sums_layer])
        return spn.initialize_weights(root), value_op


class OpTestResult:
    """Result of a single test of a single op."""

    def __init__(self, op_name, on_gpu, graph_size, indices, latent_indicators, setup_time, weights_init_time,
                 run_times, output_correct):
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
            return ("%15s %5d %7s %5s %11.2f %17.2f %15.2f %14.2f %10s" %
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

    def __init__(self, batch_size, num_cols, sum_sizes, num_runs, without_cpu, without_gpu,
                 log_devs, profile, profiles_dir, file, num_parallel):
        self.batch_size = batch_size
        self.num_cols = num_cols
        self.sum_sizes = sum_sizes
        self.num_runs = num_runs
        self.num_parallel = num_parallel
        self.without_cpu = without_cpu
        self.without_gpu = without_gpu
        self.log_devs = log_devs
        self.profile = profile
        self.profiles_dir = profiles_dir
        self.file = file
        self.test_failed = False

        print1("Params:", file)
        print1("- batch_size=%s" % batch_size, file)
        print1("- sum_sizes=%s" % sum_sizes, file)
        print1("- num_runs=%s" % num_runs, file)
        print1("- num_parallel=%s" % num_parallel, file)
        print1("", file=file)

    def _run_op_test(self, op_fun, inputs, sum_indices=None, inf_type=spn.InferenceType.MARGINAL,
                     log=False, on_gpu=True, latent_indicators=None):
        """Run a single test for a single op."""

        # Preparations
        op_name = op_fun.__name__
        device_name = '/gpu:0' if on_gpu else '/cpu:0'

        # Print
        print2("--> %s: on_gpu=%s, inputs_shape=%s, indices=%s, inference=%s, log=%s, IndicatorLeaf=%s"
               % (op_name, on_gpu, inputs.shape, ("No" if sum_indices is None else "Yes"),
                  ("MPE" if inf_type == spn.InferenceType.MPE else "MARGINAL"), log,
                  ("No" if latent_indicators is None else "Yes")), self.file)

        input_size = inputs.shape[1]

        # Create graph
        tf.reset_default_graph()

        # Compute the true output
        sum_sizes = [len(ind) for ind in sum_indices]
        latent_indicators_per_sum = np.split(latent_indicators, latent_indicators.shape[1], axis=1) if latent_indicators is not None \
            else None
        sum_sizes_np = self._repeat_elements(sum_sizes)
        true_out = self._true_out(inf_type, inputs, latent_indicators_per_sum, sum_indices, sum_sizes,
                                  sum_sizes_np)
        if log:
            true_out = np.log(true_out)

        # Set up the graph
        with tf.device(device_name):
            # Create input
            inputs_pl = spn.RawLeaf(num_vars=input_size)
            feed_dict = {inputs_pl: inputs}

            if latent_indicators is not None:
                if op_fun is Ops.sum:
                    latent_indicators_pl = [spn.IndicatorLeaf(num_vars=1, num_vals=s) for s in sum_sizes_np]
                    latent_indicators = latent_indicators_per_sum
                elif op_fun is Ops.par_sums:
                    latent_indicators_pl = [spn.IndicatorLeaf(num_vars=self.num_parallel, num_vals=len(ind))
                              for ind in sum_indices]
                    latent_indicators = np.split(latent_indicators, len(self.sum_sizes), axis=1)
                else:
                    latent_indicators = [latent_indicators]
                    latent_indicators_pl = [spn.IndicatorLeaf(num_vars=len(sum_sizes_np), num_vals=max(sum_sizes))]
                for iv_pl, iv in zip(latent_indicators_pl, latent_indicators):
                    feed_dict[iv_pl] = iv
            else:
                latent_indicators_pl = None

            # Create ops
            start_time = time.time()
            init_ops, ops = op_fun(
                inputs_pl, sum_indices, self.num_parallel, inf_type, log, latent_indicators=latent_indicators_pl)
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
            for n in range(self.num_runs):
                # Run
                start_time = time.time()
                out = sess.run(ops, feed_dict=feed_dict)
                run_times.append(time.time() - start_time)
                # Test value
                try:
                    np.testing.assert_array_almost_equal(out, true_out)
                except AssertionError:
                    output_correct = False
                    self.test_failed = True

            if self.profile:
                # Create a suitable filename suffix
                fnm_suffix = op_name
                fnm_suffix += ("_GPU" if on_gpu else "_CPU")
                fnm_suffix += ("_MPE-LOG" if log else "_MPE") if inf_type == \
                    spn.InferenceType.MPE else ("_MARGINAL-LOG" if log else "_MARGINAL")
                fnm_suffix += ("_IV" if latent_indicators is not None else "")
                # Create a profiling report
                profile_report(sess, ops, feed_dict, self.profiles_dir,
                               "sum_value_varying_sizes", fnm_suffix)

        # Return stats
        return OpTestResult(op_name, on_gpu, graph_size, ("Yes"), ("No" if latent_indicators is None else "Yes"),
                            setup_time, weights_init_time, run_times, output_correct)

    def _true_out(self, inf_type, inputs, latent_indicators_per_sum, sum_indices, sum_sizes, sum_sizes_np):
        """ Computes true output """
        numpy_inputs = self._repeat_elements([(inputs, ind) for ind in sum_indices])
        w = np.ones(sum(sum_sizes) * self.num_parallel)
        true_out = sums_layer_value_numpy(numpy_inputs, sum_sizes_np, w,
                                          latent_indicators=latent_indicators_per_sum, inference_type=inf_type)
        return true_out

    def _repeat_elements(self, arr):
        """
        Repeats the elements int the input array, e.g.
        [1, 2, 3] -> [1, 1, 1, 2, 2, 2, 3, 3, 3]
        """
        ret = list(itertools.chain(*[list(itertools.repeat(np_in, self.num_parallel))
                                     for np_in in arr]))
        return ret

    def _run_test(self, test_name, op_funs, inputs, indices, latent_indicators, inf_type, log):
        """Run a single test for multiple ops and devices."""
        cpu_results = []
        gpu_results = []
        for op_fun, inp, ind, iv in zip(op_funs, inputs, indices, latent_indicators):
            # Decide on which devices
            use_gpu = ([True] if not self.without_gpu else []) + \
                      ([False] if not self.without_cpu else [])

            # Go through all combinations of devices and IndicatorLeaf
            for on_gpu in use_gpu:
                (gpu_results if on_gpu else cpu_results).append(
                    self._run_op_test(op_fun, inp, sum_indices=ind, inf_type=inf_type, log=log,
                                      latent_indicators=iv, on_gpu=on_gpu))
        return TestResults(test_name, cpu_results, gpu_results)

    def run(self):
        """Run all tests."""
        print1("Running tests:", self.file)
        results = []

        sum_sizes_repeated = self._repeat_elements(self.sum_sizes)
        latent_indicators = np.stack(
            [np.random.choice(s + 1, self.batch_size) - 1 for s in sum_sizes_repeated], axis=1)

        # Sum
        sum_inputs = np.random.rand(self.batch_size, self.num_cols)
        sum_indices = [random.sample(range(self.num_cols), size)
                       for size in self.sum_sizes]

        for inf_type, log, iv in itertools.product(
            [spn.InferenceType.MARGINAL, spn.InferenceType.MPE], [False, True], [None, latent_indicators]
        ):
            name = 'InferenceType: {}{}'.format(
                "MARGINAL" if inf_type == spn.InferenceType.MARGINAL else "MPE",
                "-LOG" if log else "")
            r = self._run_test(name, [Ops.sums_layer, Ops.par_sums, Ops.sum],
                               3 * [sum_inputs], 3 * [sum_indices], 3 * [iv],
                               inf_type=inf_type, log=log)
            results.append(r)

        # Print results
        for res in results:
            res.print(self.file)

        if self.test_failed:
            print("\n ATLEAST ONE TEST FAILED!")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--num-input-rows', default=500, type=int,
                        help="Num of rows of inputs")
    parser.add_argument('--num-input-cols', default=100, type=int,
                        help="Num of cols of inputs")
    parser.add_argument("--sum-sizes", default=[70, 80, 90, 100], type=int, nargs='+',
                        help="The size of each sum being modeled. Will be repeated "
                             "num-repetitions times, e.g. "
                             "[1, 2, 3] --> [1, 1, 1, 2, 2, 2, 3, 3, 3]")
    parser.add_argument('--num-sums', default=100, type=int,
                        help="Num of sums modelled in a single layer")
    parser.add_argument('--num-repetitions', default=10, type=int,
                        help="Num repetitions for each input and sum size")
    parser.add_argument('--num-ops', default=10, type=int,
                        help="Num of ops used for tests")
    parser.add_argument('--num-runs', default=100, type=int,
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
        # TODO quite some renaming!
        t = PerformanceTest(batch_size=args.num_input_rows, sum_sizes=args.sum_sizes,
                            num_runs=args.num_runs, without_cpu=args.without_cpu,
                            without_gpu=args.without_gpu, log_devs=args.log_devices,
                            profiles_dir=args.profiles_dir, file=f, profile=args.profile,
                            num_parallel=args.num_repetitions, num_cols=args.num_input_cols)
        t.run()
    finally:
        if f is not None:
            f.close()


if __name__ == '__main__':
    main()
