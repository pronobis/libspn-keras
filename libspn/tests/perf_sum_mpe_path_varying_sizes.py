#!/usr/bin/env python3

# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

import argparse
import itertools
import random
import time

import colorama as col
import numpy as np
import tensorflow as tf
from context import libspn as spn
import functools
import operator

from libspn.tests.profiler import profile_report
from libspn.tests.perf_sum_value_varying_sizes import sums_layer_numpy_common, OpTestResult, \
    PerformanceTest, TestResults

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


def sums_layer_mpe_path_numpy(inputs, sums_sizes, weights, counts, ivs=None):
    """ Computes the output of _compute_mpe_path with numpy """
    inputs_to_reduce, iv_mask_per_sum, weights_per_sum = sums_layer_numpy_common(
        inputs, ivs, sums_sizes, weights)
    # Get max index for sum node
    max_indices = [np.argmax(x * np.reshape(w/np.sum(w), (1, -1)) * iv, axis=1)
                   for x, w, iv in zip(inputs_to_reduce, weights_per_sum, iv_mask_per_sum)]

    counts_per_sum = []
    for i, (mi, size) in enumerate(zip(max_indices, sums_sizes)):
        # Initialize the counts
        cnt = np.zeros((mi.shape[0], size))
        cnt[range(cnt.shape[0]), mi] = counts[:, i]
        counts_per_sum.append(cnt)

    # Go from counts per sum to counts per input
    counts_concatenated = np.concatenate(counts_per_sum, axis=1)
    # print([len(c) for c in counts_per_sum])
    indices = np.cumsum([len(t[1]) if t[1] else t[0].shape[1] for t in inputs])[:-1]
    counts_per_input = np.split(counts_concatenated, indices, axis=1)

    # 'Scatter' the values to the indices given by each second tuple element in inputs if not None
    # otherwise, the indices are just [0, ..., size-1]
    scattered = []
    for c, inp in zip(counts_per_input, inputs):
        # print(inp[0].shape, c.shape)
        s = np.zeros_like(inp[0])
        if inp[1]:
            s[:, inp[1]] = c
        else:
            s = c
        scattered.append(s)
    return scattered


class Ops:
    @staticmethod
    def sum(inputs, sum_indices, repetitions, inf_type, counts, log=False, ivs=None):
        """ Creates the graph using only Sum nodes """
        sum_nodes = []
        for ind in sum_indices:
            sum_nodes.extend([spn.Sum((inputs, ind)) for _ in range(repetitions)])
        weights = [s.generate_weights() for s in sum_nodes]
        if ivs:
            [s.set_ivs(iv) for s, iv in zip(sum_nodes, ivs)]

        root, value_op = Ops._build_root_and_value(inf_type, log, sum_nodes)

        if log:
            w_vals = [node.get_log_value() for node in weights]
            if ivs is None:
                iv_vals = [None for _ in sum_nodes]
            else:
                iv_vals = [node.get_log_value() for node in ivs]
            value_input_vals = [inputs.get_log_value()
                                for _ in range(repetitions * len(sum_indices))]
        else:
            w_vals = [node.get_value() for node in weights]
            if ivs is None:
                iv_vals = [None for _ in sum_nodes]
            else:
                iv_vals = [node.get_value() for node in ivs]
            value_input_vals = [inputs.get_value() for _ in range(repetitions * len(sum_indices))]

        ops = [node._compute_log_mpe_path(c, w_val, iv_val, val_inp_val)[-1:]
               for node, c, w_val, iv_val, val_inp_val
               in zip(sum_nodes, counts, w_vals, iv_vals, value_input_vals)]

        return spn.initialize_weights(root), tf.add_n(list(itertools.chain(*ops)))  #list(itertools.chain(*ops))

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
    def par_sums(inputs, sum_indices, repetitions, inf_type, counts, log=False, ivs=None):
        """ Creates the graph using only ParSum nodes """
        parallel_sum_nodes = []
        for ind in sum_indices:
            parallel_sum_nodes.append(spn.ParSums((inputs, ind), num_sums=repetitions))
        weights = [s.generate_weights() for s in parallel_sum_nodes]
        if ivs:
            [s.set_ivs(iv) for s, iv in zip(parallel_sum_nodes, ivs)]

        root, value_op = Ops._build_root_and_value(inf_type, log, parallel_sum_nodes)

        if log:
            w_vals = [node.get_log_value() for node in weights]
            if ivs is None:
                iv_vals = [None for _ in parallel_sum_nodes]
            else:
                iv_vals = [node.get_log_value() for node in ivs]
            value_input_vals = [inputs.get_log_value()
                                for _ in range(repetitions * len(sum_indices))]
        else:
            w_vals = [node.get_value() for node in weights]
            if ivs is None:
                iv_vals = [None for _ in parallel_sum_nodes]
            else:
                iv_vals = [node.get_value() for node in ivs]
            value_input_vals = [inputs.get_value() for _ in range(repetitions * len(sum_indices))]

        ops = [node._compute_log_mpe_path(tf.identity(c), w_val, iv_val, val_inp_val)[-1:]
               for node, c, w_val, iv_val, val_inp_val
               in zip(parallel_sum_nodes, counts, w_vals, iv_vals, value_input_vals)]

        return spn.initialize_weights(root), tf.add_n(list(itertools.chain(*ops)))  #list(itertools.chain(*ops))

    @staticmethod
    def sums_layer(inputs, sum_indices, repetitions, inf_type, counts, log=False, ivs=None,
                   sum_counts=False):
        """ Creates the graph using a SumsLayer node """
        repeated_inputs = []
        repeated_sum_sizes = []
        for ind in sum_indices:
            repeated_inputs.extend([(inputs, ind) for _ in range(repetitions)])
            repeated_sum_sizes.extend([len(ind) for _ in range(repetitions)])

        sums_layer = spn.SumsLayer(*repeated_inputs, n_sums_or_sizes=repeated_sum_sizes)
        weights = sums_layer.generate_weights()
        if ivs:
            sums_layer.set_ivs(*ivs)

        root, value_op = Ops._build_root_and_value(inf_type, log, [sums_layer])

        if log:
            if ivs is not None:
                iv_vals = [node.get_log_value() for node in ivs]
            else:
                iv_vals = [None]
            inp_val = inputs.get_log_value()
            value_input_vals = [inp_val for _ in range(repetitions * len(sum_indices))]

            ops = sums_layer._compute_log_mpe_path(counts[0],
                                                   weights.get_log_value(),
                                                   iv_vals[0],
                                                   *value_input_vals,
                                                   sum_counts=False)
        else:
            if ivs is not None:
                iv_vals = [node.get_value() for node in ivs]
            else:
                iv_vals = [None]
            inp_val = inputs.get_value()
            value_input_vals = [inp_val for _ in range(repetitions * len(sum_indices))]

            ops = sums_layer._compute_mpe_path(counts[0],
                                               weights.get_value(),
                                               iv_vals[0],
                                               *value_input_vals,
                                               sum_counts=False)

        return spn.initialize_weights(root), tf.add_n(ops[-repetitions * len(sum_indices):])

    @staticmethod
    def sums_layer_v2(inputs, sum_indices, repetitions, inf_type, counts, log=False, ivs=None,
                   sum_counts=False):
        """ Creates the graph using a SumsLayer node """
        repeated_inputs = []
        repeated_sum_sizes = []
        for ind in sum_indices:
            repeated_inputs.extend([(inputs, ind) for _ in range(repetitions)])
            repeated_sum_sizes.extend([len(ind) for _ in range(repetitions)])

        sums_layer = spn.SumsLayer(*repeated_inputs, n_sums_or_sizes=repeated_sum_sizes)
        weights = sums_layer.generate_weights()
        if ivs:
            sums_layer.set_ivs(*ivs)

        root, value_op = Ops._build_root_and_value(inf_type, log, [sums_layer])

        if log:
            if ivs is not None:
                iv_vals = [node.get_log_value() for node in ivs]
            else:
                iv_vals = [None]
            inp_val = inputs.get_log_value()
            value_input_vals = [inp_val for _ in range(repetitions * len(sum_indices))]

            ops = sums_layer._compute_log_mpe_path(counts[0],
                                                   weights.get_log_value(),
                                                   iv_vals[0],
                                                   *value_input_vals,
                                                   sum_counts=True)
        else:
            if ivs is not None:
                iv_vals = [node.get_value() for node in ivs]
            else:
                iv_vals = [None]
            inp_val = inputs.get_value()
            value_input_vals = [inp_val for _ in range(repetitions * len(sum_indices))]

            ops = sums_layer._compute_mpe_path(counts[0],
                                               weights.get_value(),
                                               iv_vals[0],
                                               *value_input_vals,
                                               sum_counts=True)

        return spn.initialize_weights(root), ops[-1]



class PerformanceTestMPEPath(PerformanceTest):

    def _run_op_test(self, op_fun, inputs, sum_indices=None, inf_type=spn.InferenceType.MARGINAL,
                     log=False, on_gpu=True, ivs=None):
        """Run a single test for a single op."""

        # Preparations
        op_name = op_fun.__name__
        device_name = '/gpu:0' if on_gpu else '/cpu:0'

        # Print
        print2("--> %s: on_gpu=%s, inputs_shape=%s, indices=%s, inference=%s, log=%s, IVs=%s"
               % (op_name, on_gpu, inputs.shape, ("No" if sum_indices is None else "Yes"),
                  ("MPE" if inf_type == spn.InferenceType.MPE else "MARGINAL"), log,
                  ("No" if ivs is None else "Yes")), self.file)

        input_size = inputs.shape[1]

        # Create graph
        tf.reset_default_graph()

        # Compute the true output
        sum_sizes = [len(ind) for ind in sum_indices]
        ivs_per_sum = np.split(ivs, ivs.shape[1], axis=1) if ivs is not None \
            else None
        sum_sizes_np = self._repeat_elements(sum_sizes)
        true_out = self._true_out(inf_type, inputs, ivs_per_sum, sum_indices, sum_sizes,
                                  sum_sizes_np)

        true_out = functools.reduce(operator.add, true_out)

        # Set up the graph
        with tf.device(device_name):
            # Create input
            inputs_pl = spn.ContVars(num_vars=input_size)
            feed_dict = {inputs_pl: inputs}

            if ivs is not None:
                if op_fun is Ops.sum:
                    ivs_pl = [spn.IVs(num_vars=1, num_vals=s) for s in sum_sizes_np]
                    ivs = ivs_per_sum
                elif op_fun is Ops.par_sums:
                    ivs_pl = [spn.IVs(num_vars=self.num_parallel, num_vals=len(ind))
                              for ind in sum_indices]
                    ivs = np.split(ivs, len(self.sum_sizes), axis=1)
                else:
                    ivs = [ivs]
                    ivs_pl = [spn.IVs(num_vars=len(sum_sizes_np), num_vals=max(sum_sizes))]
                for iv_pl, iv in zip(ivs_pl, ivs):
                    feed_dict[iv_pl] = iv
            else:
                ivs_pl = None

            counts = []
            if op_fun is Ops.sum:
                for i in range(len(sum_sizes_np)):
                    size = self.batch_size
                    init_val = np.reshape(np.arange(i * size, (i+1) * size), (-1, 1))
                    counts.append(tf.placeholder(tf.float32, [None, 1]))
                    feed_dict[counts[-1]] = init_val + 10
            elif op_fun is Ops.par_sums:
                for i in range(len(sum_indices)):
                    size = self.batch_size * self.num_parallel
                    init_val = np.transpose(np.reshape(np.arange(i * size, (i+1) * size),
                                                       (self.num_parallel, self.batch_size)))
                    counts.append(tf.placeholder(tf.float32, [None, self.num_parallel]))
                    feed_dict[counts[-1]] = init_val + 10
            else:
                init_val = np.transpose(np.reshape(np.arange(
                    self.batch_size * len(sum_indices) * self.num_parallel),
                    (len(sum_indices) * self.num_parallel, self.batch_size)
                ))
                counts.append(tf.placeholder(tf.float32,
                                             [None, len(sum_indices) * self.num_parallel]))
                feed_dict[counts[-1]] = init_val + 10
            # Create ops
            start_time = time.time()
            init_ops, ops = op_fun(
                inputs_pl, sum_indices, self.num_parallel, inf_type, counts, log, ivs=ivs_pl)

            # init_ops = tf.group(init_ops, tf.initialize_variables(counts))
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
                    # print(out)
                    # print(true_out)
                    # for o, to in zip(out, true_out):
                    np.testing.assert_array_almost_equal(out, true_out)
                except AssertionError:
                    print("out", out)
                    print(true_out)
                    output_correct = False
                    self.test_failed = True

            if self.profile:
                # Create a suitable filename suffix
                fnm_suffix = op_name
                fnm_suffix += ("_GPU" if on_gpu else "_CPU")
                fnm_suffix += ("_MPE-LOG" if log else "_MPE") if inf_type == \
                    spn.InferenceType.MPE else ("_MARGINAL-LOG" if log else "_MARGINAL")
                fnm_suffix += ("_IV" if ivs is not None else "")
                # Create a profiling report
                profile_report(sess, ops, feed_dict, self.profiles_dir,
                               "sum_value_varying_sizes", fnm_suffix)

        # Return stats
        return OpTestResult(op_name, on_gpu, graph_size, ("Yes"),
                            ("No" if ivs is None else "Yes"), setup_time,
                            weights_init_time, run_times, output_correct)

    def _true_out(self, inf_type, inputs, ivs_per_sum, sum_indices, sum_sizes, sum_sizes_np):
        """ Computes true output """
        numpy_inputs = self._repeat_elements([(inputs, ind) for ind in sum_indices])
        w = np.ones(sum(sum_sizes_np))
        batch_size = inputs.shape[0]
        counts = np.arange(batch_size * len(sum_sizes_np)).reshape((len(sum_sizes_np), batch_size))\
            .transpose() + 10

        true_outs = sums_layer_mpe_path_numpy(numpy_inputs, sum_sizes_np, w, counts,
                                              ivs=ivs_per_sum)
        return true_outs

    def _run_test(self, test_name, op_funs, inputs, indices, ivs, inf_type, log):
        """Run a single test for multiple ops and devices."""
        cpu_results = []
        gpu_results = []
        for op_fun, inp, ind, iv in zip(op_funs, inputs, indices, ivs):
            # Decide on which devices
            use_gpu = ([True] if not self.without_gpu else []) + \
                      ([False] if not self.without_cpu else [])

            # Go through all combinations of devices and IVs
            for on_gpu in use_gpu:
                (gpu_results if on_gpu else cpu_results).append(
                    self._run_op_test(op_fun, inp, sum_indices=ind, inf_type=inf_type, log=log,
                                      ivs=iv, on_gpu=on_gpu))
        return TestResults(test_name, cpu_results, gpu_results)

    def run(self):
        """Run all tests."""
        print1("Running tests:", self.file)
        results = []

        sum_sizes_repeated = self._repeat_elements(self.sum_sizes)
        ivs = np.stack(
            [np.random.choice(s + 1, self.batch_size) - 1 for s in sum_sizes_repeated], axis=1)

        # Sum
        sum_inputs = np.random.rand(self.batch_size, self.num_cols)
        sum_indices = [random.sample(range(self.num_cols), size)
                       for size in self.sum_sizes]

        for inf_type, log, iv in itertools.product(
            [spn.InferenceType.MPE], [False, True], [None, ivs]
        ):
            name = 'InferenceType: {}{}'.format(
                "MARGINAL" if inf_type == spn.InferenceType.MARGINAL else "MPE",
                "-LOG" if log else "")
            r = self._run_test(name, [Ops.sums_layer, Ops.sums_layer_v2, Ops.par_sums, Ops.sum],
                               4 * [sum_inputs], 4 * [sum_indices], 4 * [iv],
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
        t = PerformanceTestMPEPath(batch_size=args.num_input_rows, sum_sizes=args.sum_sizes,
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
