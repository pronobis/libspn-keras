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

from libspn.tests.profiler import profile_report
from libspn.tests.perf_sum_value_varying_sizes import sums_layer_numpy_common, OpTestResult, \
    PerformanceTest, TestResults

col.init()

red = col.Fore.RED
blue = col.Fore.BLUE
green = col.Fore.GREEN
yellow = col.Fore.YELLOW
magenta = col.Fore.MAGENTA


def assert_all_close(a, b, rtol=1e-6, atol=1e-6, msg=None):
    if not np.allclose(a, b, rtol=rtol, atol=atol):
        # Prints more details than np.testing.assert_allclose.
        #
        # NOTE: numpy.allclose (and numpy.testing.assert_allclose)
        # checks whether two arrays are element-wise equal within a
        # tolerance. The relative difference (rtol * abs(b)) and the
        # absolute difference atol are added together to compare against
        # the absolute difference between a and b.  Here, we want to
        # print out which elements violate such conditions.
        cond = np.logical_or(
            np.abs(a - b) > atol + rtol * np.abs(b), np.isnan(a) != np.isnan(b))
        if a.ndim:
            x = a[np.where(cond)]
            y = b[np.where(cond)]
            # print("not close where = ", np.where(cond))
        else:
            # np.where is broken for scalars
            x, y = a, b
        # print("not close lhs = ", x)
        # print("not close rhs = ", y)
        # print("not close dif = ", np.abs(x - y))
        # print("not close tol = ", atol + rtol * np.abs(y))
        # print("dtype = %s, shape = %s" % (a.dtype, a.shape))
        np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, err_msg=msg)


def print1(str, file, color=yellow):
    if file:
        print(str, file=file)
    print(color + str + col.Style.RESET_ALL)


def print2(str, file):
    if file:
        print(str, file=file)
    print(blue + str + col.Style.RESET_ALL)


def sums_layer_mpe_path_numpy(inputs, sums_sizes, weights, ivs=None,
                              inf_type=spn.InferenceType.MARGINAL):
    """ Computes the output of _compute_mpe_path with numpy """
    inputs_to_reduce, iv_mask_per_sum, weights_per_sum = sums_layer_numpy_common(
        inputs, ivs, sums_sizes, weights)
    # Get max index for sum node
    if inf_type == spn.InferenceType.MPE:
        # We don't have to think about the max individual sum outcome in this case, it will just
        # be the max of the weighted inputs at the bottom layer
        weighted_sums = [x * np.reshape(w/np.sum(w), (1, -1)) * iv
                         for x, w, iv in zip(inputs_to_reduce, weights_per_sum, iv_mask_per_sum)]
        weighted_sums_concat = np.concatenate(weighted_sums, axis=1)
        max_indices = np.argmax(weighted_sums_concat, axis=1)

        # Compute the concatenated counts
        out = np.zeros_like(weighted_sums_concat)
        out[np.arange(out.shape[0]), max_indices] = np.ones(out.shape[0])

        # Split to obtain counts per sum
        splits = np.cumsum(sums_sizes)[:-1]
        return np.split(out, splits, axis=1)
    else:
        # In this case, we first have to consider the sum with the largest outcome
        weighted_sums = [x * np.reshape(w / np.sum(w), (1, -1)) * iv
                         for x, w, iv in zip(inputs_to_reduce, weights_per_sum, iv_mask_per_sum)]
        sum_outcomes = np.concatenate([np.sum(s, axis=1, keepdims=True) for s in weighted_sums],
                                      axis=1)

        sum_indices = np.argmax(sum_outcomes, axis=1)

        # Now we compute the max per 'winning' sum
        out = [np.zeros_like(w) for w in weighted_sums]
        for i, ind in enumerate(sum_indices):
            max_ind = np.argmax(weighted_sums[ind][i])
            out[ind][i, max_ind] = 1

        return out


class Ops:

    @staticmethod
    def par_w_inp(inputs, sum_indices, repetitions, inf_type, log=False, ivs=None, weights=None):
        """ Creates the graph using only ParSum nodes """
        weights = np.split(weights, np.cumsum([len(ind) * repetitions for ind in sum_indices])[:-1])

        parallel_sum_nodes = []
        for inp in inputs:
            parallel_sum_nodes.append(spn.ParSums(inp, num_sums=repetitions))
        weight_nodes = [s.generate_weights(w.tolist()) for s, w in zip(parallel_sum_nodes, weights)]
        if ivs:
            [s.set_ivs(iv) for s, iv in zip(parallel_sum_nodes, ivs)]
        root = spn.Sum(*parallel_sum_nodes)
        root.generate_weights()

        mpe_path_gen = spn.MPEPath(value_inference_type=inf_type, log=log)
        mpe_path_gen.get_mpe_path(root)
        path_op = [mpe_path_gen.counts[w] for w in weight_nodes]
        input_counts = [mpe_path_gen.counts[inp] for inp in inputs]
        return spn.initialize_weights(root), tf.tuple(path_op + input_counts)[:len(path_op)]


    @staticmethod
    def par_w(inputs, sum_indices, repetitions, inf_type, log=False, ivs=None, weights=None):
        """ Creates the graph using only ParSum nodes """
        weights = np.split(weights, np.cumsum([len(ind) * repetitions for ind in sum_indices])[:-1])

        parallel_sum_nodes = []
        for inp in inputs:
            parallel_sum_nodes.append(spn.ParSums(inp, num_sums=repetitions))
        weight_nodes = [s.generate_weights(w.tolist()) for s, w in zip(parallel_sum_nodes, weights)]
        if ivs:
            [s.set_ivs(iv) for s, iv in zip(parallel_sum_nodes, ivs)]
        root = spn.Sum(*parallel_sum_nodes)
        root.generate_weights()

        mpe_path_gen = spn.MPEPath(value_inference_type=inf_type, log=log)
        mpe_path_gen.get_mpe_path(root)
        path_op = [mpe_path_gen.counts[w] for w in weight_nodes]
        return spn.initialize_weights(root), path_op

    @staticmethod
    def layer_cnt_gather(inputs, sum_indices, repetitions, inf_type, log=False, ivs=None, weights=None):
        """ Creates the graph using a SumsLayer node """
        spn.conf.lru_size = 0
        return Ops._sums_layer_common(inf_type, inputs, ivs, log, repetitions, sum_indices,
                                      sumslayer_count_with_matmul=False, weights=weights)

    @staticmethod
    def layer_cnt_matmul(inputs, sum_indices, repetitions, inf_type, log=False, ivs=None, weights=None):
        """ Creates the graph using a SumsLayer node """
        spn.conf.lru_size = 0
        return Ops._sums_layer_common(inf_type, inputs, ivs, log, repetitions, sum_indices,
                                      sumslayer_count_with_matmul=True, weights=weights)

    # @staticmethod
    # def sums_layer_v3(inputs, sum_indices, repetitions, inf_type, log=False, ivs=None):
    #     """ Creates the graph using a SumsLayer node """
    #     spn.conf.lru_size = 0
    #     return Ops._sums_layer_common(inf_type, inputs, ivs, log, repetitions, sum_indices,
    #                                   add_counts_in_sums_layer=True, use_lru=True)

    @staticmethod
    def _sums_layer_common(inf_type, inputs, ivs, log, repetitions, sum_indices,
                           sumslayer_count_with_matmul=True, weights=None):
        repeated_inputs = []
        repeated_sum_sizes = []
        offset = 0
        for ind in sum_indices:
            # Indices are given by looking at the sizes of the sums
            size = len(ind)
            repeated_inputs.extend([(inputs[0], list(range(offset, offset + size)))
                                    for _ in range(repetitions)])
            repeated_sum_sizes.extend([size for _ in range(repetitions)])
            offset += size

        # Globally configure to add up the sums before passing on the values to children
        spn.conf.sumslayer_count_with_matmul = sumslayer_count_with_matmul
        sums_layer = spn.SumsLayer(*repeated_inputs, num_sums_or_sizes=repeated_sum_sizes)
        weight_node = sums_layer.generate_weights(weights)
        if ivs:
            sums_layer.set_ivs(*ivs)
        # Connect a single sum to group outcomes
        root = spn.Sum(sums_layer)
        root.generate_weights()
        # Then build MPE path Ops
        mpe_path_gen = spn.MPEPath(value_inference_type=inf_type, log=log)
        mpe_path_gen.get_mpe_path(root)
        path_op = [tf.tuple([mpe_path_gen.counts[weight_node], mpe_path_gen.counts[inputs[0]]])[0]]
        return spn.initialize_weights(root), path_op


class PerformanceTestMPEPath(PerformanceTest):

    def _run_op_test(self, op_fun, inputs, sum_indices=None, inf_type=spn.InferenceType.MARGINAL,
                     log=False, on_gpu=True, ivs=None, indices=False):
        """Run a single test for a single op."""

        # Preparations
        op_name = op_fun.__name__
        device_name = '/gpu:0' if on_gpu else '/cpu:0'

        # Print
        print2("--> %s: on_gpu=%s, inputs_shape=%s, inference=%s, log=%s, IVs=%s"
               % (op_name, on_gpu, (self.batch_size, self.num_cols),
                  ("MPE" if inf_type == spn.InferenceType.MPE else "MARGINAL"), log,
                  ("No" if ivs is None else "Yes")), self.file)

        # Reset graph
        tf.reset_default_graph()

        # Compute the true output
        sum_sizes = [len(ind) for ind in sum_indices]
        ivs_per_sum = np.split(ivs, ivs.shape[1], axis=1) if ivs is not None else None
        sum_sizes_np = self._repeat_elements(sum_sizes)
        w = np.random.rand(sum(sum_sizes_np))
        true_out = self._true_out(inf_type, inputs, ivs_per_sum, sum_indices, sum_sizes,
                                  sum_sizes_np, w)

        # Compute the true output, involves grouping together individual sum counts. This is
        # different for each of the Ops considered
        max_size = max(sum_sizes)
        if op_fun in [Ops.layer_cnt_gather, Ops.layer_cnt_matmul]:
            padded = [np.concatenate(
                (o, np.zeros((self.batch_size, max_size - o.shape[1]))),
                axis=1) for o in true_out]
            true_out = [np.concatenate(padded, axis=1).reshape((-1, len(sum_sizes_np), max_size))]
        if op_fun in [Ops.par_w, Ops.par_w_inp]:
            true_out = [np.concatenate(true_out[i:i+self.num_parallel], axis=1)
                        .reshape((self.batch_size, self.num_parallel, -1))
                        for i in range(0, len(true_out), self.num_parallel)]

        # Set up the graph
        with tf.device(device_name):
            # Create input placeholders
            if op_fun in [Ops.layer_cnt_gather, Ops.layer_cnt_matmul]:
                # For the sums layer graphs, we use a single ContVars input
                input_size = sum(inp.shape[1] for inp in inputs)
                inputs_pl = [spn.ContVars(num_vars=input_size)]
                feed_dict = {inputs_pl[0]: np.concatenate(inputs, axis=1)}
            else:
                # Otherwise, each parallel sums node receives its own input
                inputs_pl = [spn.ContVars(num_vars=inp.shape[1]) for inp in inputs]
                feed_dict = {inp_pl: inp for inp_pl, inp in zip(inputs_pl, inputs)}

            # Create placeholders for IVs
            if ivs is not None:
                if op_fun in [Ops.par_w, Ops.par_w_inp]:
                    ivs_pl = [spn.IVs(num_vars=self.num_parallel, num_vals=len(ind))
                              for ind in sum_indices]
                    ivs = np.split(ivs, len(self.sum_sizes), axis=1)
                else:
                    ivs = [ivs]
                    ivs_pl = [spn.IVs(num_vars=len(sum_sizes_np), num_vals=max(sum_sizes))]
                # Set the placeholders
                for iv_pl, iv in zip(ivs_pl, ivs):
                    feed_dict[iv_pl] = iv
            else:
                ivs_pl = None

            # Set up the actual operations for the MPE path
            start_time = time.time()
            init_ops, ops = op_fun(
                inputs_pl, sum_indices, self.num_parallel, inf_type, log, ivs=ivs_pl, weights=w
            )
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
                    for o, to in zip(out, true_out):
                        assert_all_close(o, to)
                except AssertionError:
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
        return OpTestResult(op_name, on_gpu, graph_size, ("Yes"), ("No" if ivs is None else "Yes"),
                            setup_time, weights_init_time, run_times, output_correct)

    def _true_out(self, inf_type, inputs, ivs_per_sum, sum_indices, sum_sizes, sum_sizes_np, w):
        """ Computes true output """
        numpy_inputs = self._repeat_elements([(inp, None) for inp in inputs])
        true_outs = sums_layer_mpe_path_numpy(numpy_inputs, sum_sizes_np, w, ivs=ivs_per_sum,
                                              inf_type=inf_type)
        return true_outs

    def _run_test(self, test_name, op_funs, inputs, indices, ivs, inf_type, log):
        """Run a single test for multiple ops and devices."""
        cpu_results = []
        gpu_results = []
        for op_fun, inp, ind, iv in zip(op_funs, inputs, indices, ivs):
            # Run tests for CPU and GPU
            x = [inp[:, i] for i in ind]
            if not self.without_cpu:
                cpu_results.append(self._run_op_test(
                    op_fun, x, sum_indices=ind, inf_type=inf_type, log=log, ivs=iv, on_gpu=False))
            if not self.without_gpu:
                gpu_results.append(self._run_op_test(
                    op_fun, x, sum_indices=ind, inf_type=inf_type, log=log, ivs=iv, on_gpu=True))
        return TestResults(test_name, cpu_results, gpu_results)

    def run(self):
        """Run all tests."""
        print1("Running tests:", self.file)
        results = []

        # Create IVs
        sum_sizes_repeated = self._repeat_elements(self.sum_sizes)
        ivs = np.stack(
            [np.random.choice(s + 1, self.batch_size) - 1 for s in sum_sizes_repeated], axis=1)

        # Create sum inputs
        sum_inputs = np.random.rand(self.batch_size, self.num_cols)
        sum_indices = [random.sample(range(self.num_cols), size) for size in self.sum_sizes]

        # Go over all combinations of inference type, log vs. non-log and ivs vs. no ivs
        for inf_type, log, iv in itertools.product(
            [spn.InferenceType.MARGINAL, spn.InferenceType.MPE], [False, True], [None, ivs]
        ):
            name = 'InferenceType: {}{}'.format(
                "MARGINAL" if inf_type == spn.InferenceType.MARGINAL else "MPE",
                "-LOG" if log else "")
            r = self._run_test(name,
                               [Ops.layer_cnt_matmul, Ops.layer_cnt_gather,
                                Ops.par_w, Ops.par_w_inp],
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
