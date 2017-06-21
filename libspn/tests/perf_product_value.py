#!/usr/bin/env python3

# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

import tensorflow as tf
import numpy as np
from itertools import product
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

    def product(inputs, num_input_cols, inf_type, log=False, output=None):
        num_inputs = len(inputs)
        # Create permuted indices based on number and size of inputs
        inds = map(int, np.arange(num_input_cols))
        permuted_inds = list(product(inds, repeat=num_inputs))
        permuted_inds_list = [list(elem) for elem in permuted_inds]
        permuted_inds_list_of_list = []
        for elem in permuted_inds_list:
            permuted_inds_list_of_list.append([elem[i:i+1] for i in
                                               range(0, len(elem), 1)])

        # Create inputs list by combining inputs and indices
        permuted_inputs = []
        for indices in permuted_inds_list_of_list:
            permuted_inputs.append([tuple(i) for i in zip(inputs, indices)])

        # Generate 'num_prods' Product nodes, connecting each to its inputs
        p = []
        for perm_inps in permuted_inputs:
            p = p + [spn.Product(*perm_inps)]

        # Connect all product nodes to a single root Sum node and generate its
        # weights
        root = spn.Sum(*p)
        root.generate_weights()

        if log:
            value_op = root.get_log_value(inference_type=inf_type)
        else:
            value_op = root.get_value(inference_type=inf_type)

        return spn.initialize_weights(root), value_op

    def perm_products(inputs, num_input_cols, inf_type, log=False, output=None):
        # Generate a single PermProducts node, modeling 'num_prods' product
        # nodes within, connecting it to inputs
        p = spn.PermProducts(*inputs)

        # Connect the PermProducts nodes to a single root Sum node and generate
        # its weights
        root = spn.Sum(p)
        root.generate_weights()

        if log:
            value_op = root.get_log_value(inference_type=inf_type)
        else:
            value_op = root.get_value(inference_type=inf_type)

        return spn.initialize_weights(root), value_op

class OpTestResult:
    """Result of a single test of a single op."""

    def __init__(self, op_name, on_gpu, graph_size, setup_time, run_times,
                 output_correct):
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
            return ("%3s %11s %5s %11s %15s %14s %10s" %
                    (dev, 'op', 'size', 'setup_time', 'first_run_time',
                     'rest_run_time', 'correct'))

        def get_res(res):
            """Helper function printing a single result."""
            return ("%15s %5d %11.2f %15.2f %14.2f %10s" %
                    (res.op_name, res.graph_size, res.setup_time * 1000,
                     res.run_times[0] * 1000, np.mean(res.run_times[1:]) * 1000,
                     res.output_correct))

        # Print results
        print1("\n-----------------------", file)
        print1("%s" % self.test_name, file)
        print1("-----------------------", file)
        print1(get_header("CPU"), file)
        for res in sorted(self.cpu_results, key=lambda x: len(x.op_name)):
            print1(get_res(res), file, red)
        print1(get_header("GPU"), file)
        for res in sorted(self.gpu_results, key=lambda x: len(x.op_name)):
            print1(get_res(res), file, red)


class PerformanceTest:

    def __init__(self, num_inputs, num_input_rows, num_input_cols, num_ops,
                 num_runs, without_cpu, without_gpu, log_devs, profile,
                 profiles_dir, file):
        self.num_inputs = num_inputs
        self.num_input_rows = num_input_rows
        self.num_input_cols = num_input_cols
        self.num_prods = pow(num_input_cols, num_inputs)
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
        print1("- num_inputs=%s" % num_inputs, file)
        print1("- num_input_rows=%s" % num_input_rows, file)
        print1("- num_input_cols=%s" % num_input_cols, file)
        print1("- num_prods=%s" % self.num_prods, file)
        print1("- num_ops=%s" % num_ops, file)
        print1("- num_runs=%s" % num_runs, file)
        print1("", file=file)


    def _true_output(self, inputs, inf_type=None):
        if inf_type == spn.InferenceType.MARGINAL:
            np_sum_op = np.sum
        elif inf_type == spn.InferenceType.MPE:
            np_sum_op = np.amax
        else:
            sys.exit('ERROR: Incorrect inference type: ', inf_type)

        # Create permuted indices based on number and size of inputs
        inds = map(int, np.arange(self.num_input_cols))
        permuted_inds = list(product(inds, repeat=self.num_inputs))
        off_sets = list(range(0, (self.num_inputs * self.num_input_cols),
                             self.num_input_cols))
        permuted_inds_list = []
        for perm_inds in permuted_inds:
            permuted_inds_list.append([p_ind + off_set for p_ind, off_set in
                                       zip(list(perm_inds), off_sets)])

        concatenated_inputs=np.concatenate(inputs, axis=1)
        products_output = np.concatenate([np.prod(concatenated_inputs[:, p_inds],
                                                  axis=1, keepdims=True) for
                                          p_inds in permuted_inds_list], axis=1)

        root_weight = 1.0 / self.num_prods
        return np_sum_op(products_output * root_weight, axis=1, keepdims=True)


    def _run_op_test(self, op_fun, inputs, log=False, on_gpu=True,
                     inf_type=spn.InferenceType.MARGINAL):
        """Run a single test for a single op."""
        # Preparations
        op_name = op_fun.__name__
        device_name = '/gpu:0' if on_gpu else '/cpu:0'

        # Print
        print2("--> %s: on_gpu=%s, num_inputs=%s, inputs_shape=%s, inference=%s, log=%s"
               % (op_name, on_gpu, self.num_inputs, inputs[0].shape, ("MPE" if \
                  inf_type == spn.InferenceType.MPE else "MARGINAL"), log),
                  self.file)

        # Compute true output
        true_out = self._true_output(inputs, inf_type)

        # Create graph
        tf.reset_default_graph()
        with tf.device(device_name):
            # Create input
            inputs_pl = [spn.ContVars(num_vars=self.num_input_cols) for _ in
                         range(self.num_inputs)]
            # Create ops
            start_time = time.time()
            init_ops, ops = op_fun(inputs_pl, self.num_input_cols, inf_type, log)
            for _ in range(self.num_ops - 1):
                # The tuple ensures that the next op waits for the output
                # of the previous op, effectively stacking the ops
                # but using the original input every time
                init_ops, ops = op_fun(inputs_pl, self.num_input_cols, inf_type,
                                       log, tf.tuple([ops])[0])
            setup_time = time.time() - start_time
        # Get num of graph ops
        graph_size = len(tf.get_default_graph().get_operations())
        # Run op multiple times
        output_correct = True
        with tf.Session(config=tf.ConfigProto(
                allow_soft_placement=False,
                log_device_placement=self.log_devs)) as sess:
            # Initialize weights of all the sum nodes in the graph
            init_ops.run()
            # Create feed dictionary
            feed = {inp_pl: inp for inp_pl, inp in zip(inputs_pl, inputs)}
            run_times = []
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
                             spn.InferenceType.MPE else ("_MARGINAL-LOG" if \
                             log else "_MARGINAL")

                with open('%s/timeline_value_%s.json' % (self.profiles_dir,
                          file_name), 'w') as f:
                    f.write(chrome_trace)

        # Return stats
        return OpTestResult(op_name, on_gpu, graph_size, setup_time, run_times,
                            output_correct)

    def _run_test(self, test_name, op_funs, inputs, inf_type, log):
        """Run a single test for multiple ops and devices."""
        cpu_results = []
        gpu_results = []
        for op_fun, inp in zip(op_funs, inputs):
            if not self.without_cpu:
                cpu_results.append(
                    self._run_op_test(op_fun, inp, log=log, on_gpu=False,
                                      inf_type=inf_type))
            if not self.without_gpu:
                gpu_results.append(
                    self._run_op_test(op_fun, inp, log=log, on_gpu=True,
                                      inf_type=inf_type))
        return TestResults(test_name, cpu_results, gpu_results)


    def run(self):
        """Run all tests."""
        print1("Running tests:", self.file)
        results = []

        # Product
        product_inputs = [np.random.rand(self.num_input_rows, self.num_input_cols)
                          for _ in range(self.num_inputs)]

        # PermProduct
        perm_products_inputs = product_inputs

        r = self._run_test('InferenceType: MARGINAL',
                           [Ops.product, Ops.perm_products],
                           [product_inputs, perm_products_inputs],
                           inf_type=spn.InferenceType.MARGINAL, log=False)
        results.append(r)

        r = self._run_test('InferenceType: MARGINAL-LOG',
                           [Ops.product, Ops.perm_products],
                           [product_inputs, perm_products_inputs],
                           inf_type=spn.InferenceType.MARGINAL, log=True)
        results.append(r)

        r = self._run_test('InferenceType: MPE',
                           [Ops.product, Ops.perm_products],
                           [product_inputs, perm_products_inputs],
                           inf_type=spn.InferenceType.MPE, log=False)
        results.append(r)

        r = self._run_test('InferenceType: MPE-LOG',
                           [Ops.product, Ops.perm_products],
                           [product_inputs, perm_products_inputs],
                           inf_type=spn.InferenceType.MPE, log=True)
        results.append(r)

        # Print results
        for res in results:
            res.print(self.file)

        if self.test_failed:
            print("\n ATLEAST ONE TEST FAILED!")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--num-inputs', default=2, type=int,
                        help="Num of input nodes")
    parser.add_argument('--num-input-rows', default=200, type=int,
                        help="Num of rows of inputs")
    parser.add_argument('--num-input-cols', default=10, type=int,
                        help="Num of cols of inputs")
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

    # Needed to generate indices for partially optimized cases
    if args.num_inputs < 2:
        sys.exit('ERROR: num_inputs must be >= 2')

    # Open a file
    f = None
    if args.save_to:
        f = open(args.save_to, 'w')

    try:
        t = PerformanceTest(args.num_inputs, args.num_input_rows,
                            args.num_input_cols, args.num_ops, args.num_runs,
                            args.without_cpu, args.without_gpu, args.log_devices,
                            args.profile, args.profiles_dir, f)
        t.run()
    finally:
        if f is not None:
            f.close()


if __name__ == '__main__':
    main()
