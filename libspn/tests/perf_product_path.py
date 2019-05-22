#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
from itertools import product, chain
from context import libspn as spn
import time
import argparse
import colorama as col
import sys
from tensorflow.python.client import timeline
import os
import random
col.init()


red = col.Fore.RED
blue = col.Fore.BLUE
green = col.Fore.GREEN
yellow = col.Fore.YELLOW
magenta = col.Fore.MAGENTA
cyan = col.Fore.CYAN
white = col.Fore.WHITE
black = col.Fore.BLACK


def print1(str, file, color=yellow):
    if file:
        print(str, file=file)
    print(color + str + col.Style.RESET_ALL)


def print2(str, file):
    if file:
        print(str, file=file)
    print(blue + str + col.Style.RESET_ALL)


class Ops:

    def product(inputs, num_inputs, num_input_cols, num_prods, inf_type,
                indices=None, log=False, output=None):
        p = []
        for inps, n_inp_cols in zip(inputs, num_input_cols):
            num_inputs = len(inps)
            # Create permuted indices based on number and size of inputs
            inds = map(int, np.arange(n_inp_cols))
            permuted_inds = list(product(inds, repeat=num_inputs))
            permuted_inds_list = [list(elem) for elem in permuted_inds]
            permuted_inds_list_of_list = []
            for elem in permuted_inds_list:
                permuted_inds_list_of_list.append([elem[i:i+1] for i in
                                                   range(0, len(elem), 1)])

            # Create inputs list by combining inputs and indices
            permuted_inputs = []
            for indices in permuted_inds_list_of_list:
                permuted_inputs.append([tuple(i) for i in zip(inps, indices)])

            # Generate 'n_prods' Product nodes, connecting each to its inputs
            for perm_inps in permuted_inputs:
                p = p + [spn.Product(*perm_inps)]

        # Connect all product nodes to a single root Sum node and generate its
        # weights
        root = spn.Sum(*p)
        root.generate_weights()

        if log:
            mpe_path_gen = spn.MPEPath(value_inference_type=inf_type, log=True)
        else:
            mpe_path_gen = spn.MPEPath(value_inference_type=inf_type, log=False)

        mpe_path_gen.get_mpe_path(root)
        path_ops = [mpe_path_gen.counts[inp] for inp in
                    list(chain.from_iterable(inputs))]
        return spn.initialize_weights(root), path_ops

    def perm_products(inputs, num_inputs, num_input_cols, num_prods, inf_type,
                      indices=None, log=False, output=None):
        if indices is not None:
            # Create inputs list with indices
            inputs_list = [[(inp, ind) for inp, ind in zip(inps, inds)] for inps,
                           inds in zip(inputs, indices)]
        else:
            inputs_list = inputs

        if isinstance(inputs, list):  # Is a list of RawLeaf inputs - Multiple inputs
            # Generate 'len(inputs)' PermuteProducts nodes, modeling 'n_prods' products
            # within each
            p = [spn.PermuteProducts(*inps) for inps in inputs]
        else:  # Is a single input of type RawLeaf - A single input
            num_inputs_array = np.array(num_inputs)
            num_input_cols_array = np.array(num_input_cols)
            num_cols = num_input_cols[0]
            num_vars = int(np.sum(num_inputs_array * num_input_cols_array))

            indices_list = [list(range(i, i+num_cols)) for i in range(0, num_vars,
                                                                      num_cols)]
            num_inputs_cumsum = np.cumsum(num_inputs_array).tolist()
            num_inputs_cumsum.insert(0, 0)

            inputs_list = [[(inputs, inds) for inds in indices_list[start:stop]]
                           for start, stop in zip(num_inputs_cumsum[:-1],
                                                  num_inputs_cumsum[1:])]

            # Generate 'len(inputs)' PermuteProducts nodes, modeling 'n_prods'
            # products within each, and inputs for each node emination from a
            # commoninput source
            p = [spn.PermuteProducts(*inps) for inps in inputs_list]

        # Connect all PermuteProducts nodes to a single root Sum node and generate
        # its weights
        root = spn.Sum(*p)
        root.generate_weights()

        if log:
            mpe_path_gen = spn.MPEPath(value_inference_type=inf_type, log=True)
        else:
            mpe_path_gen = spn.MPEPath(value_inference_type=inf_type, log=False)

        mpe_path_gen.get_mpe_path(root)
        if isinstance(inputs, list):  # Is a list of RawLeaf inputs - Multiple inputs
            path_ops = [mpe_path_gen.counts[inp] for inp in
                        list(chain.from_iterable(inputs))]
        else:  # Is a single input of type RawLeaf - A single input
            path_ops = mpe_path_gen.counts[inputs]
        return spn.initialize_weights(root), path_ops

    def products(inputs, num_inputs, num_input_cols, num_prods, inf_type,
                 indices=None, log=False, output=None):
        p = []
        # Generate 'len(inputs)' Products node, modelling 'n_prods' ∈ 'num_prods'
        # products within each
        for inps, n_inp_cols, n_prods in zip(inputs, num_input_cols, num_prods):
            num_inputs = len(inps)
            # Create permuted indices based on number and size of inps
            inds = map(int, np.arange(n_inp_cols))
            permuted_inds = list(product(inds, repeat=num_inputs))
            permuted_inds_list = [list(elem) for elem in permuted_inds]
            permuted_inds_list_of_list = []
            for elem in permuted_inds_list:
                permuted_inds_list_of_list.append([elem[i:i+1] for i in
                                                   range(0, len(elem), 1)])

            # Create inputs-list by combining inps and indices
            permuted_inputs = []
            for indices in permuted_inds_list_of_list:
                permuted_inputs.append([tuple(i) for i in zip(inps, indices)])
            permuted_inputs = list(chain.from_iterable(permuted_inputs))

            # Generate a single Products node, modeling 'n_prods' product nodes
            # within, connecting it to inputs
            p = p + [spn.Products(*permuted_inputs, num_prods=n_prods)]

        # Connect all product nodes to a single root Sum node and generate its
        # weights
        root = spn.Sum(*p)
        root.generate_weights()

        if log:
            mpe_path_gen = spn.MPEPath(value_inference_type=inf_type, log=True)
        else:
            mpe_path_gen = spn.MPEPath(value_inference_type=inf_type, log=False)

        mpe_path_gen.get_mpe_path(root)
        path_ops = [mpe_path_gen.counts[inp] for inp in
                    list(chain.from_iterable(inputs))]
        return spn.initialize_weights(root), path_ops

    def products_layer(inputs, num_inputs, num_input_cols, num_prods, inf_type,
                       indices=None, log=False, output=None):
        products_inputs = []
        num_or_size_prods = []
        if isinstance(inputs, list):  # Is a list of RawLeaf inputs - Multiple inputs
            for inps, n_inp_cols, n_prods in zip(inputs, num_input_cols, num_prods):
                num_inputs = len(inps)
                # Create permuted indices based on number and size of inputs
                inds = map(int, np.arange(n_inp_cols))
                permuted_inds = list(product(inds, repeat=num_inputs))
                permuted_inds_list = [list(elem) for elem in permuted_inds]
                permuted_inds_list_of_list = []
                for elem in permuted_inds_list:
                    permuted_inds_list_of_list.append([elem[i:i+1] for i in
                                                       range(0, len(elem), 1)])

                # Create inputs list by combining inputs and indices
                permuted_inputs = []
                for indices in permuted_inds_list_of_list:
                    permuted_inputs.append([tuple(i) for i in zip(inps, indices)])
                products_inputs += list(chain.from_iterable(permuted_inputs))

                # Create products-size list
                num_or_size_prods += [num_inputs] * n_prods
        else:  # Is a single input of type RawLeaf - A single input
            outer_offset = 0
            permuted_inds_list = []
            for n_inps, n_inp_cols in zip(num_inputs, num_input_cols):
                # Create permuted indices based on number and size of inputs
                inds = map(int, np.arange(n_inp_cols))
                permuted_inds = list(product(inds, repeat=n_inps))
                offsets = \
                    np.array(list(range(0, (n_inps * n_inp_cols), n_inp_cols))) \
                    + outer_offset
                outer_offset += n_inps * n_inp_cols
                for perm_inds in permuted_inds:
                    permuted_inds_list.append([p_ind + off for p_ind, off in
                                               zip(list(perm_inds), offsets)])

            # Content of list object 'perm_inds' needs to be of type int, if not
            # input_parser in Input class complains
            products_inputs = [(inputs, list(map(int, perm_inds))) for perm_inds
                               in permuted_inds_list]
            num_or_size_prods = [len(perm_inds) for perm_inds in permuted_inds_list]

        # Generate a single ProductsLayer node, modeling 'sum(num_prods)' products
        # within, connecting it to inputs
        p = spn.ProductsLayer(*products_inputs, num_or_size_prods=num_or_size_prods)

        # Connect all product nodes to a single root Sum node and generate its
        # weights
        root = spn.Sum(p)
        root.generate_weights()

        if log:
            mpe_path_gen = spn.MPEPath(value_inference_type=inf_type, log=True)
        else:
            mpe_path_gen = spn.MPEPath(value_inference_type=inf_type, log=False)

        mpe_path_gen.get_mpe_path(root)
        if isinstance(inputs, list):
            path_ops = [mpe_path_gen.counts[inp] for inp in
                        list(chain.from_iterable(inputs))]
        else:
            path_ops = mpe_path_gen.counts[inputs]
        return spn.initialize_weights(root), path_ops


class OpTestResult:
    """Result of a single test of a single op."""

    def __init__(self, op_name, on_gpu, graph_size, indices, single_input,
                 setup_time, run_times, output_correct):
        self.op_name = op_name
        self.on_gpu = on_gpu
        self.graph_size = graph_size
        self.indices = indices
        self.single_input = single_input
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
            return ("%3s %11s %5s %5s %13s %11s %15s %14s %10s" %
                    (dev, 'op', 'size', 'indices', 'single_input', 'setup_time',
                     'first_run_time', 'rest_run_time', 'correct'))

        def get_res(res):
            """Helper function printing a single result."""
            return ("%15s %5d %5s %10s %14.2f %15.2f %14.2f %10s" %
                    (res.op_name, res.graph_size, res.indices, res.single_input,
                     res.setup_time * 1000, res.run_times[0] * 1000,
                     np.mean(res.run_times[1:]) * 1000, res.output_correct))

        # Print results
        print()
        print1("--------------------------------------------------------", file)
        print1("%s" % self.test_name, file)
        print1("--------------------------------------------------------", file)
        print1(get_header("CPU"), file)
        for res in sorted(self.cpu_results, key=lambda x: len(x.op_name)):
            print1(get_res(res), file, (black if res.op_name is "product" else
                   white if res.op_name is "products" else
                   (green if res.single_input is "Yes" else cyan) if res.op_name
                   is "products_layer" else (blue if res.indices is "No" else
                                             (magenta if res.single_input is "No"
                                              else red))))

        print1(get_header("GPU"), file)
        for res in sorted(self.gpu_results, key=lambda x: len(x.op_name)):
            print1(get_res(res), file, (black if res.op_name is "product" else
                   white if res.op_name is "products" else
                   (green if res.single_input is "Yes" else cyan) if res.op_name
                   is "products_layer" else (blue if res.indices is "No" else
                                             (magenta if res.single_input is "No"
                                              else red))))


class PerformanceTest:

    def __init__(self, num_input_rows, num_input_cols, num_batches, num_runs,
                 without_cpu, without_gpu, without_product, without_products,
                 without_indices, without_single_input, log_devs, profile,
                 profiles_dir, file):
        self.num_input_rows = num_input_rows
        self.num_input_cols = num_input_cols
        self.num_batches = num_batches
        self.num_runs = num_runs
        self.without_cpu = without_cpu
        self.without_gpu = without_gpu
        self.without_product = without_product
        self.without_products = without_products
        self.without_indices = without_indices
        self.without_single_input = without_single_input
        self.log_devs = log_devs
        self.profile = profile
        self.profiles_dir = profiles_dir
        self.file = file
        self.test_failed = False

        print1("Params:", file)
        print1("- num_input_rows=%s" % num_input_rows, file)
        print1("- num_input_cols=%s" % num_input_cols, file)
        print1("- num_batches=%s" % num_batches, file)
        print1("- num_runs=%s" % num_runs, file)
        print1("", file=file)

    def _true_output(self, inputs, num_inputs, num_prods, indices=None,
                     single_input=False):
        permuted_indices = []
        products_output = []

        for inps, n_inps, n_inp_cols in zip(inputs, num_inputs,
                                            self.num_input_cols):
            # Create permuted indices based on number and size of inputs
            inds = map(int, np.arange(n_inp_cols))
            permuted_inds = list(product(inds, repeat=n_inps))
            off_sets = list(range(0, (n_inps * n_inp_cols), n_inp_cols))
            permuted_inds_list = []
            for perm_inds in permuted_inds:
                permuted_inds_list.append([p_ind + off_set for p_ind, off_set in
                                           zip(list(perm_inds), off_sets)])
            permuted_indices.append(permuted_inds_list)

            concatenated_inputs = np.concatenate(inps, axis=1)
            products_output.append(np.concatenate([np.prod(concatenated_inputs[:, p_inds],
                                                  axis=1, keepdims=True) for p_inds
                                                  in permuted_inds_list], axis=1))

        product_counts = [np.zeros_like(np.concatenate(inps, axis=1)) for inps in inputs]
        products_output = np.concatenate(products_output, axis=1)

        root_weight = 1.0 / sum(num_prods)
        root_counts = np.eye(sum(num_prods))[np.argmax((products_output
                                                        * root_weight), axis=1)]
        root_counts = np.split(root_counts, np.cumsum(num_prods), axis=1)[:-1]

        input_sizes = list(chain(*[[col] * inp for col, inp in
                           zip(self.num_input_cols, num_inputs)]))

        for prod_cnts, perm_inds, r_cnts in zip(product_counts, permuted_indices, root_counts):
            for idx, p_inds in enumerate(perm_inds):
                prod_cnts[:, p_inds] += r_cnts[:, [idx]]

        product_counts = np.concatenate(product_counts, axis=1)
        if single_input:
            return product_counts
        else:
            return np.split(product_counts, np.cumsum(input_sizes), axis=1)

    def _run_op_test(self, op_fun, inputs, num_inputs, indices=None, single_input=False,
                     log=False, on_gpu=True, inf_type=spn.InferenceType.MARGINAL):
        """Run a single test for a single op."""
        # Preparations
        op_name = op_fun.__name__
        device_name = '/gpu:0' if on_gpu else '/cpu:0'

        # Print
        print2("--> %s: on_gpu=%s, num_inputs=%s, inputs_shape=%s, indices=%s,\
 single_input= %s, inference=%s, log=%s"
               % (op_name, on_gpu, num_inputs, inputs[0][0].shape, ("False" if
                  indices is None else "True"), single_input, ("MPE" if
                  inf_type == spn.InferenceType.MPE else "MARGINAL"), log), self.file)

        # Decern number of products modelled in each node
        num_prods = [pow(n_inp_cols, n_inps) for n_inps, n_inp_cols in
                     zip(num_inputs, self.num_input_cols)]

        # Compute true output
        true_out = self._true_output(inputs, num_inputs, num_prods, indices,
                                     single_input)

        # Create graph
        tf.reset_default_graph()
        with tf.device(device_name):
            # Create inputs
            if single_input:
                num_inputs_array = np.array(num_inputs)
                num_input_cols_array = np.array(self.num_input_cols)
                num_vars = int(np.sum(num_inputs_array * num_input_cols_array))
                inputs_pl = spn.RawLeaf(num_vars=num_vars)
            else:
                inputs_pl = [[spn.RawLeaf(num_vars=n_inp_cols) for _ in
                              range(n_inps)] for n_inps, n_inp_cols in
                             zip(num_inputs, self.num_input_cols)]
            # Create ops
            start_time = time.time()
            init_ops, ops = op_fun(inputs_pl, num_inputs, self.num_input_cols,
                                   num_prods, inf_type, indices, log)
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
            if op_fun is not Ops.products_layer and single_input is False:
                # Create feed dictionary
                feed = {inp_pl: inp for inp_pl, inp in zip(chain(*inputs_pl),
                                                           chain(*inputs))}
            else:
                concatenated_input = np.concatenate(list(chain(*inputs)), axis=1)
            run_times = []
            batch_size = self.num_input_rows // self.num_batches
            for n in range(self.num_runs):
                # Run
                if op_fun is Ops.products_layer:
                    outputs = []
                    start_time = time.time()
                    for i in range(self.num_batches):
                        start = i * batch_size
                        stop = (i + 1) * batch_size
                        # Create feed dictionary
                        if single_input:
                            feed = {inputs_pl: concatenated_input[start:stop, :]}
                        else:
                            feed = {inp_pl: inp[start:stop, :] for inp_pl, inp
                                    in zip(chain(*inputs_pl), chain(*inputs))}
                        out = sess.run(ops, feed_dict=feed)
                        outputs.append(out)
                    run_times.append(time.time() - start_time)
                    out = np.vstack(outputs)
                else:
                    start_time = time.time()
                    # Create feed dictionary
                    if single_input:
                        feed = {inputs_pl: concatenated_input}
                    out = sess.run(ops, feed_dict=feed)
                    run_times.append(time.time() - start_time)

                # Test value
                try:
                    for o, true_o in zip(out, true_out):
                        np.testing.assert_array_almost_equal(o, true_o)
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
                file_name += ("_SINGLE-INPUT" if (op_fun is Ops.products_layer
                              and single_input is True) else "")

                with open('%s/timeline_path_%s.json' % (self.profiles_dir,
                          file_name), 'w') as f:
                    f.write(chrome_trace)

        # Return stats
        return OpTestResult(op_name, on_gpu, graph_size, ("No" if (op_fun is
                            Ops.perm_products and indices is None and single_input
                            is False) else "Yes"), ("Yes" if single_input else
                            "No"), setup_time, run_times, output_correct)

    def _run_test(self, test_name, op_funs, inputs, num_inputs, indices, inf_type,
                  log):
        """Run a single test for multiple ops and devices."""

        # Atleast two inputs are needed for modelling multiple products in PermuteProducts
        if not all(n_inp >= 2 for n_inp in num_inputs):
            sys.exit('ERROR: All num_inputs must be >= 2')

        # num_inputs and num_input_cols should be of the same length
        if len(num_inputs) != len(self.num_input_cols):
            sys.exit('Error: Lengths of num_inputs and num_input_cols must be'
                     ' the same!')

        cpu_results = []
        gpu_results = []
        for op_fun in op_funs:
            if not self.without_cpu:
                cpu_results.append(
                    self._run_op_test(op_fun, inputs, num_inputs, indices=None,
                                      single_input=False, log=log, on_gpu=False,
                                      inf_type=inf_type))
                # PermProds with indices
                if op_fun is Ops.perm_products and not self.without_indices:
                        cpu_results.append(
                            self._run_op_test(op_fun, inputs, num_inputs, indices,
                                              single_input=False, log=log,
                                              on_gpu=False, inf_type=inf_type))
                # PermProds with single-input
                if op_fun is Ops.perm_products:
                        cpu_results.append(
                            self._run_op_test(op_fun, inputs, num_inputs,
                                              indices=None, single_input=True,
                                              log=log, on_gpu=False,
                                              inf_type=inf_type))
                # ProductsLayer with single-input
                if op_fun is Ops.products_layer and not self.without_single_input:
                        cpu_results.append(
                            self._run_op_test(op_fun, inputs, num_inputs,
                                              indices=None, single_input=True,
                                              log=log, on_gpu=False,
                                              inf_type=inf_type))
            if not self.without_gpu:
                gpu_results.append(
                    self._run_op_test(op_fun, inputs, num_inputs, indices=None,
                                      single_input=False, log=log, on_gpu=True,
                                      inf_type=inf_type))
                # PermProds with indices
                if op_fun is Ops.perm_products and not self.without_indices:
                        gpu_results.append(
                            self._run_op_test(op_fun, inputs, num_inputs, indices,
                                              single_input=False, log=log,
                                              on_gpu=True, inf_type=inf_type))
                # PermProds with single-input
                if op_fun is Ops.perm_products:
                        gpu_results.append(
                            self._run_op_test(op_fun, inputs, num_inputs,
                                              indices=None, single_input=True,
                                              log=log, on_gpu=True,
                                              inf_type=inf_type))
                # ProductsLayer with single-input
                if op_fun is Ops.products_layer and not self.without_single_input:
                        gpu_results.append(
                            self._run_op_test(op_fun, inputs, num_inputs,
                                              indices=None, single_input=True,
                                              log=log, on_gpu=True,
                                              inf_type=inf_type))
        return TestResults(test_name, cpu_results, gpu_results)

    def run(self):
        """Run all tests."""
        print1("Running tests:", self.file)
        results = []

        num_inputs = [5] * 10
        inputs = [[np.random.rand(self.num_input_rows, n_inp_cols) for _ in
                  range(n_inps)] for n_inps, n_inp_cols in zip(num_inputs,
                  self.num_input_cols)]
        indices = [[random.sample(range(n_inp_cols), k=n_inp_cols) for _ in
                    range(n_inps)] for n_inps, n_inp_cols in
                   zip(num_inputs, self.num_input_cols)]
        num_prods = [pow(n_inp_cols, n_inps) for n_inps, n_inp_cols in
                     zip(num_inputs, self.num_input_cols)]

        r = self._run_test(('NON-PADDED\nInferenceType: MARGINAL\n'
                            'num_inputs: %s \nnum_prods: %s' % (num_inputs, num_prods)),
                           [] + ([Ops.product] if not self.without_product else [])
                           + ([Ops.products] if not self.without_products else [])
                           + [Ops.perm_products, Ops.products_layer],
                           inputs, num_inputs, indices,
                           inf_type=spn.InferenceType.MARGINAL, log=False)
        results.append(r)

        r = self._run_test(('NON-PADDED\nInferenceType: MARGINAL-LOG\n'
                            'num_inputs: %s \nnum_prods: %s' % (num_inputs, num_prods)),
                           [] + ([Ops.product] if not self.without_product else [])
                           + ([Ops.products] if not self.without_products else [])
                           + [Ops.perm_products, Ops.products_layer],
                           inputs, num_inputs, indices,
                           inf_type=spn.InferenceType.MARGINAL, log=True)
        results.append(r)

        num_inputs = [2, 3, 4, 5, 4, 3, 2, 3, 4, 5]
        inputs = [[np.random.rand(self.num_input_rows, n_inp_cols) for _ in
                  range(n_inps)] for n_inps, n_inp_cols in zip(num_inputs,
                  self.num_input_cols)]
        indices = [[random.sample(range(n_inp_cols), k=n_inp_cols) for _ in
                    range(n_inps)] for n_inps, n_inp_cols in
                   zip(num_inputs, self.num_input_cols)]
        num_prods = [pow(n_inp_cols, n_inps) for n_inps, n_inp_cols in
                     zip(num_inputs, self.num_input_cols)]

        r = self._run_test(('PADDED\nInferenceType: MARGINAL\n'
                            'num_inputs: %s \nnum_prods: %s' % (num_inputs, num_prods)),
                           [] + ([Ops.product] if not self.without_product else [])
                           + ([Ops.products] if not self.without_products else [])
                           + [Ops.perm_products, Ops.products_layer],
                           inputs, num_inputs, indices,
                           inf_type=spn.InferenceType.MARGINAL, log=False)
        results.append(r)

        r = self._run_test(('PADDED\nInferenceType: MARGINAL-LOG\n'
                            'num_inputs: %s \nnum_prods: %s' % (num_inputs, num_prods)),
                           [] + ([Ops.product] if not self.without_product else [])
                           + ([Ops.products] if not self.without_products else [])
                           + [Ops.perm_products, Ops.products_layer],
                           inputs, num_inputs, indices,
                           inf_type=spn.InferenceType.MARGINAL, log=True)
        results.append(r)

        # Print results
        for res in results:
            res.print(self.file)

        if self.test_failed:
            print("\n ATLEAST ONE TEST FAILED!")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--num-input-rows', default=1000, type=int,
                        help="Num of rows of inputs")
    parser.add_argument('--num-input-cols', default=[5] * 10, type=list,
                        help="Num of cols of inputs")
    parser.add_argument('--num-batches', default=1, type=int,
                        help="Num of mini batches for ProductsLayer evaluation")
    parser.add_argument('--num-runs', default=50, type=int,
                        help="Number of times each test is run")
    parser.add_argument('--log-devices', action='store_true',
                        help="Log on which device op is run. Affects run time!")
    parser.add_argument('--without-cpu', action='store_true',
                        help="Do not run CPU tests")
    parser.add_argument('--without-gpu', action='store_true',
                        help="Do not run GPU tests")
    parser.add_argument('--without-product', action='store_true',
                        help="Do not run tests for Product")
    parser.add_argument('--without-products', action='store_true',
                        help="Do not run tests for Products")
    parser.add_argument('--without-indices', action='store_true',
                        help="Do not run test cases for PermProds with indices")
    parser.add_argument('--without-single-input', action='store_true',
                        help="Do not run test cases for ProductsLayer with single-input")
    parser.add_argument('--profile', default=False, action='store_true',
                        help="Run test one more time and profile")
    parser.add_argument('--profiles-dir', default='profiles', type=str,
                        help="Run test one more time and profile")
    parser.add_argument('--save-to', default='', type=str,
                        help="Save results to file")
    args = parser.parse_args()

    # Atleast two columns per input are needed for modelling multiple products
    # in PermuteProducts
    if not all(n_inp_cols >= 2 for n_inp_cols in args.num_input_cols):
        sys.exit('ERROR: All num_input_cols must be >= 2')

    # Open a file
    f = None
    if args.save_to:
        f = open(args.save_to, 'w')

    try:
        t = PerformanceTest(args.num_input_rows, args.num_input_cols, args.num_batches,
                            args.num_runs, args.without_cpu, args.without_gpu,
                            args.without_product, args.without_products,
                            args.without_indices, args.without_single_input,
                            args.log_devices, args.profile, args.profiles_dir, f)
        t.run()
    finally:
        if f is not None:
            f.close()


if __name__ == '__main__':
    main()
