#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
from context import libspn as spn
import time
import argparse
import colorama as col
import scipy as scp
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

    def mnist_01(inputs, num_decomps, num_subsets, num_mixtures, num_input_mixtures,
                 balanced, input_dist, node_type, inf_type, log=False):

        # Learning Parameters
        additive_smoothing = 100
        min_additive_smoothing = 1

        # Weight initialization
        weight_init_value = tf.initializers.random_uniform(10, 11)

        # Generate SPN structure
        dense_gen = spn.DenseSPNGenerator(num_decomps=num_decomps,
                                                    num_subsets=num_subsets,
                                                    num_mixtures=num_mixtures,
                                                    input_dist=(spn.DenseSPNGenerator.
                                                                InputDist.RAW if input_dist is
                                                                "RAW" else spn.
                                                                DenseSPNGenerator.
                                                                InputDist.MIXTURE),
                                                    num_input_mixtures=num_input_mixtures,
                                                    balanced=balanced,
                                                    node_type=node_type)
        root0 = dense_gen.generate(inputs, root_name="root_0")
        root1 = dense_gen.generate(inputs, root_name="root_1")
        root = spn.Sum(root0, root1, name="root")
        spn.generate_weights(root, initializer=weight_init_value)
        latent = root.generate_latent_indicators()

        # Add EM Learning
        additive_smoothing_var = tf.Variable(additive_smoothing, dtype=spn.conf.dtype)
        learning = spn.HardEMLearning(root, log=log, value_inference_type=inf_type,
                                  additive_smoothing=additive_smoothing_var)

        return root, latent, learning, additive_smoothing, min_additive_smoothing, \
            additive_smoothing_var

    def mnist_all(inputs, num_decomps, num_subsets, num_mixtures, num_input_mixtures,
                  balanced, input_dist, node_type, inf_type, log=False):

        # Learning Parameters
        additive_smoothing = 0
        min_additive_smoothing = 0
        initial_accum_value = 20

        # Weight initialization
        weight_init_value = tf.initializers.random_uniform(0, 1)

        # Add random values before max
        use_unweighted = True

        # Generate SPN structure
        dense_gen = spn.DenseSPNGenerator(num_decomps=num_decomps,
                                                    num_subsets=num_subsets,
                                                    num_mixtures=num_mixtures,
                                                    input_dist=(spn.DenseSPNGenerator.
                                                                InputDist.RAW if input_dist is
                                                                "RAW" else spn.
                                                                DenseSPNGenerator.
                                                                InputDist.MIXTURE),
                                                    num_input_mixtures=num_input_mixtures,
                                                    balanced=balanced,
                                                    node_type=node_type)
        class_roots = [dense_gen.generate(inputs, root_name=("Class_%d" % i))
                       for i in range(10)]
        root = spn.Sum(*class_roots, name="root")
        spn.generate_weights(root, intializer=weight_init_value)
        latent = root.generate_latent_indicators()

        # Add EM Learning
        additive_smoothing_var = tf.Variable(additive_smoothing, dtype=spn.conf.dtype)
        learning = spn.HardEMLearning(root, log=log, value_inference_type=inf_type,
                                  additive_smoothing=additive_smoothing_var,
                                  initial_accum_value=initial_accum_value,
                                  use_unweighted=use_unweighted)

        return root, latent, learning, additive_smoothing, min_additive_smoothing, \
            additive_smoothing_var


class OpTestResult:
    """Result of a single test of a single op."""

    def __init__(self, op_name, on_gpu, node_type, spn_size, tf_size, memory_used,
                 input_dist, setup_time, weights_init_time, run_times, test_accuracy):
        self.op_name = op_name
        self.on_gpu = on_gpu
        self.node_type = node_type
        self.spn_size = spn_size
        self.tf_size = tf_size
        self.memory_used = memory_used
        self.input_dist = input_dist
        self.setup_time = setup_time
        self.weights_init_time = weights_init_time
        self.run_times = run_times
        self.test_accuracy = test_accuracy


class TestResults:
    """Results for a single test for multiple ops and devices."""

    def __init__(self, test_name, cpu_results, gpu_results):
        self.test_name = test_name
        self.cpu_results = cpu_results
        self.gpu_results = gpu_results

    def print(self, file):
        def get_header(dev):
            return ("%4s %11s %13s %9s %8s %9s %11s %11s %17s %15s %14s %11s" %
                    (dev, 'op', 'node_type', 'SPN_size', 'TF_size', 'mem_used',
                     'input_dist', 'setup_time', 'weights_init_time', 'first_run_time',
                     'rest_run_time', 'test_accuracy'))

        def get_res(res):
            """Helper function printing a single result."""
            return ("%16s %11s %7d %10d %11.4f %11s %10.4f %12.4f %15.4f %15.4f %14.4f" %
                    (res.op_name, res.node_type, res.spn_size, res.tf_size,
                     (0.0 if res.memory_used is None else res.memory_used / 1000000),
                     res.input_dist, res.setup_time, res.weights_init_time,
                     res.run_times[0], np.mean(res.run_times[1:]),
                     res.test_accuracy * 100))

        # Print results
        print1("\n-----------------------", file)
        print1("%s" % self.test_name, file)
        print1("-----------------------", file)
        print1(get_header("CPU"), file)
        for res in sorted(self.cpu_results, key=lambda x: len(x.op_name)):
            print1(get_res(res), file, (red if res.input_dist is "RAW" else green))
        print1(get_header("GPU"), file)
        for res in sorted(self.gpu_results, key=lambda x: len(x.op_name)):
            print1(get_res(res), file, (red if res.input_dist is "RAW" else green))


class PerformanceTest:

    def __init__(self, num_decomps, num_subsets, num_mixtures, num_input_mixtures,
                 balanced, num_epochs, without_cpu, without_gpu, log_devs, profile,
                 profiles_dir, file):
        self.num_decomps = num_decomps
        self.num_subsets = num_subsets
        self.num_mixtures = num_mixtures
        self.num_input_mixtures = num_input_mixtures
        self.balanced = balanced
        self.num_epochs = num_epochs
        self.without_cpu = without_cpu
        self.without_gpu = without_gpu
        self.log_devs = log_devs
        self.profile = profile
        self.profiles_dir = profiles_dir
        self.file = file
        self.test_failed = False

        print1("Params:", file)
        print1("- num_decomps=%s" % num_decomps, file)
        print1("- num_subsets=%s" % num_subsets, file)
        print1("- num_mixtures=%s" % num_mixtures, file)
        print1("- num_input_mixtures=%s" % num_input_mixtures, file)
        print1("- balanced=%s" % balanced, file)
        print1("- num_epochs=%s" % num_epochs, file)
        print1("", file=file)

    def _data_set(self, op_fun):
        # TRAINING SET
        datasets = tf.contrib.learn.datasets.mnist.read_data_sets("mnist")

        # Process data
        def process_set(data):
            threshold = 20
            images = np.reshape(data, (-1, 28, 28))
            resized = []
            for i in range(images.shape[0]):
                resized.append((scp.misc.imresize(images[i, :, :], 0.5, interp='nearest').ravel()
                               > threshold).astype(dtype=int))
            images = np.vstack(resized)
            return images

        train_images = process_set(datasets.train.images)
        test_images = process_set(datasets.test.images)
        train_labels = datasets.train.labels
        test_labels = datasets.test.labels

        if op_fun is Ops.mnist_01:
            train_images_0 = train_images[train_labels == 0]
            train_images_1 = train_images[train_labels == 1]
            train_set = np.concatenate([train_images_0, train_images_1], 0)
            train_labels = np.concatenate([np.ones((train_images_0.shape[0]))*0,
                                           np.ones((train_images_1.shape[0]))*1])

            test_images_0 = test_images[test_labels == 0]
            test_images_1 = test_images[test_labels == 1]
            test_set = np.concatenate([test_images_0, test_images_1], 0)
            test_labels = np.concatenate([np.ones((test_images_0.shape[0]))*0,
                                          np.ones((test_images_1.shape[0]))*1])
        elif op_fun is Ops.mnist_all:
            train_set = train_images
            test_set = test_images

        train_labels = np.reshape(train_labels, (-1, 1))
        test_labels = np.reshape(test_labels, (-1, 1))

        return train_set, train_labels, test_set, test_labels

    def _run_op_test(self, op_fun, input_dist='RAW', node_type=None,
                     inf_type=spn.InferenceType.MARGINAL, log=False, on_gpu=True):
        """Run a single test for a single op."""
        # Preparations
        op_name = op_fun.__name__
        device_name = '/gpu:0' if on_gpu else '/cpu:0'

        # Print
        print2("--> %s: on_gpu=%s, input_dist=%s, inference=%s, node_type=%s, log=%s"
               % (op_name, on_gpu, input_dist, ("MPE" if inf_type == spn.InferenceType.MPE
                  else "MARGINAL"), ("SINGLE" if node_type == spn.DenseSPNGenerator.
                  NodeType.SINGLE else "BLOCK" if node_type == spn.DenseSPNGenerator.
                  NodeType.BLOCK else "LAYER"), log), self.file)

        train_set, train_labels, test_set, test_labels = self._data_set(op_fun)

        # Create graph
        tf.reset_default_graph()
        with tf.device(device_name):
            # Create input latent_indicators
            inputs_pl = spn.IndicatorLeaf(num_vars=196, num_vals=2)
            # Create dense SPN and generate TF graph for training
            start_time = time.time()
            # Generate SPN
            root, latent, learning, additive_smoothing, min_additive_smoothing, \
                additive_smoothing_var = op_fun(inputs_pl, self.num_decomps,
                                                self.num_subsets, self.num_mixtures,
                                                self.num_input_mixtures, self.balanced,
                                                input_dist, node_type, inf_type, log)
            # Add Learning Ops
            init_weights = spn.initialize_weights(root)
            reset_accumulators = learning.reset_accumulators()
            accumulate_updates = learning.accumulate_updates()
            update_spn = learning.update_spn()

            # Generate Testing Ops
            mpe_state_gen = spn.MPEState(log=log, value_inference_type=spn.InferenceType.MPE)
            mpe_latent_indicators, mpe_latent = mpe_state_gen.get_state(root, inputs_pl, latent)
            setup_time = time.time() - start_time

            if on_gpu:
                max_bytes_used_op = tf.contrib.memory_stats.MaxBytesInUse()
        # Get num of SPN ops
        spn_size = root.get_num_nodes()
        # Get num of graph ops
        tf_size = len(tf.get_default_graph().get_operations())

        # Smoothing Decay for Additive Smoothing
        smoothing_decay = 0.2

        # Run op multiple times
        with tf.Session(config=tf.ConfigProto(
                allow_soft_placement=False,
                log_device_placement=self.log_devs)) as sess:
            # Initialize weights of the SPN
            start_time = time.time()
            init_weights.run()
            weights_init_time = time.time() - start_time

            # Reset accumulators
            sess.run(reset_accumulators)

            run_times = []
            # Create feed dictionary
            feed = {inputs_pl: train_set, latent: train_labels}
            # Run Training
            for epoch in range(self.num_epochs):
                start_time = time.time()
                # Adjust smoothing
                ads = max(np.exp(-epoch*smoothing_decay)*additive_smoothing,
                          min_additive_smoothing)
                sess.run(additive_smoothing_var.assign(ads))
                # Run accumulate_updates
                sess.run(accumulate_updates, feed_dict=feed)
                # Update weights
                sess.run(update_spn)
                # Reset accumulators
                sess.run(reset_accumulators)
                run_times.append(time.time() - start_time)

            if on_gpu:
                memory_used = sess.run(max_bytes_used_op)
            else:
                memory_used = None

            if self.profile:
                # Add additional options to trace the session execution
                options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata_acc_updt = tf.RunMetadata()
                run_metadata_spn_updt = tf.RunMetadata()
                run_metadata_acc_rst = tf.RunMetadata()

                # Run a single epoch
                # Run accumulate_updates
                sess.run(accumulate_updates, feed_dict=feed, options=options,
                         run_metadata=run_metadata_acc_updt)
                # Update weights
                sess.run(update_spn, options=options, run_metadata=run_metadata_spn_updt)
                # Reset accumulators
                sess.run(reset_accumulators, options=options, run_metadata=run_metadata_acc_rst)

                # Create the Timeline object, and write it to a json file
                fetched_timeline_acc_updt = timeline.Timeline(run_metadata_acc_updt.step_stats)
                fetched_timeline_spn_updt = timeline.Timeline(run_metadata_spn_updt.step_stats)
                fetched_timeline_acc_rst = timeline.Timeline(run_metadata_acc_rst.step_stats)

                chrome_trace_acc_updt = fetched_timeline_acc_updt.generate_chrome_trace_format()
                chrome_trace_spn_updt = fetched_timeline_spn_updt.generate_chrome_trace_format()
                chrome_trace_acc_rst = fetched_timeline_acc_rst.generate_chrome_trace_format()

                if not os.path.exists(self.profiles_dir):
                    os.makedirs(self.profiles_dir)

                file_name = op_name
                file_name += ("_GPU_" if on_gpu else "_CPU_")
                file_name += input_dist  # "RAW" or "MIXTURE"
                file_name += ("_ SINGLE" if node_type ==
                              spn.DenseSPNGenerator.NodeType.SINGLE else
                              "_BLOCK" if node_type ==
                              spn.DenseSPNGenerator.NodeType.BLOCK else
                              "_LAYER")
                file_name += ("_MPE-LOG" if log else "_MPE") if inf_type == \
                    spn.InferenceType.MPE else ("_MARGINAL-LOG" if log else
                                                "_MARGINAL")

                with open('%s/timeline_%s_acc_updt.json' % (self.profiles_dir,
                          file_name), 'w') as f:
                    f.write(chrome_trace_acc_updt)
                with open('%s/timeline_%s_spn_updt.json' % (self.profiles_dir,
                          file_name), 'w') as f:
                    f.write(chrome_trace_spn_updt)
                with open('%s/timeline_%s_acc_rst.json' % (self.profiles_dir,
                          file_name), 'w') as f:
                    f.write(chrome_trace_acc_rst)

            # Run Testing
            mpe_latent_val = sess.run([mpe_latent], feed_dict={inputs_pl: test_set,
                                      latent: np.ones((test_set.shape[0], 1))*-1})
            result = (mpe_latent_val == test_labels)
            test_accuracy = np.sum(result) / test_labels.size

        # Return stats
        return OpTestResult(op_name, on_gpu, ("SINGLE" if node_type == spn.
                            DenseSPNGenerator.NodeType.SINGLE else "BLOCK"
                            if node_type == spn.DenseSPNGenerator.NodeType.BLOCK
                            else "LAYER"), spn_size, tf_size, memory_used, input_dist,
                            setup_time, weights_init_time, run_times, test_accuracy)

    def _run_test(self, test_name, op_funs, node_types, inf_type, log):
        """Run a single test for multiple ops and devices."""
        cpu_results = []
        gpu_results = []
        for op_fun in op_funs:
            for n_type in node_types:
                if not self.without_cpu:
                    cpu_results.append(
                        self._run_op_test(op_fun, input_dist="RAW",
                                          node_type=n_type, inf_type=inf_type,
                                          log=log, on_gpu=False))
                    cpu_results.append(
                        self._run_op_test(op_fun, input_dist="MIXTURE",
                                          node_type=n_type, inf_type=inf_type,
                                          log=log, on_gpu=False))
                if not self.without_gpu:
                    gpu_results.append(
                        self._run_op_test(op_fun, input_dist="RAW",
                                          node_type=n_type, inf_type=inf_type,
                                          log=log, on_gpu=True))
                    gpu_results.append(
                        self._run_op_test(op_fun, input_dist="MIXTURE",
                                          node_type=n_type, inf_type=inf_type,
                                          log=log, on_gpu=True))
        return TestResults(test_name, cpu_results, gpu_results)

    def run(self):
        """Run all tests."""
        print1("Running tests:", self.file)
        results = []

        r = self._run_test('InferenceType: MARGINAL-LOG',
                           [Ops.mnist_01, Ops.mnist_all],
                           [spn.DenseSPNGenerator.NodeType.SINGLE,
                            spn.DenseSPNGenerator.NodeType.BLOCK,
                            spn.DenseSPNGenerator.NodeType.LAYER],
                           inf_type=spn.InferenceType.MARGINAL, log=True)
        results.append(r)

        r = self._run_test('InferenceType: MPE-LOG',
                           [Ops.mnist_01, Ops.mnist_all],
                           [spn.DenseSPNGenerator.NodeType.SINGLE,
                            spn.DenseSPNGenerator.NodeType.BLOCK,
                            spn.DenseSPNGenerator.NodeType.LAYER],
                           inf_type=spn.InferenceType.MPE, log=True)
        results.append(r)

        # Print results
        for res in results:
            res.print(self.file)

        if self.test_failed:
            print("\n ATLEAST ONE TEST FAILED!")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--num-decomps', default=1, type=int,
                        help="Num of decompositions at each level")
    parser.add_argument('--num-subsets', default=2, type=int,
                        help="Num of subsets in each desomposition")
    parser.add_argument('--num-mixtures', default=3, type=int,
                        help="Num of mixtures for each subset")
    parser.add_argument('--num-input-mixtures', default=2, type=int,
                        help="Num of input mixtures")
    parser.add_argument('--balanced', default=True, action='store_true',
                        help="Generated dense SPN is balanced between decompositions")
    parser.add_argument('--num-epochs', default=25, type=int,
                        help="Num of epochs to train the network")
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
        t = PerformanceTest(args.num_decomps, args.num_subsets, args.num_mixtures,
                            args.num_input_mixtures, args.balanced, args.num_epochs,
                            args.without_cpu, args.without_gpu, args.log_devices,
                            args.profile, args.profiles_dir, f)
        t.run()
    finally:
        if f is not None:
            f.close()


if __name__ == '__main__':
    main()
