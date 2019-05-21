#!/usr/bin/env python3

import libspn as spn
import itertools
import tensorflow as tf
import numpy as np
from libspn.tests.test import argsprod


def printc(string):
    COLOR = '\033[1m\033[93m'
    ENDC = '\033[0m'
    print(COLOR + string + ENDC)


class TestDenseSPNGenerator(tf.test.TestCase):

    @argsprod([1, 2], [2, 4], [1, 2], [1, 2], [[2, 2], [1, 1]],
              [spn.DenseSPNGenerator.InputDist.MIXTURE,
               spn.DenseSPNGenerator.InputDist.RAW],
              [True, False], [spn.DenseSPNGenerator.NodeType.SINGLE,
                              spn.DenseSPNGenerator.NodeType.BLOCK,
                              spn.DenseSPNGenerator.NodeType.LAYER],
              [True])
    def test_generate_spn(self, num_decomps, num_subsets, num_mixtures, num_input_mixtures,
                          input_dims, input_dist, balanced, node_type, log_weights):
        """A generic test for DenseSPNGenerator."""

        if input_dist == spn.DenseSPNGenerator.InputDist.RAW \
            and num_input_mixtures != 1:
            # Redundant test case, so just return
            return

        # Input parameters
        num_inputs = input_dims[0]
        num_vars = input_dims[1]
        num_vals = 2

        printc("\n- num_inputs: %s" % num_inputs)
        printc("- num_vars: %s" % num_vars)
        printc("- num_vals: %s" % num_vals)
        printc("- num_decomps: %s" % num_decomps)
        printc("- num_subsets: %s" % num_subsets)
        printc("- num_mixtures: %s" % num_mixtures)
        printc("- input_dist: %s" % ("MIXTURE" if input_dist ==
               spn.DenseSPNGenerator.InputDist.MIXTURE else "RAW"))
        printc("- balanced: %s" % balanced)
        printc("- num_input_mixtures: %s" % num_input_mixtures)
        printc("- node_type: %s" % ("SINGLE" if node_type ==
               spn.DenseSPNGenerator.NodeType.SINGLE else "BLOCK" if
               node_type == spn.DenseSPNGenerator.NodeType.BLOCK else
               "LAYER"))
        printc("- log_weights: %s" % log_weights)

        # Inputs
        inputs = [spn.IndicatorLeaf(num_vars=num_vars, num_vals=num_vals, name=("IndicatorLeaf_%d" % (i+1)))
                  for i in range(num_inputs)]

        gen = spn.DenseSPNGenerator(num_decomps=num_decomps,
                                    num_subsets=num_subsets,
                                    num_mixtures=num_mixtures,
                                    input_dist=input_dist,
                                    balanced=balanced,
                                    num_input_mixtures=num_input_mixtures,
                                    node_type=node_type)

        # Generate Sub-SPNs
        sub_spns = [gen.generate(*inputs, root_name=("sub_root_%d" % (i+1)))
                    for i in range(3)]

        # Generate random weights for the first sub-SPN
        with tf.name_scope("Weights"):
            spn.generate_weights(sub_spns[0], tf.initializers.random_uniform(0.0, 1.0),
                                 log=log_weights)

        # Initialize weights of the first sub-SPN
        sub_spn_init = spn.initialize_weights(sub_spns[0])

        # Testing validity of the first sub-SPN
        self.assertTrue(sub_spns[0].is_valid())

        # Generate value ops of the first sub-SPN
        sub_spn_v = sub_spns[0].get_value()
        sub_spn_v_log = sub_spns[0].get_log_value()

        # Generate path ops of the first sub-SPN
        sub_spn_mpe_path_gen = spn.MPEPath(log=False)
        sub_spn_mpe_path_gen_log = spn.MPEPath(log=True)
        sub_spn_mpe_path_gen.get_mpe_path(sub_spns[0])
        sub_spn_mpe_path_gen_log.get_mpe_path(sub_spns[0])
        sub_spn_path = [sub_spn_mpe_path_gen.counts[inp] for inp in inputs]
        sub_spn_path_log = [sub_spn_mpe_path_gen_log.counts[inp] for inp in inputs]

        # Collect all weight nodes of the first sub-SPN
        sub_spn_weight_nodes = []

        def fun(node):
            if node.is_param:
                sub_spn_weight_nodes.append(node)
        spn.traverse_graph(sub_spns[0], fun=fun)

        # Generate an upper-SPN over sub-SPNs
        products_lower = []
        for sub_spn in sub_spns:
            products_lower.append([v.node for v in sub_spn.values])

        num_top_mixtures = [2, 1, 3]
        sums_lower = []
        for prods, num_top_mix in zip(products_lower, num_top_mixtures):
            if node_type == spn.DenseSPNGenerator.NodeType.SINGLE:
                sums_lower.append([spn.Sum(*prods) for _ in range(num_top_mix)])
            elif node_type == spn.DenseSPNGenerator.NodeType.BLOCK:
                sums_lower.append([spn.ParallelSums(*prods, num_sums=num_top_mix)])
            else:
                sums_lower.append([spn.SumsLayer(*prods * num_top_mix,
                                                 num_or_size_sums=num_top_mix)])

        # Generate upper-SPN
        root = gen.generate(*list(itertools.chain(*sums_lower)), root_name="root")

        # Generate random weights for the SPN
        with tf.name_scope("Weights"):
            spn.generate_weights(root, tf.initializers.random_uniform(0.0, 1.0),
                                 log=log_weights)

        # Initialize weight of the SPN
        spn_init = spn.initialize_weights(root)

        # Testing validity of the SPN
        self.assertTrue(root.is_valid())

        # Generate value ops of the SPN
        spn_v = root.get_value()
        spn_v_log = root.get_log_value()

        # Generate path ops of the SPN
        spn_mpe_path_gen = spn.MPEPath(log=False)
        spn_mpe_path_gen_log = spn.MPEPath(log=True)
        spn_mpe_path_gen.get_mpe_path(root)
        spn_mpe_path_gen_log.get_mpe_path(root)
        spn_path = [spn_mpe_path_gen.counts[inp] for inp in inputs]
        spn_path_log = [spn_mpe_path_gen_log.counts[inp] for inp in inputs]

        # Collect all weight nodes in the SPN
        spn_weight_nodes = []

        def fun(node):
            if node.is_param:
                spn_weight_nodes.append(node)
        spn.traverse_graph(root, fun=fun)

        # Create a session
        with self.test_session() as sess:
            # Initializing weights
            sess.run(sub_spn_init)
            sess.run(spn_init)

            # Generate input feed
            feed = np.array(list(itertools.product(range(num_vals),
                                                   repeat=(num_inputs*num_vars))))
            batch_size = feed.shape[0]
            feed_dict = {}
            for inp, f in zip(inputs, np.split(feed, num_inputs, axis=1)):
                feed_dict[inp] = f

            # Compute all values and paths of sub-SPN
            sub_spn_out = sess.run(sub_spn_v, feed_dict=feed_dict)
            sub_spn_out_log = sess.run(tf.exp(sub_spn_v_log), feed_dict=feed_dict)
            sub_spn_out_path = sess.run(sub_spn_path, feed_dict=feed_dict)
            sub_spn_out_path_log = sess.run(sub_spn_path_log, feed_dict=feed_dict)

            # Compute all values and paths of the complete SPN
            spn_out = sess.run(spn_v, feed_dict=feed_dict)
            spn_out_log = sess.run(tf.exp(spn_v_log), feed_dict=feed_dict)
            spn_out_path = sess.run(spn_path, feed_dict=feed_dict)
            spn_out_path_log = sess.run(spn_path_log, feed_dict=feed_dict)

            # Test if partition function of the sub-SPN and of the
            # complete SPN is 1.0
            self.assertAlmostEqual(sub_spn_out.sum(), 1.0, places=6)
            self.assertAlmostEqual(sub_spn_out_log.sum(), 1.0, places=6)
            self.assertAlmostEqual(spn_out.sum(), 1.0, places=6)
            self.assertAlmostEqual(spn_out_log.sum(), 1.0, places=6)

            # Test if the sum of counts for each value of each variable
            # (6 variables, with 2 values each) = batch-size / num-vals
            self.assertEqual(np.sum(np.hstack(sub_spn_out_path), axis=0).tolist(),
                             [batch_size // num_vals]*num_inputs*num_vars*num_vals)
            self.assertEqual(np.sum(np.hstack(sub_spn_out_path_log), axis=0).tolist(),
                             [batch_size // num_vals]*num_inputs*num_vars*num_vals)
            self.assertEqual(np.sum(np.hstack(spn_out_path), axis=0).tolist(),
                             [batch_size // num_vals] * num_inputs * num_vars * num_vals)
            self.assertEqual(np.sum(np.hstack(spn_out_path_log), axis=0).tolist(),
                             [batch_size // num_vals] * num_inputs * num_vars * num_vals)


if __name__ == '__main__':
    tf.test.main()
