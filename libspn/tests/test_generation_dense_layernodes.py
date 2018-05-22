#!/usr/bin/env python3

# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

from context import libspn as spn
from libspn import conf
from test import TestCase
import itertools
import tensorflow as tf
import numpy as np


def printc(string):
    COLOR = '\033[1m\033[93m'
    ENDC = '\033[0m'
    print(COLOR + string + ENDC)


class TestDenseSPNGeneratorLayerNodes(TestCase):

    def test_generte_set(self):
        """Generation of sets of inputs with __generate_set"""
        gen = spn.DenseSPNGeneratorLayerNodes(num_decomps=2,
                                              num_subsets=3,
                                              num_mixtures=2)
        v1 = spn.IVs(num_vars=2, num_vals=4)
        v2 = spn.ContVars(num_vars=3, name="ContVars1")
        v3 = spn.ContVars(num_vars=2, name="ContVars2")
        s1 = spn.Sum(v3)
        n1 = spn.Concat(v2)
        out = gen._DenseSPNGeneratorLayerNodes__generate_set([spn.Input(v1, [0, 3, 2, 6, 7]),
                                                              spn.Input(v2, [1, 2]),
                                                              spn.Input(s1, None),
                                                              spn.Input(n1, None)])
        # Since order is undetermined, we check items
        self.assertEqual(len(out), 6)
        self.assertIn(tuple(sorted([(v2, 1), (n1, 1)])), out)
        self.assertIn(tuple(sorted([(v2, 2), (n1, 2)])), out)
        self.assertIn(tuple(sorted([(n1, 0)])), out)
        self.assertIn(tuple(sorted([(v1, 0), (v1, 2), (v1, 3)])), out)
        self.assertIn(tuple(sorted([(v1, 6), (v1, 7)])), out)
        self.assertIn(tuple(sorted([(s1, 0)])), out)

    def test_generte_set_errors(self):
        """Detecting structure errors in __generate_set"""
        gen = spn.DenseSPNGeneratorLayerNodes(num_decomps=2,
                                              num_subsets=3,
                                              num_mixtures=2)
        v1 = spn.IVs(num_vars=2, num_vals=4)
        v2 = spn.ContVars(num_vars=3, name="ContVars1")
        v3 = spn.ContVars(num_vars=2, name="ContVars2")
        s1 = spn.Sum(v3, v2)
        n1 = spn.Concat(v2)

        with self.assertRaises(spn.StructureError):
            gen._DenseSPNGeneratorLayerNodes__generate_set([spn.Input(v1, [0, 3, 2, 6, 7]),
                                                            spn.Input(v2, [1, 2]),
                                                            spn.Input(s1, None),
                                                            spn.Input(n1, None)])

    def tearDown(self):
        tf.reset_default_graph()

    def generic_dense_test(self, num_decomps, num_subsets, num_mixtures, input_dist,
                           num_input_mixtures, balanced, node_type, log_weights, case):
        """A generic test for DenseSPNGeneratorLayerNodes."""
        self.tearDown()

        def use_custom_ops(custom_ops=True):
            if custom_ops:
                conf.custom_gather_cols = True
                conf.custom_gather_cols_3d = True
                conf.custom_scatter_cols = True
                conf.custom_scatter_values = True
            else:
                conf.custom_gather_cols = False
                conf.custom_gather_cols_3d = False
                conf.custom_scatter_cols = False
                conf.custom_scatter_values = False

        printc("Case: %s" % case)
        printc("- num_decomps: %s" % num_decomps)
        printc("- num_subsets: %s" % num_subsets)
        printc("- num_mixtures: %s" % num_mixtures)
        printc("- input_dist: %s" % ("MIXTURE" if input_dist ==
               spn.DenseSPNGeneratorLayerNodes.InputDist.MIXTURE else "RAW"))
        printc("- balanced: %s" % balanced)
        printc("- num_input_mixtures: %s" % num_input_mixtures)
        printc("- node_type: %s" % ("SINGLE" if node_type ==
               spn.DenseSPNGeneratorLayerNodes.NodeType.SINGLE else "BLOCK" if
               node_type == spn.DenseSPNGeneratorLayerNodes.NodeType.BLOCK else
               "LAYER"))
        printc("- log_weights: %s" % log_weights)

        if log_weights:
            use_custom_ops(custom_ops=False)
        else:
            use_custom_ops(custom_ops=True)

        # Input parameters
        num_inputs = 2
        num_vars = 3
        num_vals = 2

        # Inputs
        inputs = [spn.IVs(num_vars=num_vars, num_vals=num_vals, name="IVs%s" % i)
                  for i in range(num_inputs)]

        gen = spn.DenseSPNGeneratorLayerNodes(num_decomps=num_decomps,
                                              num_subsets=num_subsets,
                                              num_mixtures=num_mixtures,
                                              input_dist=input_dist,
                                              balanced=balanced,
                                              num_input_mixtures=num_input_mixtures,
                                              node_type=node_type)

        # Generating SPN
        root = gen.generate(*inputs, root_name="root")

        # Generating random weights
        with tf.name_scope("Weights"):
            spn.generate_weights(root, spn.ValueType.RANDOM_UNIFORM(), log=log_weights)

        # Generating weight initializers
        init = spn.initialize_weights(root)

        # Testing validity of the SPN
        self.assertTrue(root.is_valid())

        # Generating value ops
        v = root.get_value()
        v_log = root.get_log_value()

        # Generating path ops
        mpe_path_gen = spn.MPEPath(log=False)
        mpe_path_gen_log = spn.MPEPath(log=True)
        mpe_path_gen.get_mpe_path(root)
        mpe_path_gen_log.get_mpe_path(root)
        path = [mpe_path_gen.counts[inp] for inp in inputs]
        path_log = [mpe_path_gen_log.counts[inp] for inp in inputs]

        if log_weights:
            # Collect all weight nodes in the SPN
            weight_nodes = []

            def fun(node):
                if node.is_param:
                    weight_nodes.append(node)
            spn.traverse_graph(root, fun=fun)

            # Generating gradient ops
            gradient_gen = spn.Gradient(log=True)
            gradient_gen.get_gradients(root)
            custom_gradients = [tf.reduce_sum(gradient_gen.gradients[weight], axis=0)
                                for weight in weight_nodes]
            tf_gradients = tf.gradients(v_log, [w.variable for w in weight_nodes])

        # Creating session
        with tf.Session() as sess:
            # Initializing weights
            init.run()

            # Generating random feed
            feed = np.array(list(itertools.product(range(num_vals),
                                                   repeat=(num_inputs*num_vars))))
            batch_size = feed.shape[0]
            feed_dict = {}
            for inp, f in zip(inputs, np.split(feed, num_inputs, axis=1)):
                feed_dict[inp] = f

            # Computing all values and paths
            out = sess.run(v, feed_dict=feed_dict)
            out_log = sess.run(tf.exp(v_log), feed_dict=feed_dict)
            out_path = sess.run(path, feed_dict=feed_dict)
            out_path_log = sess.run(path_log, feed_dict=feed_dict)
            # Compute gradients and assert
            if log_weights:
                out_custom_gradients = sess.run(custom_gradients, feed_dict=feed_dict)
                out_tf_gradients = sess.run(tf_gradients, feed_dict=feed_dict)
                # Assert custom gradients with tf gradients
                for out_cust_grad, out_tf_grad in zip(out_custom_gradients, out_tf_gradients):
                    if list(out_cust_grad.shape) != list(out_tf_grad.shape):
                        out_cust_grad = np.expand_dims(out_cust_grad, axis=0)
                    np.testing.assert_almost_equal(out_cust_grad, out_tf_grad, decimal=4)

            # Test if partition function is 1.0
            self.assertAlmostEqual(out.sum(), 1.0, places=6)
            self.assertAlmostEqual(out_log.sum(), 1.0, places=6)
            # Test if the sum of counts for each value of each variable
            # (6 variables, with 2 values each) = batch-size / num-vals
            self.assertEqual(np.sum(np.hstack(out_path), axis=0).tolist(),
                             [batch_size // num_vals]*num_inputs*num_vars*num_vals)
            self.assertEqual(np.sum(np.hstack(out_path_log), axis=0).tolist(),
                             [batch_size // num_vals]*num_inputs*num_vars*num_vals)
            self.write_tf_graph(sess, self.sid(), self.cid())

    def test_generate_spn(self):
        """Generate and test dense SPNs with varying combination of parameters"""
        num_decomps = [1, 2]
        num_subsets = [2, 3, 6]
        num_mixtures = [1, 2]
        num_input_mixtures = [1, 2]
        input_dist = [spn.DenseSPNGeneratorLayerNodes.InputDist.MIXTURE,
                      spn.DenseSPNGeneratorLayerNodes.InputDist.RAW]
        balanced = [True, False]
        node_type = [spn.DenseSPNGeneratorLayerNodes.NodeType.SINGLE,
                     spn.DenseSPNGeneratorLayerNodes.NodeType.BLOCK,
                     spn.DenseSPNGeneratorLayerNodes.NodeType.LAYER]
        log_weights = [True, False]
        case = 0

        for n_dec in num_decomps:
            for n_sub in num_subsets:
                for n_mix in num_mixtures:
                    for n_imix in num_input_mixtures:
                        for dist in input_dist:
                            for bal in balanced:
                                for n_type in node_type:
                                    for log_w in log_weights:
                                        case += 1
                                        self.generic_dense_test(num_decomps=n_dec,
                                                                num_subsets=n_sub,
                                                                num_mixtures=n_mix,
                                                                input_dist=dist,
                                                                num_input_mixtures=n_imix,
                                                                balanced=bal,
                                                                node_type=n_type,
                                                                log_weights=log_w,
                                                                case=case)


if __name__ == '__main__':
    tf.test.main()
