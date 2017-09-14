#!/usr/bin/env python3

# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

from context import libspn as spn
from test import TestCase
import itertools
import tensorflow as tf
import numpy as np


def printc(string):
    COLOR = '\033[1m\033[93m'
    ENDC = '\033[0m'
    print(COLOR + string + ENDC)


class TestDenseSPNGeneratorMultiNodes(TestCase):

    def test_generte_set(self):
        """Generation of sets of inputs with __generate_set"""
        gen = spn.DenseSPNGeneratorMultiNodes(num_decomps=2,
                                              num_subsets=3,
                                              num_mixtures=2)
        v1 = spn.IVs(num_vars=2, num_vals=4)
        v2 = spn.ContVars(num_vars=3, name="ContVars1")
        v3 = spn.ContVars(num_vars=2, name="ContVars2")
        s1 = spn.Sum(v3)
        n1 = spn.Concat(v2)
        out = gen._DenseSPNGeneratorMultiNodes__generate_set([spn.Input(v1, [0, 3, 2, 6, 7]),
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
        gen = spn.DenseSPNGeneratorMultiNodes(num_decomps=2,
                                              num_subsets=3,
                                              num_mixtures=2)
        v1 = spn.IVs(num_vars=2, num_vals=4)
        v2 = spn.ContVars(num_vars=3, name="ContVars1")
        v3 = spn.ContVars(num_vars=2, name="ContVars2")
        s1 = spn.Sum(v3, v2)
        n1 = spn.Concat(v2)

        with self.assertRaises(spn.StructureError):
            gen._DenseSPNGeneratorMultiNodes__generate_set([spn.Input(v1, [0, 3, 2, 6, 7]),
                                                            spn.Input(v2, [1, 2]),
                                                            spn.Input(s1, None),
                                                            spn.Input(n1, None)])

    def tearDown(self):
        tf.reset_default_graph()

    def generic_dense_test(self, name, num_decomps, num_subsets, num_mixtures,
                           input_dist, num_input_mixtures, balanced, multi_nodes,
                           case):
        """A generic test for DenseSPNGeneratorMultiNodes."""
        self.tearDown()

        # Inputs
        v1 = spn.IVs(num_vars=3, num_vals=2, name="IVs1")
        v2 = spn.IVs(num_vars=3, num_vals=2, name="IVs2")

        gen = spn.DenseSPNGeneratorMultiNodes(num_decomps=num_decomps,
                                              num_subsets=num_subsets,
                                              num_mixtures=num_mixtures,
                                              input_dist=input_dist,
                                              balanced=balanced,
                                              num_input_mixtures=num_input_mixtures,
                                              multi_nodes=multi_nodes)

        # Generating SPN
        root = gen.generate(v1, v2)

        # Generating random weights
        with tf.name_scope("Weights"):
            spn.generate_weights(root, spn.ValueType.RANDOM_UNIFORM())

        # Generating weight initializers
        init = spn.initialize_weights(root)

        # Testing validity
        self.assertTrue(root.is_valid())

        # Generating value ops
        v = root.get_value()
        v_log = root.get_log_value()

        printc("Case: %s" % case)
        printc("- num_decomps: %s" % num_decomps)
        printc("- num_subsets: %s" % num_subsets)
        printc("- num_mixtures: %s" % num_mixtures)
        printc("- input_dist: %s" % ("MIXTURE" if input_dist ==
               spn.DenseSPNGeneratorMultiNodes.InputDist.MIXTURE else "RAW"))
        printc("- balanced: %s" % balanced)
        printc("- num_input_mixtures: %s" % num_input_mixtures)
        printc("- multi_nodes: %s" % multi_nodes)

        # Creating session
        with tf.Session() as sess:
            # Initializing weights
            init.run()
            # Computing all values
            feed = np.array(list(itertools.product(range(2), repeat=6)))
            feed_v1 = feed[:, :3]
            feed_v2 = feed[:, 3:]
            out = sess.run(v, feed_dict={v1: feed_v1, v2: feed_v2})
            out_log = sess.run(tf.exp(v_log), feed_dict={v1: feed_v1, v2: feed_v2})
            # Test if partition function is 1.0
            self.assertAlmostEqual(out.sum(), 1.0, places=6)
            self.assertAlmostEqual(out_log.sum(), 1.0, places=6)
            self.write_tf_graph(sess, self.sid(), self.cid())

    def test_generate_spn(self):
        """Generate and test dense SPNs with varying combination of parameters"""
        num_decomps = [1, 2]
        num_subsets = [2, 3, 6]
        num_mixtures = [1, 2]
        num_input_mixtures = [1, 2]
        input_dist = [spn.DenseSPNGeneratorMultiNodes.InputDist.MIXTURE,
                      spn.DenseSPNGeneratorMultiNodes.InputDist.RAW]
        balanced = [True, False]
        multi_nodes = [True, False]
        name = ["mixture", "raw"]
        case = 0

        for n_dec in num_decomps:
            for n_sub in num_subsets:
                for n_mix in num_mixtures:
                    for n_imix in num_input_mixtures:
                        for dist, n in zip(input_dist, name):
                            for bal in balanced:
                                for m_nodes in multi_nodes:
                                    case += 1
                                    self.generic_dense_test(name=n,
                                                            num_decomps=n_dec,
                                                            num_subsets=n_sub,
                                                            num_mixtures=n_mix,
                                                            input_dist=dist,
                                                            num_input_mixtures=n_imix,
                                                            balanced=bal,
                                                            multi_nodes=m_nodes,
                                                            case=case)


if __name__ == '__main__':
    tf.test.main()
