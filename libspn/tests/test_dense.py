#!/usr/bin/env python3

# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

import unittest
import itertools
import os
import tensorflow as tf
import numpy as np
from context import libspn as spn


class TestDenseSPNGenerator(unittest.TestCase):

    def tearDown(self):
        tf.reset_default_graph()

    def test_generte_set(self):
        """Generation of sets of inputs with __generate_set"""
        gen = spn.DenseSPNGenerator(num_decomps=2,
                                    num_subsets=3,
                                    num_mixtures=2)
        v1 = spn.IVs(num_vars=2, num_vals=4)
        v2 = spn.ContVars(num_vars=3, name="ContVars1")
        v3 = spn.ContVars(num_vars=2, name="ContVars2")
        s1 = spn.Sum(v3)
        n1 = spn.Concat(v2)
        out = gen._DenseSPNGenerator__generate_set([spn.Input(v1, [0, 3, 2, 6, 7]),
                                                    spn.Input(v2, [1, 2]),
                                                    spn.Input(s1, None),
                                                    spn.Input(n1, None)])
        # scope_dict:
        # Scope({IVs(0x7f00cb4049b0):0}): {(IVs(0x7f00cb4049b0), 0),
        #                                  (IVs(0x7f00cb4049b0), 2),
        #                                  (IVs(0x7f00cb4049b0), 3)},
        # Scope({IVs(0x7f00cb4049b0):1}): {(IVs(0x7f00cb4049b0), 7),
        #                                  (IVs(0x7f00cb4049b0), 6)},
        # Scope({ContVars1(0x7f00b7982ef0):1}): {(Concat(0x7f00cb404d68), 1),
        #                                        (ContVars1(0x7f00b7982ef0), 1)},
        # Scope({ContVars1(0x7f00b7982ef0):2}): {(Concat(0x7f00cb404d68), 2),
        #                                        (ContVars1(0x7f00b7982ef0), 2)},
        # Scope({ContVars1(0x7f00b7982ef0):0}): {(Concat(0x7f00cb404d68), 0)},
        # Scope({ContVars2(0x7f00cb391eb8):0, ContVars2(0x7f00cb391eb8):1}): {
        #                                         (Sum(0x7f00cb404a90), 0)}}

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
        gen = spn.DenseSPNGenerator(num_decomps=2,
                                    num_subsets=3,
                                    num_mixtures=2)
        v1 = spn.IVs(num_vars=2, num_vals=4)
        v2 = spn.ContVars(num_vars=3, name="ContVars1")
        v3 = spn.ContVars(num_vars=2, name="ContVars2")
        s1 = spn.Sum(v3, v2)
        n1 = spn.Concat(v2)

        with self.assertRaises(spn.StructureError):
            gen._DenseSPNGenerator__generate_set([spn.Input(v1, [0, 3, 2, 6, 7]),
                                                  spn.Input(v2, [1, 2]),
                                                  spn.Input(s1, None),
                                                  spn.Input(n1, None)])

    def generic_dense_test(self, name, num_decomps, num_subsets, num_mixtures,
                           input_dist, num_input_mixtures, write_log):
        """A generic test for DenseSPNGenerator."""
        v1 = spn.IVs(num_vars=3, num_vals=2, name="IVs1")
        v2 = spn.IVs(num_vars=3, num_vals=2, name="IVs2")

        gen = spn.DenseSPNGenerator(num_decomps=num_decomps,
                                    num_subsets=num_subsets,
                                    num_mixtures=num_mixtures,
                                    input_dist=input_dist,
                                    num_input_mixtures=num_input_mixtures)

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
            if write_log:
                # Writing log
                writer = tf.train.SummaryWriter(
                    os.path.realpath(os.path.join(
                        os.getcwd(), os.path.dirname(__file__),
                        "logs", "test_dense", name)),
                    sess.graph)
                writer.add_graph(sess.graph)
                writer.close()

    def test_generate_spn_mixture(self):
        """Generate a dense SPN with mixtures over inputs"""
        self.generic_dense_test(name="mixture",
                                num_decomps=2,
                                num_subsets=3,
                                num_mixtures=2,
                                input_dist=spn.DenseSPNGenerator.InputDist.MIXTURE,
                                num_input_mixtures=None,
                                write_log=False)

    def test_generate_spn_raw(self):
        """Generate a dense SPN with raw inputs"""
        self.generic_dense_test(name="raw",
                                num_decomps=2,
                                num_subsets=3,
                                num_mixtures=2,
                                input_dist=spn.DenseSPNGenerator.InputDist.RAW,
                                num_input_mixtures=None,
                                write_log=False)


if __name__ == '__main__':
    unittest.main()
