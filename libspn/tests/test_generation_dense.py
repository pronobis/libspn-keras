#!/usr/bin/env python3

from context import libspn as spn
from test import TestCase
import itertools
import tensorflow as tf
import numpy as np


class TestDenseSPNGenerator(TestCase):

    def test_generte_set(self):
        """Generation of sets of inputs with __generate_set"""
        gen = spn.DenseSPNGenerator(num_decomps=2,
                                    num_subsets=3,
                                    num_mixtures=2)
        v1 = spn.IndicatorLeaf(num_vars=2, num_vals=4)
        v2 = spn.RawLeaf(num_vars=3, name="RawLeaf1")
        v3 = spn.RawLeaf(num_vars=2, name="RawLeaf2")
        s1 = spn.Sum(v3)
        n1 = spn.Concat(v2)
        out = gen._DenseSPNGenerator__generate_set([spn.Input(v1, [0, 3, 2, 6, 7]),
                                                    spn.Input(v2, [1, 2]),
                                                    spn.Input(s1, None),
                                                    spn.Input(n1, None)])
        # scope_dict:
        # Scope({IndicatorLeaf(0x7f00cb4049b0):0}): {(IndicatorLeaf(0x7f00cb4049b0), 0),
        #                                  (IndicatorLeaf(0x7f00cb4049b0), 2),
        #                                  (IndicatorLeaf(0x7f00cb4049b0), 3)},
        # Scope({IndicatorLeaf(0x7f00cb4049b0):1}): {(IndicatorLeaf(0x7f00cb4049b0), 7),
        #                                  (IndicatorLeaf(0x7f00cb4049b0), 6)},
        # Scope({RawLeaf1(0x7f00b7982ef0):1}): {(Concat(0x7f00cb404d68), 1),
        #                                        (RawLeaf1(0x7f00b7982ef0), 1)},
        # Scope({RawLeaf1(0x7f00b7982ef0):2}): {(Concat(0x7f00cb404d68), 2),
        #                                        (RawLeaf1(0x7f00b7982ef0), 2)},
        # Scope({RawLeaf1(0x7f00b7982ef0):0}): {(Concat(0x7f00cb404d68), 0)},
        # Scope({RawLeaf2(0x7f00cb391eb8):0, RawLeaf2(0x7f00cb391eb8):1}): {
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
        v1 = spn.IndicatorLeaf(num_vars=2, num_vals=4)
        v2 = spn.RawLeaf(num_vars=3, name="RawLeaf1")
        v3 = spn.RawLeaf(num_vars=2, name="RawLeaf2")
        s1 = spn.Sum(v3, v2)
        n1 = spn.Concat(v2)

        with self.assertRaises(spn.StructureError):
            gen._DenseSPNGenerator__generate_set([spn.Input(v1, [0, 3, 2, 6, 7]),
                                                  spn.Input(v2, [1, 2]),
                                                  spn.Input(s1, None),
                                                  spn.Input(n1, None)])

    def generic_dense_test(self, name, num_decomps, num_subsets, num_mixtures,
                           input_dist, num_input_mixtures):
        """A generic test for DenseSPNGenerator."""
        v1 = spn.IndicatorLeaf(num_vars=3, num_vals=2, name="IndicatorLeaf1")
        v2 = spn.IndicatorLeaf(num_vars=3, num_vals=2, name="IndicatorLeaf2")

        gen = spn.DenseSPNGenerator(num_decomps=num_decomps,
                                    num_subsets=num_subsets,
                                    num_mixtures=num_mixtures,
                                    input_dist=input_dist,
                                    num_input_mixtures=num_input_mixtures)

        # Generating SPN
        root = gen.generate(v1, v2)

        # Generating random weights
        with tf.name_scope("Weights"):
            spn.generate_weights(root, tf.initializers.random_uniform(0.0, 1.0))

        # Generating weight initializers
        init = spn.initialize_weights(root)

        # Testing validity
        self.assertTrue(root.is_valid())

        # Generating value ops
        v = root.get_value()
        v_log = root.get_log_value()

        # Creating session
        with self.test_session() as sess:
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

    def test_generate_spn_mixture(self):
        """Generate a dense SPN with mixtures over inputs"""
        self.generic_dense_test(name="mixture",
                                num_decomps=2,
                                num_subsets=3,
                                num_mixtures=2,
                                input_dist=spn.DenseSPNGenerator.InputDist.MIXTURE,
                                num_input_mixtures=None)

    def test_generate_spn_raw(self):
        """Generate a dense SPN with raw inputs"""
        self.generic_dense_test(name="raw",
                                num_decomps=2,
                                num_subsets=3,
                                num_mixtures=2,
                                input_dist=spn.DenseSPNGenerator.InputDist.RAW,
                                num_input_mixtures=None)


if __name__ == '__main__':
    tf.test.main()
