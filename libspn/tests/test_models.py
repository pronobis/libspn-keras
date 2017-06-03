#!/usr/bin/env python3

# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

import tensorflow as tf
from context import libspn as spn
import numpy as np
import itertools
import os

# spn.config_logger(spn.DEBUG2)


class TestModels(tf.test.TestCase):

    def generic_model_test(self, name, root, sample_ivs, class_ivs, write_log):
        # Generating weight initializers
        init = spn.initialize_weights(root)

        # Testing validity
        self.assertTrue(root.is_valid())

        # Generating value ops
        v = root.get_value()
        v_log = root.get_log_value()

        # Creating session
        with tf.Session() as sess:
            if write_log:
                # Writing log
                writer = tf.summary.FileWriter(
                    os.path.realpath(os.path.join(
                        os.getcwd(), os.path.dirname(__file__),
                        "logs", "test_model", name)),
                    sess.graph)
                writer.add_graph(sess.graph)
                writer.close()
            # Initializing weights
            init.run()
            # Computing all values
            feed_samples = list(itertools.product(range(2), repeat=6))
            if class_ivs is not None:
                feed_class = np.array([i for i in range(class_ivs.num_vals)
                                       for _ in range(len(feed_samples))]).reshape(-1, 1)
                feed_samples = np.array(feed_samples * class_ivs.num_vals)
                feed_dict = {sample_ivs: feed_samples, class_ivs: feed_class}
            else:
                feed_samples = np.array(feed_samples)
                feed_dict = {sample_ivs: feed_samples}
            out = sess.run(v, feed_dict=feed_dict)
            out_log = sess.run(tf.exp(v_log), feed_dict=feed_dict)

            # Test if partition function is 1.0
            self.assertAlmostEqual(out.sum(), 1.0, places=6)
            self.assertAlmostEqual(out_log.sum(), 1.0, places=6)

    def test_discretedense_1class_internalivs(self):
        model = spn.DiscreteDenseModel(
            num_classes=1,
            num_decomps=2,
            num_subsets=3,
            num_mixtures=2,
            input_dist=spn.DenseSPNGenerator.InputDist.MIXTURE,
            num_input_mixtures=None,
            weight_init_value=spn.ValueType.RANDOM_UNIFORM(0, 1))
        root = model.build(num_vars=6, num_vals=2)
        self.generic_model_test("1class",
                                root, model.sample_ivs, None,
                                write_log=True)

    def test_discretedense_3class_internalivs(self):
        model = spn.DiscreteDenseModel(
            num_classes=3,
            num_decomps=2,
            num_subsets=3,
            num_mixtures=2,
            input_dist=spn.DenseSPNGenerator.InputDist.MIXTURE,
            num_input_mixtures=None,
            weight_init_value=spn.ValueType.RANDOM_UNIFORM(0, 1))
        root = model.build(num_vars=6, num_vals=2)
        self.generic_model_test("3class",
                                root, model.sample_ivs, model.class_ivs,
                                write_log=True)

    def test_discretedense_1class_externalivs(self):
        model = spn.DiscreteDenseModel(
            num_classes=1,
            num_decomps=2,
            num_subsets=3,
            num_mixtures=2,
            input_dist=spn.DenseSPNGenerator.InputDist.MIXTURE,
            num_input_mixtures=None,
            weight_init_value=spn.ValueType.RANDOM_UNIFORM(0, 1))
        sample_ivs = spn.IVs(num_vars=6, num_vals=2)
        root = model.build(sample_ivs)
        self.generic_model_test("1class",
                                root, sample_ivs, None,
                                write_log=True)

    def test_discretedense_3class_externalivs(self):
        model = spn.DiscreteDenseModel(
            num_classes=3,
            num_decomps=2,
            num_subsets=3,
            num_mixtures=2,
            input_dist=spn.DenseSPNGenerator.InputDist.MIXTURE,
            num_input_mixtures=None,
            weight_init_value=spn.ValueType.RANDOM_UNIFORM(0, 1))
        sample_ivs = spn.IVs(num_vars=6, num_vals=2)
        class_ivs = spn.IVs(num_vars=1, num_vals=3)
        root = model.build(sample_ivs, class_input=class_ivs)
        self.generic_model_test("3class",
                                root, sample_ivs, class_ivs,
                                write_log=True)


if __name__ == '__main__':
    tf.test.main()
