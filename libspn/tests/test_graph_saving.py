#!/usr/bin/env python3

# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

import tensorflow as tf
from context import libspn as spn
import os
import numpy as np
import itertools


class TestGraphSaving(tf.test.TestCase):

    @classmethod
    def setUpClass(cls):
        super(TestGraphSaving, cls).setUpClass()
        cls.out_dir = os.path.realpath(os.path.join(os.getcwd(),
                                                    os.path.dirname(__file__),
                                                    "out"))

    @staticmethod
    def out_path(p):
        if isinstance(p, list):
            return [os.path.join(TestGraphSaving.out_dir, i) for i in p]
        else:
            return os.path.join(TestGraphSaving.out_dir, p)

    def test_withoutparams_initfixed(self):
        # Build an SPN
        model = spn.Poon11NaiveMixtureModel()
        root1 = model.build()

        # Save
        path = self.out_path("test_withoutparams_initfixed.spn")
        saver = spn.JSONSaver(path, pretty=True)
        saver.save(root1, save_param_vals=False)

        # Reset graph
        tf.reset_default_graph()

        # Load
        loader = spn.JSONLoader(path)
        root2 = loader.load()
        ivs2 = loader.find_node('IVs')
        init2 = spn.initialize_weights(root2)
        val_mpe2 = root2.get_value(inference_type=spn.InferenceType.MPE)
        val_marginal2 = root2.get_value(inference_type=spn.InferenceType.MARGINAL)

        # Check model after loading
        self.assertTrue(root2.is_valid())
        with tf.Session() as sess:
            init2.run()
            out_marginal2 = sess.run(val_marginal2, feed_dict={ivs2: model.feed})
            out_mpe2 = sess.run(val_mpe2, feed_dict={ivs2: model.feed})
        self.assertAlmostEqual(out_marginal2[np.all(model.feed >= 0, axis=1), :].sum(),
                               1.0, places=6)
        np.testing.assert_array_almost_equal(out_marginal2, model.true_values)
        np.testing.assert_array_almost_equal(out_mpe2, model.true_mpe_values)

    def test_withoutparams_initrandom(self):
        # Build an SPN
        feed = np.array(list(itertools.product(range(2), repeat=6)))
        model = spn.DiscreteDenseModel(
            num_classes=1, num_decomps=1, num_subsets=3,
            num_mixtures=2, weight_init_value=spn.ValueType.RANDOM_UNIFORM(0, 1))
        root1 = model.build(num_vars=6, num_vals=2)

        # Save
        path = self.out_path("test_withoutparams_initrandom.spn")
        saver = spn.JSONSaver(path, pretty=True)
        saver.save(root1, save_param_vals=False)

        # Reset graph
        tf.reset_default_graph()

        # Load
        loader = spn.JSONLoader(path)
        root2 = loader.load()
        ivs2 = loader.find_node('SampleIVs')
        init2 = spn.initialize_weights(root2)
        val_marginal2 = root2.get_value(inference_type=spn.InferenceType.MARGINAL)

        # Check model after loading
        self.assertTrue(root2.is_valid())
        with tf.Session() as sess:
            init2.run()
            out_marginal2 = sess.run(val_marginal2, feed_dict={ivs2: feed})
        self.assertAlmostEqual(out_marginal2.sum(), 1.0, places=6)

    def test_withparams_initfixed(self):
        # Build an SPN
        model = spn.Poon11NaiveMixtureModel()
        root1 = model.build()
        init1 = spn.initialize_weights(root1)

        with tf.Session() as sess:
            # Initialize
            init1.run()

            # Save
            path = self.out_path("test_withparams_initfixed.spn")
            saver = spn.JSONSaver(path, pretty=True)
            saver.save(root1, save_param_vals=True)

        # Reset graph
        tf.reset_default_graph()

        with tf.Session() as sess:
            # Load
            loader = spn.JSONLoader(path)
            root2 = loader.load(load_param_vals=True)
            ivs2 = loader.find_node('IVs')
            val_mpe2 = root2.get_value(inference_type=spn.InferenceType.MPE)
            val_marginal2 = root2.get_value(inference_type=spn.InferenceType.MARGINAL)

            # Check model after loading
            self.assertTrue(root2.is_valid())
            out_marginal2 = sess.run(val_marginal2, feed_dict={ivs2: model.feed})
            out_mpe2 = sess.run(val_mpe2, feed_dict={ivs2: model.feed})
            self.assertAlmostEqual(out_marginal2[np.all(model.feed >= 0, axis=1), :].sum(),
                                   1.0, places=6)
            np.testing.assert_array_almost_equal(out_marginal2, model.true_values)
            np.testing.assert_array_almost_equal(out_mpe2, model.true_mpe_values)

            # Writing log
            # writer = tf.summary.FileWriter(
            #     os.path.realpath(os.path.join(
            #         os.getcwd(), os.path.dirname(__file__),
            #         "logs", "test_graph_saving", "test_withparams_initfixed")),
            #     sess.graph)
            # writer.add_graph(sess.graph)
            # writer.close()

    def test_withparams_initrandom(self):
        # Build an SPN
        feed = np.array(list(itertools.product(range(2), repeat=6)))
        model = spn.DiscreteDenseModel(
            num_classes=1, num_decomps=1, num_subsets=3,
            num_mixtures=2, weight_init_value=spn.ValueType.RANDOM_UNIFORM(0, 1))
        root1 = model.build(num_vars=6, num_vals=2)
        init1 = spn.initialize_weights(root1)

        with tf.Session() as sess:
            # Initialize
            init1.run()

            # Save
            path = self.out_path("test_withparams_initrandom.spn")
            saver = spn.JSONSaver(path, pretty=True)
            saver.save(root1, save_param_vals=True)

        # Reset graph
        tf.reset_default_graph()

        with tf.Session() as sess:
            # Load
            loader = spn.JSONLoader(path)
            root2 = loader.load(load_param_vals=True)
            ivs2 = loader.find_node('SampleIVs')
            val_marginal2 = root2.get_value(inference_type=spn.InferenceType.MARGINAL)

            # Check model after loading
            self.assertTrue(root2.is_valid())
            out_marginal2 = sess.run(val_marginal2, feed_dict={ivs2: feed})
            self.assertAlmostEqual(out_marginal2.sum(), 1.0, places=6)

            # Writing log
            # writer = tf.summary.FileWriter(
            #     os.path.realpath(os.path.join(
            #         os.getcwd(), os.path.dirname(__file__),
            #         "logs", "test_graph_saving", "test_withparams_initrandom")),
            #     sess.graph)
            # writer.add_graph(sess.graph)
            # writer.close()


if __name__ == '__main__':
    tf.test.main()
