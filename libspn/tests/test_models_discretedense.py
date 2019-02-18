#!/usr/bin/env python3

# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

from context import libspn as spn
from test import TestCase
import tensorflow as tf
import numpy as np
import itertools

# spn.config_logger(spn.DEBUG2)


class TestModelsDiscreteDense(TestCase):

    def generic_model_test(self, name, root, sample_ivs, class_ivs):
        # Generating weight initializers
        init = spn.initialize_weights(root)

        # Testing validity
        self.assertTrue(root.is_valid())

        # Generating value ops
        v = root.get_value()
        v_log = root.get_log_value()

        # Creating session
        with tf.Session() as sess:
            self.write_tf_graph(sess, self.sid(), self.cid())
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
            weight_init_value=tf.initializers.random_uniform(0.0, 1.0)(0, 1))
        root = model.build(num_vars=6, num_vals=2)
        self.generic_model_test("1class",
                                root, model.sample_ivs, None)

    def test_discretedense_3class_internalivs(self):
        model = spn.DiscreteDenseModel(
            num_classes=3,
            num_decomps=2,
            num_subsets=3,
            num_mixtures=2,
            input_dist=spn.DenseSPNGenerator.InputDist.MIXTURE,
            num_input_mixtures=None,
            weight_init_value=tf.initializers.random_uniform(0.0, 1.0)(0, 1))
        root = model.build(num_vars=6, num_vals=2)
        self.generic_model_test("3class",
                                root, model.sample_ivs, model.class_ivs)

    def test_discretedense_1class_externalivs(self):
        model = spn.DiscreteDenseModel(
            num_classes=1,
            num_decomps=2,
            num_subsets=3,
            num_mixtures=2,
            input_dist=spn.DenseSPNGenerator.InputDist.MIXTURE,
            num_input_mixtures=None,
            weight_init_value=tf.initializers.random_uniform(0.0, 1.0)(0, 1))
        sample_ivs = spn.IVs(num_vars=6, num_vals=2)
        root = model.build(sample_ivs)
        self.generic_model_test("1class",
                                root, sample_ivs, None)

    def test_discretedense_3class_externalivs(self):
        model = spn.DiscreteDenseModel(
            num_classes=3,
            num_decomps=2,
            num_subsets=3,
            num_mixtures=2,
            input_dist=spn.DenseSPNGenerator.InputDist.MIXTURE,
            num_input_mixtures=None,
            weight_init_value=tf.initializers.random_uniform(0.0, 1.0)(0, 1))
        sample_ivs = spn.IVs(num_vars=6, num_vals=2)
        class_ivs = spn.IVs(num_vars=1, num_vals=3)
        root = model.build(sample_ivs, class_input=class_ivs)
        self.generic_model_test("3class",
                                root, sample_ivs, class_ivs)

    def test_discretedense_saving_1class_internalivs(self):
        model1 = spn.DiscreteDenseModel(
            num_classes=1,
            num_decomps=2,
            num_subsets=3,
            num_mixtures=2,
            input_dist=spn.DenseSPNGenerator.InputDist.MIXTURE,
            num_input_mixtures=None,
            weight_init_value=tf.initializers.random_uniform(0.0, 1.0)(0, 1))
        model1.build(num_vars=6, num_vals=2)
        init1 = spn.initialize_weights(model1.root)

        feed_samples = np.array(list(itertools.product(range(2), repeat=6)))

        with tf.Session() as sess:
            # Initialize
            init1.run()

            # Save
            path = self.out_path(self.cid() + ".spn")
            model1.save_to_json(path, pretty=True, save_param_vals=True,
                                sess=sess)

        # Reset graph
        tf.reset_default_graph()

        with tf.Session() as sess:
            # Load
            model2 = spn.Model.load_from_json(path,
                                              load_param_vals=True,
                                              sess=sess)
            self.assertIs(type(model2), spn.DiscreteDenseModel)

            val_marginal2 = model2.root.get_value(
                inference_type=spn.InferenceType.MARGINAL)

            # Check model after loading
            self.assertTrue(model2.root.is_valid())
            out_marginal2 = sess.run(val_marginal2,
                                     feed_dict={model2.sample_ivs: feed_samples})
            self.assertAlmostEqual(out_marginal2.sum(), 1.0, places=6)

            # Writing graph
            self.write_tf_graph(sess, self.sid(), self.cid())

    def test_discretedense_saving_3class_internalivs(self):
        model1 = spn.DiscreteDenseModel(
            num_classes=3,
            num_decomps=2,
            num_subsets=3,
            num_mixtures=2,
            input_dist=spn.DenseSPNGenerator.InputDist.MIXTURE,
            num_input_mixtures=None,
            weight_init_value=tf.initializers.random_uniform(0.0, 1.0)(0, 1))
        model1.build(num_vars=6, num_vals=2)
        init1 = spn.initialize_weights(model1.root)

        feed_samples = list(itertools.product(range(2), repeat=6))
        feed_class = np.array([i for i in range(3)
                               for _ in range(len(feed_samples))]).reshape(-1, 1)
        feed_samples = np.array(feed_samples * 3)

        with tf.Session() as sess:
            # Initialize
            init1.run()

            # Save
            path = self.out_path(self.cid() + ".spn")
            model1.save_to_json(path, pretty=True, save_param_vals=True,
                                sess=sess)

        # Reset graph
        tf.reset_default_graph()

        with tf.Session() as sess:
            # Load
            model2 = spn.Model.load_from_json(path,
                                              load_param_vals=True,
                                              sess=sess)
            self.assertIs(type(model2), spn.DiscreteDenseModel)

            val_marginal2 = model2.root.get_value(
                inference_type=spn.InferenceType.MARGINAL)

            # Check model after loading
            self.assertTrue(model2.root.is_valid())
            out_marginal2 = sess.run(val_marginal2,
                                     feed_dict={model2.sample_ivs: feed_samples,
                                                model2.class_ivs: feed_class})
            self.assertAlmostEqual(out_marginal2.sum(), 1.0, places=6)

            # Writing graph
            self.write_tf_graph(sess, self.sid(), self.cid())

    def test_discretedense_saving_1class_externalivs(self):
        model1 = spn.DiscreteDenseModel(
            num_classes=1,
            num_decomps=2,
            num_subsets=3,
            num_mixtures=2,
            input_dist=spn.DenseSPNGenerator.InputDist.MIXTURE,
            num_input_mixtures=None,
            weight_init_value=tf.initializers.random_uniform(0.0, 1.0)(0, 1))
        sample_ivs1 = spn.IVs(num_vars=6, num_vals=2)
        model1.build(sample_ivs1)
        init1 = spn.initialize_weights(model1.root)

        feed_samples = np.array(list(itertools.product(range(2), repeat=6)))

        with tf.Session() as sess:
            # Initialize
            init1.run()

            # Save
            path = self.out_path(self.cid() + ".spn")
            model1.save_to_json(path, pretty=True, save_param_vals=True,
                                sess=sess)

        # Reset graph
        tf.reset_default_graph()

        with tf.Session() as sess:
            # Load
            model2 = spn.Model.load_from_json(path,
                                              load_param_vals=True,
                                              sess=sess)
            self.assertIs(type(model2), spn.DiscreteDenseModel)

            val_marginal2 = model2.root.get_value(
                inference_type=spn.InferenceType.MARGINAL)

            # Check model after loading
            self.assertTrue(model2.root.is_valid())
            out_marginal2 = sess.run(
                val_marginal2,
                feed_dict={model2.sample_inputs[0].node: feed_samples})
            self.assertAlmostEqual(out_marginal2.sum(), 1.0, places=6)

            # Writing graph
            self.write_tf_graph(sess, self.sid(), self.cid())

    def test_discretedense_saving_3class_externalivs(self):
        model1 = spn.DiscreteDenseModel(
            num_classes=3,
            num_decomps=2,
            num_subsets=3,
            num_mixtures=2,
            input_dist=spn.DenseSPNGenerator.InputDist.MIXTURE,
            num_input_mixtures=None,
            weight_init_value=tf.initializers.random_uniform(0.0, 1.0)(0, 1))
        sample_ivs1 = spn.IVs(num_vars=6, num_vals=2)
        class_ivs1 = spn.IVs(num_vars=1, num_vals=3)
        model1.build(sample_ivs1, class_input=class_ivs1)
        init1 = spn.initialize_weights(model1.root)

        feed_samples = list(itertools.product(range(2), repeat=6))
        feed_class = np.array([i for i in range(3)
                               for _ in range(len(feed_samples))]).reshape(-1, 1)
        feed_samples = np.array(feed_samples * 3)

        with tf.Session() as sess:
            # Initialize
            init1.run()

            # Save
            path = self.out_path(self.cid() + ".spn")
            model1.save_to_json(path, pretty=True, save_param_vals=True,
                                sess=sess)

        # Reset graph
        tf.reset_default_graph()

        with tf.Session() as sess:
            # Load
            model2 = spn.Model.load_from_json(path,
                                              load_param_vals=True,
                                              sess=sess)
            self.assertIs(type(model2), spn.DiscreteDenseModel)

            val_marginal2 = model2.root.get_value(
                inference_type=spn.InferenceType.MARGINAL)

            # Check model after loading
            self.assertTrue(model2.root.is_valid())
            out_marginal2 = sess.run(
                val_marginal2,
                feed_dict={model2.sample_inputs[0].node: feed_samples,
                           model2.class_input.node: feed_class})
            self.assertAlmostEqual(out_marginal2.sum(), 1.0, places=6)

            # Writing graph
            self.write_tf_graph(sess, self.sid(), self.cid())


if __name__ == '__main__':
    tf.test.main()
