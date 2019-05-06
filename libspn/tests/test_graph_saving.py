#!/usr/bin/env python3

from context import libspn as spn
from test import TestCase
import tensorflow as tf
import numpy as np
import itertools


class TestGraphSaving(TestCase):

    def test_withoutparams_initfixed(self):
        # Build an SPN
        model = spn.Poon11NaiveMixtureModel()
        root1 = model.build()

        # Save
        path = self.out_path(self.cid() + ".spn")
        saver = spn.JSONSaver(path, pretty=True)
        saver.save(root1, save_param_vals=False)

        # Reset graph
        tf.reset_default_graph()

        # Load
        loader = spn.JSONLoader(path)
        root2 = loader.load()
        latent_indicators2 = loader.find_node('IndicatorLeaf')
        init2 = spn.initialize_weights(root2)
        val_mpe2 = root2.get_value(inference_type=spn.InferenceType.MPE)
        val_marginal2 = root2.get_value(inference_type=spn.InferenceType.MARGINAL)

        # Check model after loading
        self.assertTrue(root2.is_valid())
        with self.test_session() as sess:
            init2.run()
            out_marginal2 = sess.run(val_marginal2, feed_dict={latent_indicators2: model.feed})
            out_mpe2 = sess.run(val_mpe2, feed_dict={latent_indicators2: model.feed})
        self.assertAlmostEqual(out_marginal2[np.all(model.feed >= 0, axis=1), :].sum(),
                               1.0, places=6)
        np.testing.assert_array_almost_equal(out_marginal2, model.true_values)
        np.testing.assert_array_almost_equal(out_mpe2, model.true_mpe_values)

    def test_withoutparams_initrandom(self):
        # Build an SPN
        feed = np.array(list(itertools.product(range(2), repeat=6)))
        model = spn.DiscreteDenseModel(
            num_classes=1, num_decomps=1, num_subsets=3,
            num_mixtures=2, weight_initializer=tf.initializers.random_uniform(0.0, 1.0))
        root1 = model.build(num_vars=6, num_vals=2)

        # Save
        path = self.out_path(self.cid() + ".spn")
        saver = spn.JSONSaver(path, pretty=True)
        saver.save(root1, save_param_vals=False)

        # Reset graph
        tf.reset_default_graph()

        # Load
        loader = spn.JSONLoader(path)
        root2 = loader.load()
        latent_indicators2 = loader.find_node('SampleIndicatorLeaf')
        init2 = spn.initialize_weights(root2)
        val_marginal2 = root2.get_value(inference_type=spn.InferenceType.MARGINAL)

        # Check model after loading
        self.assertTrue(root2.is_valid())
        with self.test_session() as sess:
            init2.run()
            out_marginal2 = sess.run(val_marginal2, feed_dict={latent_indicators2: feed})
        self.assertAlmostEqual(out_marginal2.sum(), 1.0, places=6)

    def test_withparams_initfixed(self):
        # Build an SPN
        model = spn.Poon11NaiveMixtureModel()
        root1 = model.build()
        init1 = spn.initialize_weights(root1)

        with self.test_session() as sess:
            # Initialize
            init1.run()

            # Save
            path = self.out_path(self.cid() + ".spn")
            saver = spn.JSONSaver(path, pretty=True)
            saver.save(root1, save_param_vals=True)

        # Reset graph
        tf.reset_default_graph()

        with self.test_session() as sess:
            # Load
            loader = spn.JSONLoader(path)
            root2 = loader.load(load_param_vals=True)
            latent_indicators2 = loader.find_node('IndicatorLeaf')
            val_mpe2 = root2.get_value(inference_type=spn.InferenceType.MPE)
            val_marginal2 = root2.get_value(inference_type=spn.InferenceType.MARGINAL)

            # Check model after loading
            self.assertTrue(root2.is_valid())
            out_marginal2 = sess.run(val_marginal2, feed_dict={latent_indicators2: model.feed})
            out_mpe2 = sess.run(val_mpe2, feed_dict={latent_indicators2: model.feed})
            self.assertAlmostEqual(out_marginal2[np.all(model.feed >= 0, axis=1), :].sum(),
                                   1.0, places=6)
            np.testing.assert_array_almost_equal(out_marginal2, model.true_values)
            np.testing.assert_array_almost_equal(out_mpe2, model.true_mpe_values)

            # Writing graph
            self.write_tf_graph(sess, self.sid(), self.cid())

    def test_withparams_initrandom(self):
        # Build an SPN
        feed = np.array(list(itertools.product(range(2), repeat=6)))
        model = spn.DiscreteDenseModel(
            num_classes=1, num_decomps=1, num_subsets=3,
            num_mixtures=2, weight_initializer=tf.initializers.random_uniform(0.0, 1.0))
        root1 = model.build(num_vars=6, num_vals=2)
        init1 = spn.initialize_weights(root1)

        with self.test_session() as sess:
            # Initialize
            init1.run()

            # Save
            path = self.out_path(self.cid() + ".spn")
            saver = spn.JSONSaver(path, pretty=True)
            saver.save(root1, save_param_vals=True)

        # Reset graph
        tf.reset_default_graph()

        with self.test_session() as sess:
            # Load
            loader = spn.JSONLoader(path)
            root2 = loader.load(load_param_vals=True)
            latent_indicators2 = loader.find_node('SampleIndicatorLeaf')
            val_marginal2 = root2.get_value(inference_type=spn.InferenceType.MARGINAL)

            # Check model after loading
            self.assertTrue(root2.is_valid())
            out_marginal2 = sess.run(val_marginal2, feed_dict={latent_indicators2: feed})
            self.assertAlmostEqual(out_marginal2.sum(), 1.0, places=6)

            # Writing graph
            self.write_tf_graph(sess, self.sid(), self.cid())


if __name__ == '__main__':
    tf.test.main()
