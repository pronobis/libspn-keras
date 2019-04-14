#!/usr/bin/env python3

from context import libspn as spn
from test import TestCase
import tensorflow as tf
import numpy as np


class TestInference(TestCase):

    def test_marginal_value(self):
        """Calculation of SPN marginal value"""
        # Generate SPN
        model = spn.Poon11NaiveMixtureModel()
        model.build()
        # Set default inference type for each node
        model.root.set_inference_types(spn.InferenceType.MARGINAL)
        # Get values
        init = spn.initialize_weights(model.root)
        val_marginal = model.root.get_value(
            inference_type=spn.InferenceType.MARGINAL)
        val_default = model.root.get_value()
        val_log_marginal = model.root.get_log_value(
            inference_type=spn.InferenceType.MARGINAL)
        val_log_default = model.root.get_log_value()
        with self.test_session() as sess:
            init.run()
            out_default = sess.run(val_default, feed_dict={model.latent_indicators: model.feed})
            out_marginal = sess.run(val_marginal, feed_dict={model.latent_indicators: model.feed})
            out_log_default = sess.run(tf.exp(val_log_default),
                                       feed_dict={model.latent_indicators: model.feed})
            out_log_marginal = sess.run(tf.exp(val_log_marginal),
                                        feed_dict={model.latent_indicators: model.feed})
        # Check if values sum to 1
        # WARNING: Below does not pass test for places=7 with float32 dtype
        self.assertAlmostEqual(out_default[np.all(model.feed >= 0, axis=1), :].sum(),
                               1.0, places=6)
        self.assertAlmostEqual(out_marginal[np.all(model.feed >= 0, axis=1), :].sum(),
                               1.0, places=6)
        self.assertAlmostEqual(out_log_default[np.all(model.feed >= 0, axis=1), :].sum(),
                               1.0, places=6)
        self.assertAlmostEqual(out_log_marginal[np.all(model.feed >= 0, axis=1), :].sum(),
                               1.0, places=6)
        # Check joint probabilities
        np.testing.assert_array_almost_equal(out_default, model.true_values)
        np.testing.assert_array_almost_equal(out_marginal, model.true_values)
        np.testing.assert_array_almost_equal(out_log_default, model.true_values)
        np.testing.assert_array_almost_equal(out_log_marginal, model.true_values)

    def test_mpe_value(self):
        """Calculation of SPN MPE value"""
        # Generate SPN
        model = spn.Poon11NaiveMixtureModel()
        model.build()
        # Set default inference type for each node
        model.root.set_inference_types(spn.InferenceType.MPE)
        # Get values
        init = spn.initialize_weights(model.root)
        val_mpe = model.root.get_value(inference_type=spn.InferenceType.MPE)
        val_default = model.root.get_value()
        val_log_mpe = model.root.get_log_value(inference_type=spn.InferenceType.MPE)
        val_log_default = model.root.get_log_value()
        with self.test_session() as sess:
            init.run()
            out_default = sess.run(val_default, feed_dict={model.latent_indicators: model.feed})
            out_mpe = sess.run(val_mpe, feed_dict={model.latent_indicators: model.feed})
            out_log_default = sess.run(tf.exp(val_log_default),
                                       feed_dict={model.latent_indicators: model.feed})
            out_log_mpe = sess.run(tf.exp(val_log_mpe),
                                   feed_dict={model.latent_indicators: model.feed})
        # Check joint probabilities
        np.testing.assert_array_almost_equal(out_default, model.true_mpe_values)
        np.testing.assert_array_almost_equal(out_mpe, model.true_mpe_values)
        np.testing.assert_array_almost_equal(out_log_default, model.true_mpe_values)
        np.testing.assert_array_almost_equal(out_log_mpe, model.true_mpe_values)

    def test_mixed_value(self):
        """Calculation of a mixed MPE/marginal value"""
        # Generate SPN
        model = spn.Poon11NaiveMixtureModel()
        model.build()
        # Set default inference type for each node
        model.root.set_inference_types(spn.InferenceType.MARGINAL)
        model.root.inference_type = spn.InferenceType.MPE
        # Get values
        init = spn.initialize_weights(model.root)
        val_marginal = model.root.get_value(inference_type=spn.InferenceType.MARGINAL)
        val_mpe = model.root.get_value(inference_type=spn.InferenceType.MPE)
        val_default = model.root.get_value()
        val_log_marginal = model.root.get_log_value(inference_type=spn.InferenceType.MARGINAL)
        val_log_mpe = model.root.get_log_value(inference_type=spn.InferenceType.MPE)
        val_log_default = model.root.get_log_value()
        with self.test_session() as sess:
            init.run()
            out_default = sess.run(val_default, feed_dict={model.latent_indicators: model.feed})
            out_marginal = sess.run(val_marginal, feed_dict={model.latent_indicators: model.feed})
            out_mpe = sess.run(val_mpe, feed_dict={model.latent_indicators: model.feed})
            out_log_default = sess.run(tf.exp(val_log_default),
                                       feed_dict={model.latent_indicators: model.feed})
            out_log_marginal = sess.run(tf.exp(val_log_marginal),
                                        feed_dict={model.latent_indicators: model.feed})
            out_log_mpe = sess.run(tf.exp(val_log_mpe),
                                   feed_dict={model.latent_indicators: model.feed})
        # Check joint probabilities
        true_default = [[0.5],
                        [0.35],
                        [0.15],
                        [0.2],
                        [0.14],
                        [0.06],
                        [0.3],
                        [0.216],
                        [0.09]]
        np.testing.assert_array_almost_equal(out_default, true_default)
        np.testing.assert_array_almost_equal(out_marginal, model.true_values)
        np.testing.assert_array_almost_equal(out_mpe, model.true_mpe_values)
        np.testing.assert_array_almost_equal(out_log_default, true_default)
        np.testing.assert_array_almost_equal(out_log_marginal, model.true_values)
        np.testing.assert_array_almost_equal(out_log_mpe, model.true_mpe_values)

    def test_mpe_path(self):
        # Generate SPN
        model = spn.Poon11NaiveMixtureModel()
        model.build()
        # Add ops
        init = spn.initialize_weights(model.root)
        mpe_path_gen = spn.MPEPath(value_inference_type=spn.InferenceType.MPE,
                                   log=False)
        mpe_path_gen_log = spn.MPEPath(value_inference_type=spn.InferenceType.MPE,
                                       log=True)
        mpe_path_gen.get_mpe_path(model.root)
        mpe_path_gen_log.get_mpe_path(model.root)
        # Run
        with self.test_session() as sess:
            init.run()
            out = sess.run(mpe_path_gen.counts[model.latent_indicators],
                           feed_dict={model.latent_indicators: model.feed})
            out_log = sess.run(mpe_path_gen_log.counts[model.latent_indicators],
                               feed_dict={model.latent_indicators: model.feed})

        true_latent_indicators_counts = np.array([[0., 1., 1., 0.],
                                    [0., 1., 1., 0.],
                                    [0., 1., 0., 1.],
                                    [1., 0., 1., 0.],
                                    [1., 0., 1., 0.],
                                    [1., 0., 0., 1.],
                                    [0., 1., 1., 0.],
                                    [0., 1., 1., 0.],
                                    [0., 1., 0., 1.]],
                                   dtype=spn.conf.dtype.as_numpy_dtype)

        np.testing.assert_array_equal(out, true_latent_indicators_counts)
        np.testing.assert_array_equal(out_log, true_latent_indicators_counts)

    def test_mpe_state(self):
        # Generate SPN
        model = spn.Poon11NaiveMixtureModel()
        model.build()
        # Add ops
        init = spn.initialize_weights(model.root)
        mpe_state_gen = spn.MPEState(value_inference_type=spn.InferenceType.MPE,
                                     log=False)
        mpe_state_gen_log = spn.MPEState(value_inference_type=spn.InferenceType.MPE,
                                         log=True)
        latent_indicators_state, = mpe_state_gen.get_state(model.root, model.latent_indicators)
        latent_indicators_state_log, = mpe_state_gen_log.get_state(model.root, model.latent_indicators)
        # Run
        with self.test_session() as sess:
            init.run()
            out = sess.run(latent_indicators_state, feed_dict={model.latent_indicators: [[-1, -1]]})
            out_log = sess.run(latent_indicators_state_log, feed_dict={model.latent_indicators: [[-1, -1]]})

        # For now we only compare the actual MPE state for input IndicatorLeaf -1
        np.testing.assert_array_equal(out.ravel(), model.true_mpe_state)
        np.testing.assert_array_equal(out_log.ravel(), model.true_mpe_state)


if __name__ == '__main__':
    tf.test.main()
