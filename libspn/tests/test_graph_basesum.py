import tensorflow as tf
from libspn.tests.test import argsprod
import libspn as spn
import numpy as np
from unittest.mock import MagicMock


class TestBaseSum(tf.test.TestCase):

    @argsprod([False, True])
    def test_dropconnect(self, log):
        ivs = spn.IVs(num_vals=2, num_vars=4)
        s = spn.Sum(ivs, dropconnect_keep_prob=0.5)
        spn.generate_weights(s)
        init = spn.initialize_weights(s)

        mask = [
            [0., 1., 0., 1., 1., 1., 0., 1.],
            [1., 0., 0., 0., 0., 0., 1., 0.]
        ]
        s._create_dropout_mask = MagicMock(
            return_value=tf.expand_dims(tf.log(mask) if log else mask, 1))

        if log:
            val_op = tf.exp(s.get_log_value())
        else:
            val_op = s.get_value()

        mask = tf.constant(mask, dtype=tf.float32)
        truth = tf.reduce_mean(mask, axis=-1, keepdims=True)

        with self.test_session() as sess:
            sess.run(init)
            dropconnect_out, truth_out = sess.run(
                [val_op, truth], feed_dict={ivs: -np.ones((2, 4), dtype=np.int32)})

        self.assertAllClose(dropconnect_out, truth_out)

    @argsprod([False, True])
    def test_stochastic_argmax(self, argmax_zero):
        spn.conf.argmax_zero = argmax_zero

        N = 100000
        s = spn.Sum()

        x = tf.constant(np.repeat([[0, 1, 1, 0, 1], [1, 0, 0, 1, 0]], repeats=N, axis=0))

        argmax_op = tf.squeeze(s._reduce_argmax(tf.expand_dims(x, 1)))

        with self.test_session() as sess:
            argmax = sess.run(argmax_op)

        hist_first, _ = np.histogram(argmax[:N], bins=list(range(6)))
        hist_second, _ = np.histogram(argmax[N:], bins=list(range(6)))

        if argmax_zero:
            self.assertEqual(hist_first[1], N)
            self.assertEqual(hist_second[0], N)
        else:
            [self.assertLess(hist_first[i], N / 3 + N / 6) for i in [1, 2, 4]]
            [self.assertGreater(hist_first[i], N / 3 - N / 6) for i in [1, 2, 4]]

            [self.assertLess(hist_second[i], N / 2 + N / 6) for i in [0, 3]]
            [self.assertGreater(hist_second[i], N / 2 - N / 6) for i in [0, 3]]

    @argsprod([0.2, 0.5, 0.8, 1.0], [False, True])
    def test_sampling(self, sample_prob, log):
        N = 100000
        x = tf.expand_dims(
            tf.constant(
                np.repeat([[1, 2, 2, 1, 3], [3, 1, 1, 2, 1]], repeats=N, axis=0),
                dtype=tf.float32), axis=1)
        s = spn.Sum()

        probs = [[1/9, 2/9, 2/9, 1/9, 3/9], [3/8, 1/8, 1/8, 2/8, 1/8]]

        N_sampled = N * sample_prob
        N_argmax = N - N_sampled

        if log:
            sample_op = tf.squeeze(s._reduce_sample_log(tf.log(x), sample_prob=sample_prob))
        else:
            sample_op = tf.squeeze(s._reduce_sample(x, sample_prob=sample_prob))

        with self.test_session() as sess:
            sample_out = sess.run(sample_op)

        for samples, prob, max_ind in zip(
                np.split(sample_out, indices_or_sections=2), probs, [4, 0]):
            hist, _ = np.histogram(samples, bins=list(range(6)))
            for h, p, i in zip(hist, prob, range(5)):
                if i == max_ind:
                    estimate = N_argmax + N_sampled * p
                else:
                    estimate = N_sampled * p
                self.assertLess(h, estimate + N/6)
                self.assertGreater(h, estimate - N/6)
