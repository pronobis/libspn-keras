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
import scipy.stats as stats

# Batch size is pretty large to obtain good approximations
BATCH_SIZE = int(1e5)


class TestGaussianQuantile(TestCase):

    def test_values_per_quantile(self):
        quantiles = [np.random.rand(32, 32) + i * 2 for i in range(4)]
        data = np.concatenate(quantiles, axis=0)
        np.random.shuffle(data)
        gq = spn.GaussianQuantile(num_vars=32, num_components=4)

        values_per_quantile = gq._values_per_quantile(data)

        for val, q in zip(values_per_quantile, quantiles):
            self.assertAllClose(np.sort(q, axis=0), val)

    def test_value(self):
        num_vars = 8
        data = np.stack(
            [np.random.normal(a, size=BATCH_SIZE) for a in range(num_vars)], axis=1)

        data = np.concatenate([data, np.stack(
            [np.random.normal(a, size=BATCH_SIZE) + num_vars for a in range(num_vars)], axis=1)],
                              axis=0).astype(np.float32)

        gq = spn.GaussianQuantile(num_vars=num_vars, num_components=2)
        gq.learn_from_data(data)

        value_op = gq._compute_value()
        log_value_op = gq._compute_log_value()

        modes = np.stack([np.arange(num_vars) for _ in range(BATCH_SIZE)] +
                         [np.arange(num_vars) + num_vars for _ in range(BATCH_SIZE)], axis=0)
        val_at_mode = stats.norm.pdf(0)

        with self.test_session() as sess:
            value_out, log_value_out = sess.run(
                [value_op, log_value_op], feed_dict={gq.feed: modes})

        value_out = value_out.reshape((BATCH_SIZE * 2, num_vars, 2))
        log_value_out = log_value_out.reshape((BATCH_SIZE * 2, num_vars, 2))

        # We'll be quite tolerant for the error, as our output is really just an empirical mean
        self.assertAllClose(
            value_out[:BATCH_SIZE, :, 0], np.ones([BATCH_SIZE, num_vars]) * val_at_mode,
            rtol=1e-2, atol=1e-2)

        self.assertAllClose(
            value_out[BATCH_SIZE:, :, 1], np.ones([BATCH_SIZE, num_vars]) * val_at_mode,
            rtol=1e-2, atol=1e-2)

        self.assertAllClose(
            np.exp(log_value_out[:BATCH_SIZE, :, 0]), np.ones([BATCH_SIZE, num_vars]) * val_at_mode,
            rtol=1e-2, atol=1e-2)

        self.assertAllClose(
            np.exp(log_value_out[BATCH_SIZE:, :, 1]), np.ones([BATCH_SIZE, num_vars]) * val_at_mode,
            rtol=1e-2, atol=1e-2)

    def test_mpe_state(self):
        num_vars = 4
        data = np.stack(
            [np.random.normal(a, size=BATCH_SIZE) for a in range(num_vars)], axis=1)

        data = np.concatenate([data, np.stack(
            [np.random.normal(a, size=BATCH_SIZE) + num_vars for a in range(num_vars)], axis=1)],
                              axis=0).astype(np.float32)

        gq = spn.GaussianQuantile(num_vars=num_vars, num_components=2)
        gq.learn_from_data(data)

        batch_size = 3
        left = np.random.randint(2, size=batch_size * num_vars).reshape((-1, num_vars))
        counts = np.stack((left, 1 - left), axis=-1)

        mpe_truth = []
        for vars in left:
            for i, val in enumerate(vars):
                mpe_truth.append(i if val == 1 else i + num_vars)

        mpe_truth = np.reshape(mpe_truth, (-1, num_vars))

        mpe_state = gq._compute_mpe_state(counts)

        with self.test_session() as sess:
            mpe_state_out = sess.run(mpe_state)
        # Again we must be quite tolerant, but that's ok, the predictions are 1.0 apart
        self.assertAllClose(mpe_truth, mpe_state_out, atol=1e-1, rtol=1e-1)


if __name__ == '__main__':
    tf.test.main()
