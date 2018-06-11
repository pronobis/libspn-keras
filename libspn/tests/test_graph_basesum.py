import tensorflow as tf
from libspn.tests.test import argsprod
import libspn as spn
import numpy as np


class TestBaseSum(tf.test.TestCase):

    @argsprod([False, True])
    def test_dropout_mask(self, log):
        iv = spn.IVs(num_vals=2, num_vars=4)
        s = spn.Sum(iv)
        mask = s._dropout_mask(0.5, (10000, 1, 8), log=log)

        with self.test_session() as sess:
            mask_out = sess.run(mask)

        if log:
            mask_out = np.exp(mask_out)

        self.assertAlmostEqual(np.mean(mask_out), 0.5, places=2)
        self.assertTrue(np.all(np.greater_equal(mask_out, 0.0)))
        self.assertTrue(np.all(np.less_equal(mask_out, 1.0)))

    def test_rank_probs(self):
        arr = [7, 9, 8, 3, 2, 1, -1, -10, 5]
        arr = np.reshape(arr, (3, 1, 3))
        iv = spn.IVs(num_vals=3, num_vars=1)
        s = spn.Sum(iv)

        probs = s._rank_probs(arr)
        row_sum = 1/2 + 1/3 + 1/4
        r1 = 1/2 / row_sum
        r2 = 1/3 / row_sum
        r3 = 1/4 / row_sum
        truth = [r3, r1, r2, r1, r2, r3, r2, r3, r1]
        truth = np.reshape(truth, (3, 1, 3))

        with self.test_session() as sess:
            out = sess.run(probs)

        self.assertAllClose(out, truth)

    @argsprod([False, True])
    def test_maybe_dropout(self, log):
        probs = tf.constant([0.1, 0.2, 0.5, 0.9, 0.1])
        s = spn.Sum()
        dropped = s._maybe_dropout(
            tf.log(probs) if log else probs, dropout_keep_prob=0.5, log=log)
        with self.test_session() as sess:
            out = sess.run(dropped)
        print(np.exp(out) if log else out)