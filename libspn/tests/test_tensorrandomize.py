#!/usr/bin/env python3

# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

from context import libspn as spn
from libspn.graph.tensorrandomize import TensorRandomize
from libspn.graph.tensorproduct import TensorProduct
from libspn.graph.tensorsum import TensorSum
from libspn.graph.tensor_merge_decomps import TensorMergeDecomps
from test import TestCase
import tensorflow as tf
import numpy as np
import collections
from random import shuffle
from parameterized import parameterized
import itertools


class TestTensorRandomize(tf.test.TestCase):

    # def test_generate_permutations(self):
    #     iv = spn.IVs(num_vals=2, num_vars=13)
    #     randomize = TensorRandomize(iv, num_decomps=2)
    #     randomize.generate_permutations([4, 4])

    def test_small_spn(self):
        num_vars = 13

        iv = spn.IVs(num_vals=2, num_vars=num_vars)
        randomize = TensorRandomize(iv, num_decomps=2)
        p0 = TensorProduct(randomize, num_factors=4)
        s0 = TensorSum(p0, num_sums=3)
        p1 = TensorProduct(s0, num_factors=2)
        s1 = TensorSum(p1, num_sums=3)
        p2 = TensorProduct(s1, num_factors=2)
        m = TensorMergeDecomps(p2)
        root = TensorSum(m, num_sums=1)

        latent = root.generate_ivs(name="Latent")
        randomize.generate_permutations([4, 2, 2])
        spn.generate_weights(root, init_value=spn.ValueType.RANDOM_UNIFORM(0, 1))

        valgen = spn.LogValue()
        val = valgen.get_value(root)
        logsum = tf.reduce_logsumexp(val)

        num_possibilities = 2 ** num_vars
        nums = np.arange(num_possibilities).reshape((num_possibilities, 1))
        powers = 2 ** np.arange(num_vars).reshape((1, num_vars))
        ivs_feed = np.bitwise_and(nums, powers) // powers

        with self.test_session() as sess:
            sess.run(spn.initialize_weights(root))
            out = sess.run(logsum, {iv: ivs_feed,
                                    latent: -np.ones((ivs_feed.shape[0], 1), dtype=np.int32)})

        self.assertAllClose(out, 0.0)


if __name__ == '__main__':
    tf.test.main()
