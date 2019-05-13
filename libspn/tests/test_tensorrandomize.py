import libspn as spn
from libspn.graph.op.block_random_decompositions import BlockRandomDecompositions
from libspn.graph.op.block_permute_product import BlockPermuteProduct
from libspn.graph.op.block_reduce_product import BlockReduceProduct
from libspn.graph.op.block_sum import BlockSum
from libspn.graph.op.block_merge_decomps import BlockMergeDecomps
from libspn.graph.op.block_root_sum import BlockRootSum
import tensorflow as tf
import numpy as np


class TestTensorRandomize(tf.test.TestCase):

    def test_small_spn(self):
        num_vars = 13

        iv = spn.IndicatorLeaf(num_vals=2, num_vars=num_vars)
        randomize = BlockRandomDecompositions(iv, num_decomps=2)
        p0 = BlockPermuteProduct(randomize, num_subsets=4)
        s0 = BlockSum(p0, num_sums_per_block=3)
        p1 = BlockPermuteProduct(s0, num_subsets=2)
        s1 = BlockSum(p1, num_sums_per_block=3)
        p2 = BlockReduceProduct(s1, num_subsets=2)
        m = BlockMergeDecomps(p2, factor=2)
        root = BlockRootSum(m)

        latent = root.generate_latent_indicators(name="Latent")
        spn.generate_weights(root, initializer=tf.initializers.random_uniform())

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
