import tensorflow as tf
from libspn.tests.test import argsprod
import libspn as spn
from libspn.graph.convsum import ConvSum
import numpy as np
import random


class TestBaseSum(tf.test.TestCase):

    @argsprod([False, True], [spn.InferenceType.MARGINAL, spn.InferenceType.MPE])
    def test_compare_manual_conv(self, log_weights, inference_type):
        spn.conf.argmax_zero = True
        grid_dims = [2, 2]
        nrows, ncols = grid_dims
        num_vals = 4
        batch_size = 128
        num_vars = grid_dims[0] * grid_dims[1]
        ivs = spn.IVs(num_vars=num_vars, num_vals=num_vals)
        num_sums = 6
        weights = spn.Weights(
            num_weights=num_vals, num_sums=num_sums * num_vars,
            init_value=spn.ValueType.RANDOM_UNIFORM(), log=log_weights)

        weights_per_cell = tf.split(weights.variable, num_or_size_splits=num_vars)

        parsums = []
        for row in range(nrows):
            for col in range(ncols):
                indices = list(range(row * (ncols * num_vals) + col * num_vals,
                                     row * (ncols * num_vals) + (col + 1) * num_vals))
                parsums.append(spn.ParSums((ivs, indices), num_sums=num_sums))

        convsum = spn.LocalSum(
            ivs, num_channels=num_sums, weights=weights, grid_dim_sizes=grid_dims)

        dense_gen = spn.DenseSPNGeneratorLayerNodes(
            num_decomps=1, num_mixtures=2, num_subsets=2,
            input_dist=spn.DenseSPNGeneratorLayerNodes.InputDist.RAW,
            node_type=spn.DenseSPNGeneratorLayerNodes.NodeType.BLOCK)

        rnd = random.Random(1234)
        rnd_state = rnd.getstate()
        conv_root = dense_gen.generate(convsum, rnd=rnd)
        rnd.setstate(rnd_state)

        parsum_concat = spn.Concat(*parsums, name="ParSumConcat")
        parsum_root = dense_gen.generate(parsum_concat, rnd=rnd)

        self.assertTrue(conv_root.is_valid())
        self.assertTrue(parsum_root.is_valid())

        self.assertAllEqual(parsum_concat.get_scope(), convsum.get_scope())

        spn.generate_weights(conv_root, log=log_weights)
        spn.generate_weights(parsum_root, log=log_weights)

        convsum.set_weights(weights)
        copy_weight_ops = []
        parsum_weight_nodes = []
        for p, w in zip(parsums, weights_per_cell):
            copy_weight_ops.append(tf.assign(p.weights.node.variable, w))
            parsum_weight_nodes.append(p.weights.node)
        copy_weights_op = tf.group(*copy_weight_ops)
        # [p.set_weights(weights) for p in parsums]

        init_conv = spn.initialize_weights(conv_root)
        init_parsum = spn.initialize_weights(parsum_root)

        path_conv = spn.MPEPath(value_inference_type=inference_type)
        path_conv.get_mpe_path(conv_root)

        path_parsum = spn.MPEPath(value_inference_type=inference_type)
        path_parsum.get_mpe_path(parsum_root)

        ivs_counts_parsum = path_parsum.counts[ivs]
        ivs_counts_conv = path_conv.counts[ivs]

        # weight_counts_parsum = path_parsum.counts[weights]
        weight_counts_parsum = tf.concat(
            [path_parsum.counts[w] for w in parsum_weight_nodes], axis=1)
        weight_counts_conv = path_conv.counts[weights]

        weight_parsum_concat = tf.concat(
            [w.variable for w in parsum_weight_nodes], axis=0)

        root_val_parsum = path_parsum.value.values[parsum_root]
        root_val_conv = path_conv.value.values[conv_root]

        parsum_counts = path_parsum.counts[parsum_concat]
        conv_counts = path_conv.counts[convsum]

        ivs_feed = np.random.randint(2, size=batch_size * num_vars)\
            .reshape((batch_size, num_vars))
        with tf.Session() as sess:
            sess.run([init_conv, init_parsum])
            sess.run(copy_weights_op)
            ivs_counts_conv_out, ivs_counts_parsum_out = sess.run(
                [ivs_counts_conv, ivs_counts_parsum], feed_dict={ivs: ivs_feed})

            root_conv_value_out, root_parsum_value_out = sess.run(
                [root_val_conv, root_val_parsum], feed_dict={ivs: ivs_feed})

            weight_counts_conv_out, weight_counts_parsum_out = sess.run(
                [weight_counts_conv, weight_counts_parsum], feed_dict={ivs: ivs_feed})

            weight_value_conv_out, weight_value_parsum_out = sess.run(
                [convsum.weights.node.variable, weight_parsum_concat])

            parsum_counts_out, conv_counts_out = sess.run(
                [parsum_counts, conv_counts], feed_dict={ivs: ivs_feed})

            parsum_concat_val, convsum_val = sess.run(
                [path_parsum.value.values[parsum_concat], path_conv.value.values[convsum]],
                feed_dict={ivs: ivs_feed})

        self.assertTrue(np.all(np.less_equal(convsum_val, 0.0)))
        self.assertTrue(np.all(np.less_equal(parsum_concat_val, 0.0)))
        self.assertAllClose(weight_value_conv_out, weight_value_parsum_out)
        self.assertAllClose(root_conv_value_out, root_parsum_value_out)
        self.assertAllEqual(ivs_counts_conv_out, ivs_counts_parsum_out)
        self.assertAllEqual(parsum_counts_out, conv_counts_out)
        self.assertAllEqual(weight_counts_conv_out, weight_counts_parsum_out)







