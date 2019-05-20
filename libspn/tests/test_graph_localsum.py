import tensorflow as tf
from libspn.tests.test import argsprod
import libspn as spn
import numpy as np


class TestBaseSum(tf.test.TestCase):

    @argsprod([False, True], [spn.InferenceType.MARGINAL, spn.InferenceType.MPE])
    def test_compare_manual_conv(self, log_weights, inference_type):
        spn.conf.argmax_zero = True
        grid_dims = [2, 2]
        nrows, ncols = grid_dims
        num_vals = 4
        batch_size = 256
        num_vars = grid_dims[0] * grid_dims[1]
        indicator_leaf = spn.IndicatorLeaf(num_vars=num_vars, num_vals=num_vals)
        num_sums = 32
        weights = spn.Weights(
            num_weights=num_vals, num_sums=num_sums * num_vars,
            initializer=tf.initializers.random_uniform(), log=log_weights)

        weights_per_cell = tf.split(weights.variable, num_or_size_splits=num_vars)

        parsums = []
        for row in range(nrows):
            for col in range(ncols):
                indices = list(range(row * (ncols * num_vals) + col * num_vals,
                                     row * (ncols * num_vals) + (col + 1) * num_vals))
                parsums.append(spn.ParallelSums((indicator_leaf, indices), num_sums=num_sums))

        parsum_concat = spn.Concat(*parsums, name="ParSumConcat")
        convsum = spn.LocalSums(
            indicator_leaf, num_channels=num_sums, weights=weights, spatial_dim_sizes=grid_dims)

        prod00_conv = spn.PermuteProducts(
            (convsum, list(range(num_sums))), (convsum, list(range(num_sums, num_sums * 2))),
            name="Prod00")
        prod01_conv = spn.PermuteProducts(
            (convsum, list(range(num_sums * 2, num_sums * 3))),
            (convsum, list(range(num_sums * 3, num_sums * 4))),
            name="Prod01")
        sum00_conv = spn.ParallelSums(prod00_conv, num_sums=2)
        sum01_conv = spn.ParallelSums(prod01_conv, num_sums=2)

        prod10_conv = spn.PermuteProducts(sum00_conv, sum01_conv, name="Prod10")

        conv_root = spn.Sum(prod10_conv)

        prod00_pars = spn.PermuteProducts(
            (parsum_concat, list(range(num_sums))),
            (parsum_concat, list(range(num_sums, num_sums * 2))))
        prod01_pars = spn.PermuteProducts(
            (parsum_concat, list(range(num_sums * 2, num_sums * 3))),
            (parsum_concat, list(range(num_sums * 3, num_sums * 4))))

        sum00_pars = spn.ParallelSums(prod00_pars, num_sums=2)
        sum01_pars = spn.ParallelSums(prod01_pars, num_sums=2)

        prod10_pars = spn.PermuteProducts(sum00_pars, sum01_pars)

        parsum_root = spn.Sum(prod10_pars)

        node_pairs = [(sum00_conv, sum00_pars), (sum01_conv, sum01_pars), (conv_root, parsum_root)]

        self.assertTrue(conv_root.is_valid())
        self.assertTrue(parsum_root.is_valid())

        self.assertAllEqual(parsum_concat.get_scope(), convsum.get_scope())

        spn.generate_weights(
            conv_root, log=log_weights, initializer=tf.initializers.random_uniform())
        spn.generate_weights(
            parsum_root, log=log_weights, initializer=tf.initializers.random_uniform())

        convsum.set_weights(weights)
        copy_weight_ops = []
        parsum_weight_nodes = []
        for p, w in zip(parsums, weights_per_cell):
            copy_weight_ops.append(tf.assign(p.weights.node.variable, w))
            parsum_weight_nodes.append(p.weights.node)

        for wc, wp in node_pairs:
            copy_weight_ops.append(tf.assign(wp.weights.node.variable, wc.weights.node.variable))

        copy_weights_op = tf.group(*copy_weight_ops)

        init_conv = spn.initialize_weights(conv_root)
        init_parsum = spn.initialize_weights(parsum_root)

        path_conv = spn.MPEPath(value_inference_type=inference_type)
        path_conv.get_mpe_path(conv_root)

        path_parsum = spn.MPEPath(value_inference_type=inference_type)
        path_parsum.get_mpe_path(parsum_root)

        indicator_counts_parsum = path_parsum.counts[indicator_leaf]
        indicator_counts_convsum = path_conv.counts[indicator_leaf]

        weight_counts_parsum = tf.concat(
            [path_parsum.counts[w] for w in parsum_weight_nodes], axis=1)
        weight_counts_conv = path_conv.counts[weights]

        weight_parsum_concat = tf.concat(
            [w.variable for w in parsum_weight_nodes], axis=0)

        root_val_parsum = parsum_root.get_log_value() #path_parsum.value.values[parsum_root]
        root_val_conv = conv_root.get_log_value() #path_conv.value.values[conv_root]

        parsum_counts = path_parsum.counts[parsum_concat]
        conv_counts = path_conv.counts[convsum]

        indicator_feed = np.random.randint(-1, 2, size=batch_size * num_vars)\
            .reshape((batch_size, num_vars))
        with tf.Session() as sess:
            sess.run([init_conv, init_parsum])
            sess.run(copy_weights_op)
            indicator_counts_conv_out, indicator_counts_parsum_out = sess.run(
                [indicator_counts_convsum, indicator_counts_parsum], feed_dict={indicator_leaf: indicator_feed})

            root_conv_value_out, root_parsum_value_out = sess.run(
                [root_val_conv, root_val_parsum], feed_dict={indicator_leaf: indicator_feed})

            weight_counts_conv_out, weight_counts_parsum_out = sess.run(
                [weight_counts_conv, weight_counts_parsum], feed_dict={indicator_leaf: indicator_feed})

            weight_value_conv_out, weight_value_parsum_out = sess.run(
                [convsum.weights.node.variable, weight_parsum_concat])

            parsum_counts_out, conv_counts_out = sess.run(
                [parsum_counts, conv_counts], feed_dict={indicator_leaf: indicator_feed})

            parsum_concat_val, convsum_val = sess.run(
                [path_parsum.value.values[parsum_concat], path_conv.value.values[convsum]],
                feed_dict={indicator_leaf: indicator_feed})

        self.assertAllClose(convsum_val, parsum_concat_val)
        self.assertAllClose(weight_value_conv_out, weight_value_parsum_out)
        self.assertAllClose(root_conv_value_out, root_parsum_value_out)
        self.assertAllEqual(indicator_counts_conv_out, indicator_counts_parsum_out)
        self.assertAllEqual(parsum_counts_out, conv_counts_out)
        self.assertAllEqual(weight_counts_conv_out, weight_counts_parsum_out)







