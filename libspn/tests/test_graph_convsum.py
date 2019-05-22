import tensorflow as tf
from libspn.tests.test import argsprod
import libspn as spn
from libspn.graph.op.conv_sums import ConvSums
import numpy as np
import random


class TestBaseSum(tf.test.TestCase):

    @argsprod([False, True], [spn.InferenceType.MARGINAL, spn.InferenceType.MPE])
    def test_compare_manual_conv(self, log_weights, inference_type):
        spn.conf.argmax_zero = True
        spatial_dims = [4, 4]
        nrows, ncols = spatial_dims
        num_vals = 4
        batch_size = 128
        num_vars = spatial_dims[0] * spatial_dims[1]
        indicator_leaf = spn.IndicatorLeaf(num_vars=num_vars, num_vals=num_vals)
        num_sums = 32
        weights = spn.Weights(
            num_weights=num_vals, num_sums=num_sums, initializer=tf.initializers.random_uniform(),
            log=log_weights)

        parsums = []
        for row in range(nrows):
            for col in range(ncols):
                indices = list(range(row * (ncols * num_vals) + col * num_vals,
                                     row * (ncols * num_vals) + (col + 1) * num_vals))
                parsums.append(spn.ParallelSums((indicator_leaf, indices), num_sums=num_sums, weights=weights))

        convsum = spn.ConvSums(
            indicator_leaf, num_channels=num_sums, weights=weights, spatial_dim_sizes=spatial_dims)

        dense_gen = spn.DenseSPNGenerator(
            num_decomps=1, num_mixtures=2, num_subsets=2,
            input_dist=spn.DenseSPNGenerator.InputDist.RAW,
            node_type=spn.DenseSPNGenerator.NodeType.BLOCK)

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
        [p.set_weights(weights) for p in parsums]

        init_conv = spn.initialize_weights(conv_root)
        init_parsum = spn.initialize_weights(parsum_root)

        path_conv = spn.MPEPath(value_inference_type=inference_type)
        path_conv.get_mpe_path(conv_root)

        path_parsum = spn.MPEPath(value_inference_type=inference_type)
        path_parsum.get_mpe_path(parsum_root)

        indicator_leaf_count_parsum = path_parsum.counts[indicator_leaf]
        indicator_leaf_count_convsum = path_conv.counts[indicator_leaf]

        weight_counts_parsum = path_parsum.counts[weights]
        weight_counts_conv = path_conv.counts[weights]

        root_val_parsum = path_parsum.value.values[parsum_root]
        root_val_conv = path_conv.value.values[conv_root]

        parsum_counts = path_parsum.counts[parsum_concat]
        conv_counts = path_conv.counts[convsum]

        indicator_feed = np.random.randint(2, size=batch_size * num_vars)\
            .reshape((batch_size, num_vars))
        with tf.Session() as sess:
            sess.run([init_conv, init_parsum])
            indicator_counts_conv_out, indicator_count_parsum_out = sess.run(
                [indicator_leaf_count_convsum, indicator_leaf_count_parsum], feed_dict={indicator_leaf: indicator_feed})

            root_conv_value_out, root_parsum_value_out = sess.run(
                [root_val_conv, root_val_parsum], feed_dict={indicator_leaf: indicator_feed})

            weight_counts_conv_out, weight_counts_parsum_out = sess.run(
                [weight_counts_conv, weight_counts_parsum], feed_dict={indicator_leaf: indicator_feed})

            weight_value_conv_out, weight_value_parsum_out = sess.run(
                [convsum.weights.node.variable, parsums[0].weights.node.variable])

            parsum_counts_out, conv_counts_out = sess.run(
                [parsum_counts, conv_counts], feed_dict={indicator_leaf: indicator_feed})

            parsum_concat_val, convsum_val = sess.run(
                [path_parsum.value.values[parsum_concat], path_conv.value.values[convsum]],
                feed_dict={indicator_leaf: indicator_feed})

        self.assertTrue(np.all(np.less_equal(convsum_val, 0.0)))
        self.assertTrue(np.all(np.less_equal(parsum_concat_val, 0.0)))
        self.assertAllClose(weight_value_conv_out, weight_value_parsum_out)
        self.assertAllClose(root_conv_value_out, root_parsum_value_out)
        self.assertAllClose(indicator_counts_conv_out, indicator_count_parsum_out)
        self.assertAllClose(parsum_counts_out, conv_counts_out)
        self.assertAllClose(weight_counts_conv_out, weight_counts_parsum_out)

    def test_compute_value(self):
        indicator_leaf = spn.IndicatorLeaf(num_vals=2, num_vars=2 * 2)
        values = [[0, 1, 1, 0],
                  [-1, -1, -1, 0]]
        weights = spn.Weights(
            initializer=tf.initializers.constant([[0.2, 0.8], [0.6, 0.4]]), num_sums=2, num_weights=2)
        s = ConvSums(indicator_leaf, spatial_dim_sizes=[2, 2], num_channels=2, weights=weights)
        
        val = s.get_value(inference_type=spn.InferenceType.MARGINAL)
        
        with self.test_session() as sess:
            sess.run(weights.initialize())
            out = sess.run(val, {indicator_leaf: values})

                                  # 0    0 |  1    1 |  1    1  | 0   0
        self.assertAllClose(out, [[0.2, 0.6, 0.8, 0.4, 0.8, 0.4, 0.2, 0.6],
                                  # 1   0  | 1     0 | 1     0  | 0   0
                                  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.6]])

    def test_compute_scope(self):
        indicator_leaf = spn.IndicatorLeaf(num_vals=2, num_vars=2 * 2)
        weights = spn.Weights(
            initializer=tf.initializers.constant([[0.2, 0.8], [0.6, 0.4]]), num_sums=2, num_weights=2)
        s = ConvSums(indicator_leaf, spatial_dim_sizes=[2, 2], num_channels=2, weights=weights)

        scope = s._compute_scope(None, None, indicator_leaf._compute_scope())

        target_scope = [spn.Scope(indicator_leaf, 0)] * 2 + \
                       [spn.Scope(indicator_leaf, 1)] * 2 + \
                       [spn.Scope(indicator_leaf, 2)] * 2 + \
                       [spn.Scope(indicator_leaf, 3)] * 2
        self.assertAllEqual(scope, target_scope)

    def test_compute_value_conv(self):
        batch_size = 8
        grid_size = 16
        indicator_leaf = spn.IndicatorLeaf(num_vals=2, num_vars=grid_size ** 2)
        convsum = ConvSums(indicator_leaf, spatial_dim_sizes=[grid_size, grid_size], num_channels=4)
        weights = convsum.generate_weights(initializer=tf.initializers.random_uniform())
        values = np.random.randint(2, size=batch_size * grid_size ** 2).reshape((batch_size, -1))
        indicator_leaf_flat_value = indicator_leaf.get_value()
        indicators_spatial = tf.reshape(indicator_leaf_flat_value, (-1, grid_size, grid_size, 2))
        weights_for_conv = tf.reshape(
            tf.transpose(weights.variable), (1, 1, 2, 4))

        conv_layer = tf.layers.Conv2D(filters=4, kernel_size=1, activation=None, use_bias=False)
        conv_layer.build(input_shape=[None, grid_size, grid_size, 2])
        conv_layer.kernel = weights_for_conv

        conv_truth_op = tf.layers.flatten(conv_layer(indicators_spatial))

        conv_op = convsum.get_value()

        with self.test_session() as sess:
            sess.run(weights.initialize())
            conv_truth_out, conv_out = sess.run(
                [conv_truth_op, conv_op], {indicator_leaf: values})

        self.assertAllClose(conv_truth_out, conv_out)

    @argsprod([2, 3])
    def test_compute_value_sum(self, grid_size):
        indicator_leaf = spn.IndicatorLeaf(num_vals=2, num_vars=grid_size ** 2)
        convsum = ConvSums(indicator_leaf, spatial_dim_sizes=[grid_size, grid_size], num_channels=4)
        convsum2 = ConvSums(indicator_leaf, spatial_dim_sizes=[grid_size, grid_size], num_channels=4)
        dense_generator = spn.DenseSPNGenerator(
            num_mixtures=4, num_subsets=4, num_decomps=1, 
            input_dist=spn.DenseSPNGenerator.InputDist.MIXTURE)
        root = dense_generator.generate(convsum, convsum2)
        spn.generate_weights(root, initializer=tf.initializers.random_uniform())
        init = spn.initialize_weights(root)
        
        num_possibilities = 2 ** (grid_size ** 2)
        nums = np.arange(num_possibilities).reshape((num_possibilities, 1))
        powers = 2 ** np.arange(grid_size ** 2).reshape((1, grid_size ** 2))
        indicator_feed = np.bitwise_and(nums, powers) // powers
        
        value_op = spn.LogValue(spn.InferenceType.MARGINAL).get_value(root)
        value_op_sum = tf.reduce_logsumexp(value_op)
        
        with self.test_session() as sess:
            sess.run(init)
            root_sum = sess.run(value_op_sum, feed_dict={indicator_leaf: indicator_feed})

        print(indicator_feed[:10])
        self.assertAllClose(root_sum, 0.0)











