#!/usr/bin/env python3

import unittest
import tensorflow as tf
import numpy as np
from context import libspn as spn
from parameterized import parameterized
import itertools
import libspn as spn


INPUT_SIZES = [[10], [1] * 10, [2] * 5, [5, 5], [1, 2, 3, 4], [4, 3, 2, 1], [2, 1, 2, 3, 1, 1]]
#INPUT_SIZES = [[1] * 10, [2] * 5, [5, 5], [1, 2, 3, 4], [4, 3, 2, 1], [2, 1, 2, 3, 1, 1]]
#SUM_SIZES = [[1] * 10, [2] * 5, [5, 5], [1, 2, 3, 4], [4, 3, 2, 1], [2, 1, 2, 3, 1, 1]]
SUM_SIZES = [[10], [1] * 10, [2] * 5, [5, 5], [1, 2, 3, 4], [4, 3, 2, 1], [2, 1, 2, 3, 1, 1]]
BOOLEAN = [False, True]
INF_TYPES = [spn.InferenceType.MPE, spn.InferenceType.MARGINAL]


def arg_product(*args):
    return [tuple(elem) for elem in itertools.product(*args)]


def sumslayer_mpe_path_numpy(values, indices, weights, latent_indicators, sums_sizes, inf_type, root_weights,
                             value_only=False):
    """ Computes the output of _compute_mpe_path with numpy """
    selected_values = [(val, ind) for val, ind in zip(values, indices)]
    inputs_to_reduce, iv_mask_per_sum, weights_per_sum = sumslayer_numpy_prepare_sums(
        selected_values, latent_indicators, sums_sizes, weights)

    # Compute element-wise weighting
    weighted_sums = [x * np.reshape(w / np.sum(w), (1, -1)) * iv
                     for x, w, iv in zip(inputs_to_reduce, weights_per_sum, iv_mask_per_sum)]

    # Reduce the result
    reduce_fn = np.max if inf_type == spn.InferenceType.MPE else np.sum
    layer_out = np.stack([reduce_fn(ws, axis=1) for ws in weighted_sums], axis=1)

    # Return if only interested in the value
    root_weights /= root_weights.sum()
    if value_only:
        return reduce_fn(layer_out * root_weights, axis=1, keepdims=True)

    # Determine the index of the winning sums
    winning_indices = np.argmax(layer_out * root_weights, axis=1)

    # Now we compute the max per 'winning' sum
    weight_counts = [np.zeros_like(w) for w in weighted_sums]
    for i, winning_ind in enumerate(winning_indices):
        max_ind = np.argmax(weighted_sums[winning_ind][i])
        weight_counts[winning_ind][i, max_ind] = 1

    # We split the count matrix so that each element corresponds to an input
    input_selected_counts = np.split(
        np.concatenate(weight_counts, axis=1),
        np.cumsum([len(ind) for ind in indices])[:-1],
        axis=1
    )
    # Finally, we do a selective assignment of the splitted counts
    input_counts = []
    for inp, winning_ind, counts in zip(values, indices, input_selected_counts):
        input_c = np.zeros_like(inp)
        input_c[:, winning_ind] = counts
        input_counts.append(input_c)

    # The IndicatorLeaf counts is the same as the weight counts!
    latent_indicators_counts = weight_counts
    return weight_counts, latent_indicators_counts, input_counts


def sumslayer_numpy_prepare_sums(inputs, latent_indicators, sums_sizes, weights):
    inputs_selected = []
    # Taking care of indices
    for x, indices in inputs:
        if indices is not None:
            inputs_selected.append(x[:, indices])
        else:
            inputs_selected.append(x)
    # Concatenate and then split based on sums_sizes
    splits = np.cumsum(sums_sizes)[:-1]
    inputs_concatenated = np.concatenate(inputs_selected, axis=1)
    inputs_to_reduce = np.split(inputs_concatenated, splits, axis=1)
    weights_per_sum = np.split(weights, splits)
    iv_mask = build_iv_mask(inputs_selected, inputs_to_reduce, latent_indicators, sums_sizes)
    iv_mask_per_sum = np.split(iv_mask, splits, axis=1)

    return inputs_to_reduce, iv_mask_per_sum, weights_per_sum


def build_iv_mask(inputs_selected, inputs_to_reduce, latent_indicators, sums_sizes):
    """
    Creates concatenated Indicator matrix with boolean values that can be multiplied with the
    reducible values for masking
    """
    if not latent_indicators:
        latent_indicators_ = np.concatenate([np.ones_like(x) for x in inputs_to_reduce], 1)
    else:
        latent_indicators_ = np.ones((inputs_selected[0].shape[0], sum(sums_sizes)))
        for row in range(latent_indicators_.shape[0]):
            offset = 0
            for iv, s in zip(latent_indicators, sums_sizes):
                if 0 <= iv[row] < s:
                    latent_indicators_[row, offset:offset + s] = 0
                    latent_indicators_[row, offset + iv[row]] = 1
                offset += s
    return latent_indicators_


class TestNodesSumsLayer(tf.test.TestCase):

    @parameterized.expand(arg_product(
        INPUT_SIZES, SUM_SIZES, BOOLEAN, [True], BOOLEAN, INF_TYPES, BOOLEAN))
    def test_sumslayer_value(self, input_sizes, sum_sizes, latent_indicators, log, same_inputs, inf_type,
                             indices):
        batch_size = 32
        factor = 10
        # Construct the required inputs
        feed_dict, indices, input_nodes, input_tuples, latent_indicators, values, weights, root_weights = \
            self.sumslayer_prepare_common(
                batch_size, factor, indices, input_sizes, latent_indicators, same_inputs, sum_sizes)

        # Compute true output
        true_out = sumslayer_mpe_path_numpy(
            values, indices, weights, None if not latent_indicators else latent_indicators, sum_sizes, inf_type, root_weights,
            value_only=True)

        # Build graph
        init, latent_indicators_nodes, root, weight_node = self.build_sumslayer_common(
            feed_dict, input_tuples, latent_indicators, sum_sizes, weights, root_weights)

        # Get the desired op
        value_op = root.get_value(inf_type) if log else tf.exp(root.get_log_value(inf_type))

        # Run and assert correct
        with self.test_session() as sess:
            sess.run(init)
            out = sess.run(value_op, feed_dict=feed_dict)
        self.assertAllClose(out, true_out)

    @parameterized.expand(arg_product(
        INPUT_SIZES, SUM_SIZES, BOOLEAN, [True], BOOLEAN, INF_TYPES,
        ['gather', 'segmented'], BOOLEAN, BOOLEAN))
    def test_sumslayer_mpe_path(self, input_sizes, sum_sizes, latent_indicators, log, same_inputs, inf_type,
                                count_strategy, indices, use_unweighted):
        spn.conf.argmax_zero = True

        # Set some defaults
        if (1 in sum_sizes or 1 in input_sizes or np.all(np.equal(sum_sizes, sum_sizes[0]))) \
                and use_unweighted:
            # There is not a clean way to solve the issue avoided here. It has to do with floating
            # point errors in numpy vs. tf, leading to unpredictable behavior of argmax.
            # Unweighted values take away any weighting randomness, so the argmax will obtain some
            # values that are very likely to be equal up to these floating point errors. Hence,
            # we just set use_unweighted to False if the sum size or input size equals 1 (which is
            # typically when the values are 'pseudo'-equal)
            return None

        batch_size = 32
        factor = 10
        # Configure count strategy
        spn.conf.sumslayer_count_sum_strategy = count_strategy
        feed_dict, indices, input_nodes, input_tuples, latent_indicators, values, weights, root_weights = \
            self.sumslayer_prepare_common(
                batch_size, factor, indices, input_sizes, latent_indicators, same_inputs, sum_sizes)

        root_weights_np = np.ones_like(root_weights) if use_unweighted and log else root_weights
        weight_counts, latent_indicators_counts, value_counts = sumslayer_mpe_path_numpy(
            values, indices, weights, None if not latent_indicators else latent_indicators, sum_sizes, inf_type,
            root_weights_np)
        # Build graph
        init, latent_indicators_nodes, root, weight_node = self.build_sumslayer_common(
            feed_dict, input_tuples, latent_indicators, sum_sizes, weights, root_weights)

        # Then build MPE path Ops
        mpe_path_gen = spn.MPEPath(
            value_inference_type=inf_type, log=True, use_unweighted=use_unweighted)
        mpe_path_gen.get_mpe_path(root)
        path_op = [mpe_path_gen.counts[node] for node in [weight_node] + input_nodes + latent_indicators_nodes]

        # Run graph and do some post-processing
        with self.test_session() as sess:
            sess.run(init)
            out = sess.run(path_op, feed_dict=feed_dict)
            if latent_indicators:
                latent_indicators_counts_out = out[-1]
                latent_indicators_counts_out = np.split(latent_indicators_counts_out, indices_or_sections=len(sum_sizes),
                                          axis=1)
                latent_indicators_counts_out = [np.squeeze(iv, axis=1)[:, :size] for iv, size in
                                  zip(latent_indicators_counts_out, sum_sizes)]
                out = out[:-1]
            weight_counts_out, *input_counts_out = out
            weight_counts_out = np.split(weight_counts_out, indices_or_sections=len(sum_sizes),
                                         axis=1)
            weight_counts_out = [np.squeeze(w, axis=1)[:, :size] for w, size in
                                 zip(weight_counts_out, sum_sizes)]
        if same_inputs:
            value_counts = [np.sum(value_counts, axis=0)]

        # Test outputs
        [self.assertAllClose(inp_count_out, inp_count) for inp_count_out, inp_count in
         zip(input_counts_out, value_counts)]
        [self.assertAllClose(w_out, w_out_truth) for w_out, w_out_truth in
         zip(weight_counts_out, weight_counts)]
        if latent_indicators:
            [self.assertAllClose(iv_out, iv_true_out) for iv_out, iv_true_out in
             zip(latent_indicators_counts_out, latent_indicators_counts)]

    def build_sumslayer_common(self, feed_dict, input_tuples, latent_indicators, sum_sizes, weights,
                               root_weights):
        sumslayer = spn.SumsLayer(*input_tuples, num_or_size_sums=sum_sizes)
        if latent_indicators:
            latent_indicators_nodes = [sumslayer.generate_latent_indicators()]
            feed_dict[latent_indicators_nodes[0]] = np.stack(latent_indicators, axis=1)
        else:
            latent_indicators_nodes = []
        mask = sumslayer._build_mask()
        weights_padded = np.zeros(mask.size)
        weights_padded[mask.ravel()] = weights
        weight_node = sumslayer.generate_weights(
            initializer=tf.initializers.constant(weights_padded))
        # Connect a single sum to group outcomes
        root = spn.SumsLayer(sumslayer, num_or_size_sums=1)
        root.generate_weights(initializer=tf.initializers.constant(root_weights))
        init = spn.initialize_weights(root)
        return init, latent_indicators_nodes, root, weight_node

    @staticmethod
    def sumslayer_prepare_common(batch_size, factor, indices, input_sizes, latent_indicators, same_inputs,
                                 sum_sizes):
        if indices:
            indices = [np.random.choice(list(range(size * factor)), size=size, replace=False)
                       for size in input_sizes]
        else:
            factor = 1
            indices = [np.arange(size) for size in input_sizes]
        if not same_inputs:
            input_nodes = [spn.RawLeaf(num_vars=size * factor) for size in input_sizes]
            values = [np.random.rand(batch_size, size * factor) for size in input_sizes]
            input_tuples = [(node, ind.tolist()) for node, ind in zip(input_nodes, indices)]
            feed_dict = {node: val for node, val in zip(input_nodes, values)}
        else:
            input_nodes = [spn.RawLeaf(num_vars=max(input_sizes) * factor)]
            values = [np.random.rand(batch_size, max(input_sizes) * factor)] * len(input_sizes)
            input_tuples = [(input_nodes[0], ind.tolist()) for ind in indices]
            feed_dict = {input_nodes[0]: values[0]}
        if 1 in sum_sizes:
            latent_indicators = False
        if latent_indicators:
            latent_indicators = [np.random.randint(size, size=batch_size) for size in sum_sizes]
        weights = np.random.rand(sum(sum_sizes))
        root_weights = np.random.rand(len(sum_sizes))
        return feed_dict, indices, input_nodes, input_tuples, latent_indicators, values, weights, root_weights

    def tearDown(self):
        tf.reset_default_graph()

    def test_compute_scope(self):
        """Calculating scope of Sums"""
        # Create a graph
        v12 = spn.IndicatorLeaf(num_vars=2, num_vals=4, name="V12")
        v34 = spn.RawLeaf(num_vars=3, name="V34")

        scopes_per_node = {
            v12: [spn.Scope(v12, 0), spn.Scope(v12, 0), spn.Scope(v12, 0), spn.Scope(v12, 0),
                  spn.Scope(v12, 1), spn.Scope(v12, 1), spn.Scope(v12, 1), spn.Scope(v12, 1)],
            v34: [spn.Scope(v34, 0), spn.Scope(v34, 1), spn.Scope(v34, 2)]
        }

        def generate_scopes_from_inputs(node, inputs, num_or_size_sums, latent_indicators=False):
            # Create a flat list of scopes, where the scope elements of a single input
            # node are subsequent in the list
            flat_scopes = []
            size = 0
            for inp in inputs:
                if isinstance(inp, tuple) and inp[1]:
                    input_indices = [inp[1]] if isinstance(inp[1], int) else inp[1]
                    for i in input_indices:
                        flat_scopes.append(scopes_per_node[inp[0]][i])
                    size += len(input_indices)
                elif not isinstance(inp, tuple):
                    flat_scopes.extend(scopes_per_node[inp])
                    size += len(scopes_per_node[inp])
                else:
                    flat_scopes.extend(scopes_per_node[inp[0]])
                    size += len(scopes_per_node[inp[0]])
            if isinstance(num_or_size_sums, int):
                num_or_size_sums = num_or_size_sums * [size // num_or_size_sums]

            new_scope = []
            offset = 0
            # For each sum generate the scope based on its size
            for i, s in enumerate(num_or_size_sums):
                scope = flat_scopes[offset]
                for j in range(1, s):
                    scope |= flat_scopes[j + offset]
                offset += s
                if latent_indicators:
                    scope |= spn.Scope(node.latent_indicators.node, i)
                new_scope.append(scope)
            scopes_per_node[node] = new_scope

        def sums_layer_and_test(inputs, num_or_size_sums, name, latent_indicators=False):
            """ Create a sums layer, generate its correct scope and test """
            sums_layer = spn.SumsLayer(*inputs, num_or_size_sums=num_or_size_sums, name=name)
            if latent_indicators:
                sums_layer.generate_latent_indicators()
            generate_scopes_from_inputs(sums_layer, inputs, num_or_size_sums, latent_indicators=latent_indicators)
            self.assertListEqual(sums_layer.get_scope(), scopes_per_node[sums_layer])
            return sums_layer

        def concat_layer_and_test(inputs, name):
            """ Create a concat node, generate its scopes and assert whether it is correct """
            scope = []
            for inp in inputs:
                if isinstance(inp, tuple):
                    indices = inp[1]
                    if isinstance(inp[1], int):
                        indices = [inp[1]]
                    for i in indices:
                        scope.append(scopes_per_node[inp[0]][i])
                else:
                    scope.extend(scopes_per_node[inp])
            concat = spn.Concat(*inputs, name=name)
            self.assertListEqual(concat.get_scope(), scope)
            scopes_per_node[concat] = scope
            return concat

        ss1 = sums_layer_and_test(
            [(v12, [0, 1, 2, 3]), (v12, [1, 2, 5, 6]), (v12, [4, 5, 6, 7])], 3, "Ss1", latent_indicators=True)

        ss2 = sums_layer_and_test([(v12, [6, 7]), (v34, 0)], num_or_size_sums=[1, 2], name="Ss2")
        ss3 = sums_layer_and_test([(v12, [3, 7]), (v34, 1), (v12, [4, 5, 6]), v34],
                                  num_or_size_sums=[1, 2, 2, 2, 2], name="Ss3")

        s1 = sums_layer_and_test([(v34, [1, 2])], num_or_size_sums=1, name="S1", latent_indicators=True)
        concat_layer_and_test([(ss1, [0, 2]), (ss2, 0)], name="N1")
        concat_layer_and_test([(ss1, 1), ss3, s1], name="N2")
        n = concat_layer_and_test([(ss1, 0), ss2, (ss3, [0, 1]), s1], name="N3")
        sums_layer_and_test([(ss1, [1, 2]), ss2], num_or_size_sums=[2, 1, 1], name="Ss4")
        sums_layer_and_test([(ss1, [0, 2]), (n, [0, 1]), (ss3, [4, 2])],
                            num_or_size_sums=[3, 2, 1], name="Ss5")

    def test_compute_valid(self):
        """Calculating validity of Sums"""
        # Without IndicatorLeaf
        v12 = spn.IndicatorLeaf(num_vars=2, num_vals=4)
        v34 = spn.RawLeaf(num_vars=2)
        s1 = spn.SumsLayer((v12, [0, 1, 2, 3]), (v12, [0, 1, 2, 3]),
                           (v12, [0, 1, 2, 3]), num_or_size_sums=3)
        self.assertTrue(s1.is_valid())

        s2 = spn.SumsLayer((v12, [0, 1, 2, 4]), name="S2")
        s2b = spn.SumsLayer((v12, [0, 1, 2, 4]), num_or_size_sums=[3, 1], name="S2b")
        self.assertTrue(s2b.is_valid())
        self.assertFalse(s2.is_valid())

        s3 = spn.SumsLayer((v12, [0, 1, 2, 3]), (v34, 0), (v12, [0, 1, 2, 3]),
                           (v34, 0), num_or_size_sums=2)
        s3b = spn.SumsLayer((v12, [0, 1, 2, 3]), (v34, 0), (v12, [0, 1, 2, 3]),
                            (v34, 0), num_or_size_sums=[4, 1, 4, 1])
        s3c = spn.SumsLayer((v12, [0, 1, 2, 3]), (v34, 0), (v12, [0, 1, 2, 3]),
                            (v34, 0), num_or_size_sums=[4, 1, 5])
        self.assertFalse(s3.is_valid())
        self.assertTrue(s3b.is_valid())
        self.assertFalse(s3c.is_valid())

        p1 = spn.Product((v12, [0, 5]), (v34, 0))
        p2 = spn.Product((v12, [1, 6]), (v34, 0))
        p3 = spn.Product((v12, [1, 6]), (v34, 1))

        s4 = spn.SumsLayer(p1, p2, p1, p2, num_or_size_sums=2)
        s5 = spn.SumsLayer(p1, p3, p1, p3, p1, p3, num_or_size_sums=3)
        s6 = spn.SumsLayer(p1, p2, p3, num_or_size_sums=[2, 1])
        s7 = spn.SumsLayer(p1, p2, p3, num_or_size_sums=[1, 2])
        s8 = spn.SumsLayer(p1, p2, p3, p2, p1, num_or_size_sums=[2, 1, 2])
        self.assertTrue(s4.is_valid())
        self.assertFalse(s5.is_valid())  # p1 and p3 different scopes
        self.assertTrue(s6.is_valid())
        self.assertFalse(s7.is_valid())  # p2 and p3 different scopes
        self.assertTrue(s8.is_valid())
        # With IVS
        s6 = spn.SumsLayer(p1, p2, p1, p2, p1, p2, num_or_size_sums=3)
        s6.generate_latent_indicators()
        self.assertTrue(s6.is_valid())

        s7 = spn.SumsLayer(p1, p2, num_or_size_sums=1)
        s7.set_latent_indicators(spn.RawLeaf(num_vars=2))
        self.assertFalse(s7.is_valid())

        s7 = spn.SumsLayer(p1, p2, p3, num_or_size_sums=3)
        s7.set_latent_indicators(spn.RawLeaf(num_vars=3))
        self.assertTrue(s7.is_valid())

        s7 = spn.SumsLayer(p1, p2, p3, num_or_size_sums=[2, 1])
        s7.set_latent_indicators(spn.RawLeaf(num_vars=3))
        self.assertFalse(s7.is_valid())

        s8 = spn.SumsLayer(p1, p2, p1, p2, num_or_size_sums=2)
        s8.set_latent_indicators(spn.IndicatorLeaf(num_vars=3, num_vals=2))
        with self.assertRaises(spn.StructureError):
            s8.is_valid()

        s9 = spn.SumsLayer(p1, p2, p1, p2, num_or_size_sums=[1, 3])
        s9.set_latent_indicators(spn.RawLeaf(num_vars=2))
        with self.assertRaises(spn.StructureError):
            s9.is_valid()

        s9 = spn.SumsLayer(p1, p2, p1, p2, num_or_size_sums=[1, 3])
        s9.set_latent_indicators(spn.RawLeaf(num_vars=3))
        with self.assertRaises(spn.StructureError):
            s9.is_valid()

        s9 = spn.SumsLayer(p1, p2, p1, p2, num_or_size_sums=2)
        s9.set_latent_indicators(spn.IndicatorLeaf(num_vars=1, num_vals=4))
        self.assertTrue(s9.is_valid())

        s9 = spn.SumsLayer(p1, p2, p1, p2, num_or_size_sums=[1, 3])
        s9.set_latent_indicators(spn.IndicatorLeaf(num_vars=1, num_vals=4))
        self.assertTrue(s9.is_valid())

        s9 = spn.SumsLayer(p1, p2, p1, p2, num_or_size_sums=[1, 3])
        s9.set_latent_indicators(spn.IndicatorLeaf(num_vars=2, num_vals=2))
        self.assertFalse(s9.is_valid())

        s9 = spn.SumsLayer(p1, p2, p1, p2, num_or_size_sums=2)
        s9.set_latent_indicators(spn.IndicatorLeaf(num_vars=2, num_vals=2))
        self.assertTrue(s9.is_valid())

        s9 = spn.SumsLayer(p1, p2, p1, p2, num_or_size_sums=[1, 2, 1])
        s9.set_latent_indicators(spn.IndicatorLeaf(num_vars=2, num_vals=2))
        self.assertFalse(s9.is_valid())

        s10 = spn.SumsLayer(p1, p2, p1, p2, num_or_size_sums=2)
        s10.set_latent_indicators((v12, [0, 3, 5, 7]))
        self.assertTrue(s10.is_valid())

        s10 = spn.SumsLayer(p1, p2, p1, p2, num_or_size_sums=[1, 2, 1])
        s10.set_latent_indicators((v12, [0, 3, 5, 7]))
        self.assertFalse(s10.is_valid())

    def test_masked_weights(self):
        v12 = spn.IndicatorLeaf(num_vars=2, num_vals=4)
        v34 = spn.RawLeaf(num_vars=2)
        v5 = spn.RawLeaf(num_vars=1)
        s = spn.SumsLayer((v12, [0, 5]), v34, (v12, [3]), v5, (v12, [0, 5]), v34,
                          (v12, [3]), v5, num_or_size_sums=[3, 1, 3, 4, 1])
        s.generate_weights(initializer=tf.initializers.random_uniform(0.0, 1.0))
        with self.test_session() as sess:
            sess.run(s.weights.node.initialize())
            weights = sess.run(s.weights.node.variable)

        shape = [5, 4]
        self.assertEqual(shape, s.weights.node.variable.shape.as_list())
        [self.assertEqual(weights[row, col], 0.0) for row, col in
         [(0, -1), (1, 1), (1, 2), (1, 3), (2, -1), (4, 1), (4, 2), (4, 3)]]
        self.assertAllClose(np.sum(weights, axis=1), np.ones(5))


if __name__ == '__main__':
    unittest.main()
