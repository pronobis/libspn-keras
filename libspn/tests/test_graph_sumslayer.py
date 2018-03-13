#!/usr/bin/env python3

# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

import unittest
import tensorflow as tf
import numpy as np
from context import libspn as spn
from parameterized import parameterized
import itertools
import functools
import operator


INPUT_SIZES = [[10], [1] * 10, [2] * 5, [5, 5], [1, 2, 3, 4], [4, 3, 2, 1], [2, 1, 2, 3, 1, 1]]
SUM_SIZES = [[1] * 10, [2] * 5, [5, 5], [1, 2, 3, 4], [4, 3, 2, 1], [2, 1, 2, 3, 1, 1]]


def x_dot_w(xs, ws):
    return [np.asarray(x).dot(np.asarray(w) / np.sum(w)) for x, w in zip(xs, ws)]


def sums_layer_mpe_path_numpy(inputs, sums_sizes, weights, counts, ivs=None):
    """ Computes the output of _compute_mpe_path with numpy """
    inputs_to_reduce, iv_mask_per_sum, weights_per_sum = sums_layer_numpy_common(
        inputs, ivs, sums_sizes, weights)
    # Get max index for sum node
    max_indices = [np.argmax(x * np.reshape(w/np.sum(w), (1, -1)) * iv, axis=1)
                   for x, w, iv in zip(inputs_to_reduce, weights_per_sum, iv_mask_per_sum)]

    counts_per_sum = []
    for i, (mi, size) in enumerate(zip(max_indices, sums_sizes)):
        # Initialize the counts
        cnt = np.zeros((mi.shape[0], size))
        cnt[range(cnt.shape[0]), mi] = counts[:, i]
        counts_per_sum.append(cnt)

    # Go from counts per sum to counts per input
    counts_concatenated = np.concatenate(counts_per_sum, axis=1)
    indices = np.cumsum([len(t[1]) if t[1] else t[0].shape[1] for t in inputs])[:-1]
    counts_per_input = np.split(counts_concatenated, indices, axis=1)

    # 'Scatter' the values to the indices given by each second tuple element in inputs if not None
    # otherwise, the indices are just [0, ..., size-1]
    scattered = []
    for c, inp in zip(counts_per_input, inputs):
        s = np.zeros_like(inp[0])
        if inp[1]:
            s[:, inp[1]] = c
        else:
            s = c
        scattered.append(s)
    return scattered


def sums_layer_value_numpy(inputs, sums_sizes, weights, ivs=None,
                           inference_type=spn.InferenceType.MARGINAL):
    """ Computes value of SumsLayer using numpy """
    inputs_to_reduce, iv_mask_per_sum, weights_per_sum = sums_layer_numpy_common(
        inputs, ivs, sums_sizes, weights)

    # Finally, we reduce each sum node and concatenate the results
    reduce_fn = np.sum if inference_type == spn.InferenceType.MARGINAL else np.max
    return np.concatenate(
        [reduce_fn(x * np.reshape(w/np.sum(w), (1, -1)) * iv, axis=1, keepdims=True)
         for x, w, iv in zip(inputs_to_reduce, weights_per_sum, iv_mask_per_sum)], axis=1
    )


def sums_layer_numpy_common(inputs, ivs, sums_sizes, weights):
    inputs_selected = []
    # Taking care of indices
    for x, indices in inputs:
        if indices:
            inputs_selected.append(x[:, indices])
        else:
            inputs_selected.append(x)
    # Concatenate and then split based on sums_sizes
    splits = np.cumsum(sums_sizes)[:-1]
    inputs_concatenated = np.concatenate(inputs_selected, axis=1)
    inputs_to_reduce = np.split(inputs_concatenated, splits, axis=1)
    weights_per_sum = np.split(weights, splits)
    iv_mask = build_iv_mask(inputs_selected, inputs_to_reduce, ivs, sums_sizes)
    iv_mask_per_sum = np.split(iv_mask, splits, axis=1)

    return inputs_to_reduce, iv_mask_per_sum, weights_per_sum


def build_iv_mask(inputs_selected, inputs_to_reduce, ivs, sums_sizes):
    """
    Creates concatenated IV matrix with boolean values that can be multiplied with the
    reducible values for masking
    """
    if not ivs:
        ivs_ = np.concatenate([np.ones_like(x) for x in inputs_to_reduce], 1)
    else:
        ivs_ = np.ones((inputs_selected[0].shape[0], sum(sums_sizes)))
        for row in range(ivs_.shape[0]):
            offset = 0
            for iv, s in zip(ivs, sums_sizes):
                if 0 <= iv[row] < s:
                    ivs_[row, offset:offset + s] = 0
                    ivs_[row, offset + iv[row]] = 1
                offset += s
    return ivs_


class TestNodesSumsLayer(tf.test.TestCase):

    @parameterized.expand([
        (input_sizes, sum_sizes) for input_sizes, sum_sizes in
        itertools.product(INPUT_SIZES, SUM_SIZES)
    ])
    def test_sums_layer_numpy(self, input_sizes, sum_sizes):
        """ Tests numpy SumsLayer helper """
        batch_size = 32
        total_size = np.sum(input_sizes)
        x = np.random.rand(batch_size, total_size)
        w = np.ones(total_size)

        inputs = []
        outputs = []
        offset = 0
        for s in sum_sizes:
            inputs.append((x[:, offset:offset + s], None))
            outputs.append(np.mean(x[:, offset:offset + s], axis=1, keepdims=True))
            offset += s

        np.testing.assert_allclose(sums_layer_value_numpy(inputs, sum_sizes, w),
                                   np.concatenate(outputs, axis=1))

    @parameterized.expand([
        (input_sizes, sum_sizes) for input_sizes, sum_sizes in
        itertools.product(INPUT_SIZES, SUM_SIZES)
    ])
    def test_sums_layer_with_indices_numpy(self, input_sizes, sum_sizes):
        """ Test the numpy SumsLayer helper with indices for each input """
        batch_size = 32
        total_size = np.sum(input_sizes)
        x = np.random.rand(batch_size, total_size * 5)
        w = np.ones(total_size)

        inputs = []
        outputs = []
        offset = 0
        for s in sum_sizes:
            ind = list(np.random.choice(5 * s, s, replace=False))
            inputs.append((x[:, offset:offset + s * 5], ind))
            outputs.append(np.mean(inputs[-1][0][:, ind], axis=1, keepdims=True))
            offset += s * 5

        np.testing.assert_allclose(sums_layer_value_numpy(inputs, sum_sizes, w),
                                   np.concatenate(outputs, axis=1))

    @parameterized.expand([
        (input_sizes, sum_sizes) for input_sizes, sum_sizes in
        itertools.product(INPUT_SIZES, SUM_SIZES)
    ])
    def test_sums_layer_with_indices_ivs_numpy(self, input_sizes, sum_sizes):
        """ Tests the numpy helper with indices per input and IVs """
        batch_size = 32
        total_size = np.sum(input_sizes)
        x = np.random.rand(batch_size, total_size * 5)
        w = np.ones(total_size)

        inputs = []
        outputs = []
        ivs = []
        offset = 0
        for s in sum_sizes:
            ind = list(np.random.choice(5 * s, s, replace=False))
            iv = np.random.choice(s + 1, batch_size) - 1

            inputs.append((x[:, offset:offset + s * 5], ind))
            selected = inputs[-1][0][:, ind]

            iv_m = []
            for iv_ in iv:
                iv_m.append(np.ones(s) if iv_ == -1 else [0 if i != iv_ else 1 for i in range(s)])
            ivs.append(iv)
            outputs.append(np.mean(selected * np.stack(iv_m), axis=1, keepdims=True))
            offset += s * 5

        np.testing.assert_allclose(sums_layer_value_numpy(inputs, sum_sizes, w, ivs),
                                   np.concatenate(outputs, axis=1))

    @parameterized.expand([
        (input_sizes, sum_sizes, ivs, inference_type, log, same_input) for
        input_sizes, sum_sizes, ivs, inference_type, log, same_input in
        itertools.product(
            INPUT_SIZES, SUM_SIZES, [False, True],
            [spn.InferenceType.MARGINAL, spn.InferenceType.MPE], [False, True], [False, True])
    ])
    def test_sumslayer_varying_sizes_values(self, input_sizes, sum_sizes, ivs, inference_type,
                                            log, same_input):
        """
        Tests the SumsLayer value computation for different input sizes, sum sizes, inference types,
        with and without IVs, and in log and non-log space.
        """
        # Initialize dimensions
        batch_size = 256
        total_size = np.sum(input_sizes)
        fac = 5

        # Generate random arrays
        x = np.random.rand(batch_size, total_size * fac)
        w = np.random.rand(total_size)

        # Get feed dict, IV numeric value list, an IV node, the numeric inputs and the SPN input
        # tuples
        feed_dict, ivs_list, ivs_node, numpy_inputs, spn_inputs = self.sumslayer_common(
            batch_size, fac, x, ivs, input_sizes, sum_sizes, same_input=same_input)

        # Build SumsLayer
        n = spn.SumsLayer(*spn_inputs, num_sums_or_sizes=sum_sizes, ivs=ivs_node)
        weight_node = n.generate_weights(w)

        with self.test_session() as sess:
            # Run required op
            spn.initialize_weights(weight_node).run()
            if log:
                op = tf.exp(n.get_log_value(inference_type))
            else:
                op = n.get_value(inference_type)
            out = sess.run(op, feed_dict=feed_dict)

        # Get true output using numpy
        true_out = sums_layer_value_numpy(numpy_inputs, sum_sizes, w, ivs_list,
                                          inference_type=inference_type)
        self.assertAllClose(true_out, out)

    @parameterized.expand([
        (input_sizes, sum_sizes, ivs, log, same_input, count_matmul) for
        input_sizes, sum_sizes, ivs, log, same_input, count_matmul in
        itertools.product(
            INPUT_SIZES, SUM_SIZES, [False, True], [False, True], [True], ["gather", 'matmul'])
    ])
    def test_sumslayer_varying_sizes_mpe_path(self, input_sizes, sum_sizes, ivs, log, same_input,
                                              count_matmul_strategy):
        """
        Tests the SumsLayer MPE path computation for different input sizes, sum sizes, with and
        without IVs, and in log and non-log space
        """
        # Initialize dimensions
        batch_size = 2
        total_size = np.sum(input_sizes)
        fac = 2

        spn.conf.sumslayer_count_sum_strategy = count_matmul_strategy

        # Generate random arrays
        x = np.random.rand(batch_size, total_size * fac)
        w = np.random.rand(total_size)
        counts = np.arange(batch_size * len(sum_sizes)).reshape((batch_size, len(sum_sizes))) + 10

        # Get feed dict, IV numeric value list, an IV node, the numeric inputs and the SPN input
        # tuples
        feed_dict, ivs_list, ivs_node, numpy_inputs, spn_inputs = self.sumslayer_common(
            batch_size, fac, x, ivs, input_sizes, sum_sizes, same_input=same_input)

        # Compute the true output using numpy
        true_outs = sums_layer_mpe_path_numpy(numpy_inputs, sum_sizes, w, counts,
                                              ivs=ivs_list)
        if same_input:
            true_outs = [functools.reduce(operator.add, true_outs)]

        # Build SumsLayer
        n = spn.SumsLayer(*spn_inputs, num_sums_or_sizes=sum_sizes, ivs=ivs_node)
        weight_node = n.generate_weights(w)

        # Add counts
        counts_ph = tf.placeholder(tf.float32, shape=counts.shape)
        feed_dict[counts_ph] = counts

        with self.test_session() as sess:
            # Run required op
            spn.initialize_weights(weight_node).run()
            if log:
                # Get the value ops and log mpe path op
                w_val = weight_node.get_log_value()
                iv_val = ivs_node.get_log_value() if ivs_node else None
                value_input_vals = [inp[0].get_log_value() for inp in spn_inputs]
                op = n._compute_log_mpe_path(
                    tf.identity(counts_ph), w_val, iv_val, *value_input_vals
                )
            else:
                # Get the value ops and mpe path op
                w_val = weight_node.get_value()
                iv_val = ivs_node.get_value() if ivs_node else None
                value_input_vals = [inp[0].get_value() for inp in spn_inputs]
                print(w_val, value_input_vals)
                op = n._compute_mpe_path(
                    tf.identity(counts_ph), w_val, iv_val, *value_input_vals
                )
            print(op)
            op = [o for o in op if o is not None]
            out = sess.run(op, feed_dict=feed_dict)[2 if ivs_node else 1:]

        for o, to in zip(out, true_outs):
            self.assertAllClose(o, to)

    @staticmethod
    def sumslayer_common(batch_size, fac, x, ivs, input_sizes, sum_sizes, same_input=False):
        # Lists that will hold nodes or values
        numpy_inputs = []
        ivs_list = []
        spn_inputs = []
        feed_dict = {}
        # Offset to use for determining
        offset = 0
        node = None
        for s in input_sizes:
            slice_size = s * fac

            # Pick some random indices
            ind = [int(i) for i in np.random.choice(slice_size, s, replace=False)]
            if not same_input:
                # Append the value-indices tuple to numpy_inputs
                value = x[:, offset:offset + slice_size]
                numpy_inputs.append((value, ind))
                # Append the node-indices tuple spn_inputs
                node = spn.ContVars(num_vars=slice_size)
                spn_inputs.append((node, ind))

                # Update feed_dict
                feed_dict[node] = value

                # Update offset
                offset += slice_size
            else:
                max_size = max(input_sizes) * fac
                value = x[:, :max_size]
                numpy_inputs.append((value, ind))
                if node is None:
                    node = spn.ContVars(num_vars=max_size)
                spn_inputs.append((node, ind))

                # Update feed_dict
                feed_dict[node] = value

        if ivs:
            for s in sum_sizes:
                # Create some random IVs
                iv = np.random.choice(s + 1, batch_size) - 1
                ivs_list.append(iv)

        # Add IV node to the graph
        ivs_node = None
        if ivs and max(sum_sizes) > 1:
            ivs_node = spn.IVs(num_vars=len(sum_sizes), num_vals=max(sum_sizes))
            feed_dict[ivs_node] = np.stack(ivs_list, axis=1)
        return feed_dict, ivs_list, ivs_node, numpy_inputs, spn_inputs

    def tearDown(self):
        tf.reset_default_graph()

    def test_compute_marginal_value_varsize(self):
        """Calculating marginal value of Sums"""
        def test(values, num_sums, ivs, weights, feed, output):
            with self.subTest(values=values, num_sums=num_sums, ivs=ivs, weights=weights,
                              feed=feed):
                n = spn.SumsLayer(*values, num_sums_or_sizes=num_sums, ivs=ivs)
                w = n.generate_weights(weights)
                op = n.get_value(spn.InferenceType.MARGINAL)
                op_log = n.get_log_value(spn.InferenceType.MARGINAL)

                with tf.Session() as sess:
                    spn.initialize_weights(w).run()
                    out = sess.run(op, feed_dict=feed)
                    out_log = sess.run(tf.exp(op_log), feed_dict=feed)

                np.testing.assert_array_almost_equal(
                    out,
                    np.array(output, dtype=spn.conf.dtype.as_numpy_dtype()))
                np.testing.assert_array_almost_equal(
                    out_log,
                    np.array(output, dtype=spn.conf.dtype.as_numpy_dtype()))

        # MULTIPLE SUM NODES
        # -------------------
        num_sums = 2
        ivs = spn.IVs(num_vars=num_sums, num_vals=5)

        # Create inputs
        v1 = spn.ContVars(num_vars=4, name="ContVars1")
        v2 = spn.ContVars(num_vars=4, name="ContVars2")

        v3 = spn.ContVars(num_vars=8, name="ContVars3")
        v4 = spn.ContVars(num_vars=8, name="ContVars4")

        # Multiple inputs, multi-element batch
        test([v1, v2],
             [3, 5],
             None,
             [[0.2, 0.2, 0.3, 0.3, 0.1, 0.2, 0.3, 0.4]],
             {v1: [[0.1, 0.2, 0.3, 0.4],
                   [0.11, 0.12, 0.13, 0.14]],
              v2: [[0.5, 0.6, 0.7, 0.8],
                   [0.15, 0.16, 0.17, 0.18]]},
             [x_dot_w([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6, 0.7, 0.8]],
                      [[0.2, 0.2, 0.3], [0.3, 0.1, 0.2, 0.3, 0.4]]),
              x_dot_w([[0.11, 0.12, 0.13], [0.14, 0.15, 0.16, 0.17, 0.18]],
                      [[0.2, 0.2, 0.3], [0.3, 0.1, 0.2, 0.3, 0.4]])])

        test([(v1, [1, 2]), (v2, [0, 3])],
             [1, 3],
             None,
             [[0.4, 0.6, 0.2, 0.8]],
             {v1: [[0.1, 0.2, 0.3, 0.4],
                   [0.11, 0.12, 0.13, 0.14]],
              v2: [[0.5, 0.6, 0.7, 0.8],
                   [0.15, 0.16, 0.17, 0.18]]},
             [x_dot_w([[0.2], [0.3, 0.5, 0.8]],
                      [[0.4], [0.6, 0.2, 0.8]]),
              x_dot_w([[0.12], [0.13, 0.15, 0.18]],
                      [[0.4], [0.6, 0.2, 0.8]])])

        test([v1, v2],
             [3, 5],
             ivs,
             [[0.2, 0.2, 0.3, 0.3, 0.1, 0.2, 0.3, 0.4]],
             {v1: [[0.1, 0.2, 0.3, 0.4],
                   [0.11, 0.12, 0.13, 0.14]],
              v2: [[0.5, 0.6, 0.7, 0.8],
                   [0.15, 0.16, 0.17, 0.18]],
              ivs: [[-1, -1],
                    [-1, -1]]},
             [x_dot_w([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6, 0.7, 0.8]],
                      [[0.2, 0.2, 0.3], [0.3, 0.1, 0.2, 0.3, 0.4]]),
              x_dot_w([[0.11, 0.12, 0.13], [0.14, 0.15, 0.16, 0.17, 0.18]],
                      [[0.2, 0.2, 0.3], [0.3, 0.1, 0.2, 0.3, 0.4]])])

        test([v1, v2],
             [5, 3],
             ivs,
             [[0.2, 0.2, 0.3, 0.3, 0.1, 0.2, 0.3, 0.4]],
             {v1: [[0.1, 0.2, 0.3, 0.4],
                   [0.11, 0.12, 0.13, 0.14]],
              v2: [[0.5, 0.6, 0.7, 0.8],
                   [0.15, 0.16, 0.17, 0.18]],
              ivs: [[-1, 2],
                    [1, -1]]},
             [x_dot_w([[0.1, 0.2, 0.3, 0.4, 0.5], [0, 0, 0.8]],
                      [[0.2, 0.2, 0.3, 0.3, 0.1], [0.2, 0.3, 0.4]]),
              x_dot_w([[0.0, 0.12, 0.0, 0.0, 0.0], [0.16, 0.17, 0.18]],
                      [[0.2, 0.2, 0.3, 0.3, 0.1], [0.2, 0.3, 0.4]])])

        test([(v3, [7, 5, 3, 1]), (v4, [0, 2, 4, 6])],
             [3, 5],
             ivs,
             [[0.2, 0.2, 0.3, 0.3, 0.1, 0.2, 0.3, 0.4]],
             {v3: [[0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07],
                   [0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17]],
              v4: [[0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27],
                   [0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37]],
              ivs: [[1, 0],
                    [0, 1]]},
             [[(0.05*0.2/0.7), (0.01*0.3/1.3)],
              [(0.17*0.2/0.7), (0.3*0.1/1.3)]])

        # Single input with 1 value, multi-element batch
        test([v3],
             [2, 6],
             None,
             [[0.2, 0.2, 0.3, 0.3, 0.1, 0.2, 0.3, 0.4]],
             {v3: [[0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07],
                   [0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17]]},
             [x_dot_w([[0.0, 0.01], [0.02, 0.03, 0.04, 0.05, 0.06, 0.07]],
                      [[0.2, 0.2], [0.3, 0.3, 0.1, 0.2, 0.3, 0.4]]),
              x_dot_w([[0.1, 0.11], [0.12, 0.13, 0.14, 0.15, 0.16, 0.17]],
                      [[0.2, 0.2], [0.3, 0.3, 0.1, 0.2, 0.3, 0.4]])])
        test([v3],
             [3, 5],
             ivs,
             [0.2, 0.2, 0.3, 0.3, 0.1, 0.2, 0.3, 0.4],
             {v3: [[0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07],
                   [0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17]],
              ivs: [[-1, 3],
                    [2, -1]]},
             [[(0.0*0.2 + 0.01*0.2 + 0.02*0.3) / 0.7, (0.06*0.3) / 1.3],
              [(0.12*0.3/0.7), (0.3 * 0.13 + 0.14*0.1 + 0.15*0.2 + 0.16*0.3 + 0.17*0.4)/1.3]])

    def test_compute_marginal_value(self):
        """Calculating marginal value of Sums"""

        def test(values, num_sums, ivs, weights, feed, output):
            with self.subTest(values=values, num_sums=num_sums, ivs=ivs, weights=weights,
                              feed=feed):
                n = spn.SumsLayer(*values, num_sums_or_sizes=num_sums, ivs=ivs)
                n.generate_weights(weights)
                op = n.get_value(spn.InferenceType.MARGINAL)
                op_log = n.get_log_value(spn.InferenceType.MARGINAL)
                with tf.Session() as sess:
                    spn.initialize_weights(n).run()
                    out = sess.run(op, feed_dict=feed)
                    out_log = sess.run(tf.exp(op_log), feed_dict=feed)

                np.testing.assert_array_almost_equal(
                    out,
                    np.array(output, dtype=spn.conf.dtype.as_numpy_dtype()))
                np.testing.assert_array_almost_equal(
                    out_log,
                    np.array(output, dtype=spn.conf.dtype.as_numpy_dtype()))

        # MULTIPLE SUM NODES
        # -------------------
        num_sums = 2
        ivs = spn.IVs(num_vars=num_sums, num_vals=4)

        # Create inputs
        v1 = spn.ContVars(num_vars=4, name="ContVars1")
        v2 = spn.ContVars(num_vars=4, name="ContVars2")

        v3 = spn.ContVars(num_vars=8, name="ContVars3")
        v4 = spn.ContVars(num_vars=8, name="ContVars4")

        # Multiple inputs, multi-element batch
        test([v1, v2],
             num_sums,
             None,
             [0.2, 0.2, 0.3, 0.3, 0.1, 0.2, 0.3, 0.4],
             {v1: [[0.1, 0.2, 0.3, 0.4],
                   [0.11, 0.12, 0.13, 0.14]],
              v2: [[0.5, 0.6, 0.7, 0.8],
                   [0.15, 0.16, 0.17, 0.18]]},
             [[(0.1*0.2 + 0.2*0.2 + 0.3*0.3 + 0.4*0.3),
               (0.5*0.1 + 0.6*0.2 + 0.7*0.3 + 0.8*0.4)],
              [(0.11*0.2 + 0.12*0.2 + 0.13*0.3 + 0.14*0.3),
               (0.15*0.1 + 0.16*0.2 + 0.17*0.3 + 0.18*0.4)]])

        test([(v1, [1, 2]), (v2, [0, 3])],
             num_sums,
             None,
             [0.4, 0.6, 0.2, 0.8],
             {v1: [[0.1, 0.2, 0.3, 0.4],
                   [0.11, 0.12, 0.13, 0.14]],
              v2: [[0.5, 0.6, 0.7, 0.8],
                   [0.15, 0.16, 0.17, 0.18]]},
             [[(0.2*0.4 + 0.3*0.6), (0.5*0.2 + 0.8*0.8)],
              [(0.12*0.4 + 0.13*0.6), (0.15*0.2 + 0.18*0.8)]])

        test([v1, v2],
             num_sums,
             ivs,
             [0.2, 0.2, 0.3, 0.3, 0.1, 0.2, 0.3, 0.4],
             {v1: [[0.1, 0.2, 0.3, 0.4],
                   [0.11, 0.12, 0.13, 0.14]],
              v2: [[0.5, 0.6, 0.7, 0.8],
                   [0.15, 0.16, 0.17, 0.18]],
              ivs: [[-1, -1],
                    [-1, -1]]},
             [[(0.1*0.2 + 0.2*0.2 + 0.3*0.3 + 0.4*0.3),
               (0.5*0.1 + 0.6*0.2 + 0.7*0.3 + 0.8*0.4)],
              [(0.11*0.2 + 0.12*0.2 + 0.13*0.3 + 0.14*0.3),
               (0.15*0.1 + 0.16*0.2 + 0.17*0.3 + 0.18*0.4)]])

        test([v1, v2],
             num_sums,
             ivs,
             [0.2, 0.2, 0.3, 0.3, 0.1, 0.2, 0.3, 0.4],
             {v1: [[0.1, 0.2, 0.3, 0.4],
                   [0.11, 0.12, 0.13, 0.14]],
              v2: [[0.5, 0.6, 0.7, 0.8],
                   [0.15, 0.16, 0.17, 0.18]],
              ivs: [[-1, 2],
                    [1, -1]]},
             [[(0.1*0.2 + 0.2*0.2 + 0.3*0.3 + 0.4*0.3), (0.7*0.3)],
              [(0.12*0.2), (0.15*0.1 + 0.16*0.2 + 0.17*0.3 + 0.18*0.4)]])

        test([v1, v2],
             num_sums,
             ivs,
             [0.2, 0.2, 0.3, 0.3, 0.1, 0.2, 0.3, 0.4],
             {v1: [[0.1, 0.2, 0.3, 0.4],
                   [0.11, 0.12, 0.13, 0.14]],
              v2: [[0.5, 0.6, 0.7, 0.8],
                   [0.15, 0.16, 0.17, 0.18]],
              ivs: [[3, 2],
                    [0, 1]]},
             [[(0.4*0.3), (0.7*0.3)],
              [(0.11*0.2), (0.16*0.2)]])

        test([(v3, [7, 5, 3, 1]), (v4, [0, 2, 4, 6])],
             num_sums,
             ivs,
             [0.2, 0.2, 0.3, 0.3, 0.1, 0.2, 0.3, 0.4],
             {v3: [[0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07],
                   [0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17]],
              v4: [[0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27],
                   [0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37]],
              ivs: [[1, 0],
                    [0, 1]]},
             [[(0.05*0.2), (0.2*0.1)],
              [(0.17*0.2), (0.32*0.2)]])

        # Single input with 1 value, multi-element batch
        test([v3],
             num_sums,
             None,
             [0.2, 0.2, 0.3, 0.3, 0.1, 0.2, 0.3, 0.4],
             {v3: [[0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07],
                   [0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17]]},
             [[(0.0*0.2 + 0.01*0.2 + 0.02*0.3 + 0.03*0.3),
               (0.04*0.1 + 0.05*0.2 + 0.06*0.3 + 0.07*0.4)],
              [(0.1*0.2 + 0.11*0.2 + 0.12*0.3 + 0.13*0.3),
               (0.14*0.1 + 0.15*0.2 + 0.16*0.3 + 0.17*0.4)]])

        test([(v3, [7, 5, 3, 1])],
             num_sums,
             None,
             [0.4, 0.6, 0.2, 0.8],
             {v3: [[0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07],
                   [0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17]]},
             [[(0.07*0.4 + 0.05*0.6), (0.03*0.2 + 0.01*0.8)],
              [(0.17*0.4 + 0.15*0.6), (0.13*0.2 + 0.11*0.8)]])

        test([v3],
             num_sums,
             ivs,
             [0.2, 0.2, 0.3, 0.3, 0.1, 0.2, 0.3, 0.4],
             {v3: [[0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07],
                   [0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17]],
              ivs: [[-1, -1],
                    [-1, -1]]},
             [[(0.0*0.2 + 0.01*0.2 + 0.02*0.3 + 0.03*0.3),
               (0.04*0.1 + 0.05*0.2 + 0.06*0.3 + 0.07*0.4)],
              [(0.1*0.2 + 0.11*0.2 + 0.12*0.3 + 0.13*0.3),
               (0.14*0.1 + 0.15*0.2 + 0.16*0.3 + 0.17*0.4)]])

        test([v3],
             num_sums,
             ivs,
             [0.2, 0.2, 0.3, 0.3, 0.1, 0.2, 0.3, 0.4],
             {v3: [[0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07],
                   [0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17]],
              ivs: [[-1, 3],
                    [2, -1]]},
             [[(0.0*0.2 + 0.01*0.2 + 0.02*0.3 + 0.03*0.3), (0.07*0.4)],
              [(0.12*0.3), (0.14*0.1 + 0.15*0.2 + 0.16*0.3 + 0.17*0.4)]])

        test([v3],
             num_sums,
             ivs,
             [0.2, 0.2, 0.3, 0.3, 0.1, 0.2, 0.3, 0.4],
             {v3: [[0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07],
                   [0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17]],
              ivs: [[1, 3],
                    [2, 0]]},
             [[(0.01*0.2), (0.07*0.4)],
              [(0.12*0.3), (0.14*0.1)]])

        test([(v3, [7, 5, 3, 1, 0, 2, 4, 6])],
             num_sums,
             ivs,
             [0.2, 0.2, 0.3, 0.3, 0.1, 0.2, 0.3, 0.4],
             {v3: [[0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07],
                   [0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17]],
              ivs: [[3, 1],
                    [0, 2]]},
             [[(0.01*0.3), (0.02*0.2)],
              [(0.17*0.2), (0.14*0.3)]])

        # Multiple inputs, single-element batch
        test([v1, v2],
             num_sums,
             None,
             [0.2, 0.2, 0.3, 0.3, 0.1, 0.2, 0.3, 0.4],
             {v1: [[0.1, 0.2, 0.3, 0.4]],
              v2: [[0.5, 0.6, 0.7, 0.8]]},
             [[(0.1*0.2 + 0.2*0.2 + 0.3*0.3 + 0.4*0.3),
               (0.5*0.1 + 0.6*0.2 + 0.7*0.3 + 0.8*0.4)]])

        test([(v1, [1, 2]), (v2, [0, 3])],
             num_sums,
             None,
             [0.4, 0.6, 0.2, 0.8],
             {v1: [[0.1, 0.2, 0.3, 0.4]],
              v2: [[0.5, 0.6, 0.7, 0.8]]},
             [[(0.2*0.4 + 0.3*0.6), (0.5*0.2 + 0.8*0.8)]])

        test([v1, v2],
             num_sums,
             ivs,
             [0.2, 0.2, 0.3, 0.3, 0.1, 0.2, 0.3, 0.4],
             {v1: [[0.1, 0.2, 0.3, 0.4]],
              v2: [[0.5, 0.6, 0.7, 0.8]],
              ivs: [[-1, -1]]},
             [[(0.1*0.2 + 0.2*0.2 + 0.3*0.3 + 0.4*0.3),
               (0.5*0.1 + 0.6*0.2 + 0.7*0.3 + 0.8*0.4)]])

        test([v1, v2],
             num_sums,
             ivs,
             [0.2, 0.2, 0.3, 0.3, 0.1, 0.2, 0.3, 0.4],
             {v1: [[0.1, 0.2, 0.3, 0.4]],
              v2: [[0.5, 0.6, 0.7, 0.8]],
              ivs: [[-1, 2]]},
             [[(0.1*0.2 + 0.2*0.2 + 0.3*0.3 + 0.4*0.3), (0.7*0.3)]])

        test([v1, v2],
             num_sums,
             ivs,
             [0.2, 0.2, 0.3, 0.3, 0.1, 0.2, 0.3, 0.4],
             {v1: [[0.1, 0.2, 0.3, 0.4]],
              v2: [[0.5, 0.6, 0.7, 0.8]],
              ivs: [[3, 2]]},
             [[(0.4*0.3), (0.7*0.3)]])

        test([(v3, [7, 5, 3, 1]), (v4, [0, 2, 4, 6])],
             num_sums,
             ivs,
             [0.2, 0.2, 0.3, 0.3, 0.1, 0.2, 0.3, 0.4],
             {v3: [[0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]],
              v4: [[0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27]],
              ivs: [[1, 0]]},
             [[(0.05*0.2), (0.2*0.1)]])

        # Single input with 1 value, single-element batch
        test([v3],
             num_sums,
             None,
             [0.2, 0.2, 0.3, 0.3, 0.1, 0.2, 0.3, 0.4],
             {v3: [[0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]]},
             [[(0.0*0.2 + 0.01*0.2 + 0.02*0.3 + 0.03*0.3),
               (0.04*0.1 + 0.05*0.2 + 0.06*0.3 + 0.07*0.4)]])

        test([(v3, [7, 5, 3, 1])],
             num_sums,
             None,
             [0.4, 0.6, 0.2, 0.8],
             {v3: [[0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]]},
             [[(0.07*0.4 + 0.05*0.6), (0.03*0.2 + 0.01*0.8)]])

        test([v3],
             num_sums,
             ivs,
             [0.2, 0.2, 0.3, 0.3, 0.1, 0.2, 0.3, 0.4],
             {v3: [[0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]],
              ivs: [[-1, -1]]},
             [[(0.0*0.2 + 0.01*0.2 + 0.02*0.3 + 0.03*0.3),
               (0.04*0.1 + 0.05*0.2 + 0.06*0.3 + 0.07*0.4)]])

        test([v3],
             num_sums,
             ivs,
             [0.2, 0.2, 0.3, 0.3, 0.1, 0.2, 0.3, 0.4],
             {v3: [[0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]],
              ivs: [[-1, 3]]},
             [[(0.0*0.2 + 0.01*0.2 + 0.02*0.3 + 0.03*0.3), (0.07*0.4)]])

        test([v3],
             num_sums,
             ivs,
             [0.2, 0.2, 0.3, 0.3, 0.1, 0.2, 0.3, 0.4],
             {v3: [[0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]],
              ivs: [[1, 3]]},
             [[(0.01*0.2), (0.07*0.4)]])

        test([(v3, [7, 5, 3, 1, 0, 2, 4, 6])],
             num_sums,
             ivs,
             [0.2, 0.2, 0.3, 0.3, 0.1, 0.2, 0.3, 0.4],
             {v3: [[0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]],
              ivs: [[3, 1]]},
             [[(0.01*0.3), (0.02*0.2)]])

        # SINGLE SUM NODES
        # ----------------
        num_sums = 1
        ivs = spn.IVs(num_vars=num_sums, num_vals=4)

        # Create inputs
        v1 = spn.ContVars(num_vars=2, name="ContVars1")
        v2 = spn.ContVars(num_vars=2, name="ContVars2")

        v3 = spn.ContVars(num_vars=4, name="ContVars3")
        v4 = spn.ContVars(num_vars=4, name="ContVars4")

        # Multiple inputs, multi-element batch
        test([v1, v2],
             num_sums,
             None,
             [0.2, 0.2, 0.3, 0.3],
             {v1: [[0.1, 0.2],
                   [0.11, 0.12]],
              v2: [[0.3, 0.4],
                   [0.13, 0.14]]},
             [[(0.1*0.2 + 0.2*0.2 + 0.3*0.3 + 0.4*0.3)],
              [(0.11*0.2 + 0.12*0.2 + 0.13*0.3 + 0.14*0.3)]])

        test([(v1, [1]), (v2, [0])],
             num_sums,
             None,
             [0.4, 0.6],
             {v1: [[0.1, 0.2],
                   [0.11, 0.12]],
              v2: [[0.3, 0.4],
                   [0.13, 0.14]]},
             [[(0.2*0.4 + 0.3*0.6)],
              [(0.12*0.4 + 0.13*0.6)]])

        test([v1, v2],
             num_sums,
             ivs,
             [0.2, 0.2, 0.3, 0.3],
             {v1: [[0.1, 0.2],
                   [0.11, 0.12]],
              v2: [[0.3, 0.4],
                   [0.13, 0.14]],
              ivs: [[-1],
                    [-1]]},
             [[(0.1*0.2 + 0.2*0.2 + 0.3*0.3 + 0.4*0.3)],
              [(0.11*0.2 + 0.12*0.2 + 0.13*0.3 + 0.14*0.3)]])

        test([v1, v2],
             num_sums,
             ivs,
             [0.2, 0.2, 0.3, 0.3],
             {v1: [[0.1, 0.2],
                   [0.11, 0.12]],
              v2: [[0.3, 0.4],
                   [0.13, 0.14]],
              ivs: [[1],
                    [-1]]},
             [[(0.2*0.2)],
              [(0.11*0.2 + 0.12*0.2 + 0.13*0.3 + 0.14*0.3)]])

        test([v1, v2],
             num_sums,
             ivs,
             [0.2, 0.2, 0.3, 0.3],
             {v1: [[0.1, 0.2],
                   [0.11, 0.12]],
              v2: [[0.3, 0.4],
                   [0.13, 0.14]],
              ivs: [[2],
                    [1]]},
             [[(0.3*0.3)],
              [(0.12*0.2)]])

        test([(v3, [2, 0]), (v4, [1, 3])],
             num_sums,
             ivs,
             [0.1, 0.2, 0.3, 0.4],
             {v3: [[0.0, 0.01, 0.02, 0.03],
                   [0.1, 0.11, 0.12, 0.13]],
              v4: [[0.2, 0.21, 0.22, 0.23],
                   [0.3, 0.31, 0.32, 0.33]],
              ivs: [[3],
                    [1]]},
             [[(0.23*0.4)],
              [(0.1*0.2)]])

        # Single input with 1 value, multi-element batch
        test([v3],
             num_sums,
             None,
             [0.1, 0.2, 0.3, 0.4],
             {v3: [[0.1, 0.2, 0.3, 0.4],
                   [0.11, 0.12, 0.13, 0.14]]},
             [[(0.1*0.1 + 0.2*0.2 + 0.3*0.3 + 0.4*0.4)],
              [(0.11*0.1 + 0.12*0.2 + 0.13*0.3 + 0.14*0.4)]])

        test([(v3, [3, 1])],
             num_sums,
             None,
             [0.4, 0.6],
             {v3: [[0.1, 0.2, 0.3, 0.4],
                   [0.11, 0.12, 0.13, 0.14]]},
             [[(0.4*0.4 + 0.2*0.6)],
              [(0.14*0.4 + 0.12*0.6)]])

        test([v3],
             num_sums,
             ivs,
             [0.1, 0.2, 0.3, 0.4],
             {v3: [[0.1, 0.2, 0.3, 0.4],
                   [0.11, 0.12, 0.13, 0.14]],
              ivs: [[-1],
                    [-1]]},
             [[(0.1*0.1 + 0.2*0.2 + 0.3*0.3 + 0.4*0.4)],
              [(0.11*0.1 + 0.12*0.2 + 0.13*0.3 + 0.14*0.4)]])

        test([v3],
             num_sums,
             ivs,
             [0.1, 0.2, 0.3, 0.4],
             {v3: [[0.1, 0.2, 0.3, 0.4],
                   [0.11, 0.12, 0.13, 0.14]],
              ivs: [[-1],
                    [2]]},
             [[(0.1*0.1 + 0.2*0.2 + 0.3*0.3 + 0.4*0.4)],
              [(0.13*0.3)]])

        test([v3],
             num_sums,
             ivs,
             [0.1, 0.2, 0.3, 0.4],
             {v3: [[0.1, 0.2, 0.3, 0.4],
                   [0.11, 0.12, 0.13, 0.14]],
              ivs: [[0],
                    [2]]},
             [[(0.1*0.1)],
              [(0.13*0.3)]])

        test([(v3, [3, 1, 0, 2])],
             num_sums,
             ivs,
             [0.1, 0.2, 0.3, 0.4],
             {v3: [[0.1, 0.2, 0.3, 0.4],
                   [0.11, 0.12, 0.13, 0.14]],
              ivs: [[2],
                    [1]]},
             [[(0.1*0.3)],
              [(0.12*0.2)]])

        # Multiple inputs, single-element batch
        test([v1, v2],
             num_sums,
             None,
             [0.2, 0.2, 0.3, 0.3],
             {v1: [[0.1, 0.2]],
              v2: [[0.3, 0.4]]},
             [[(0.1*0.2 + 0.2*0.2 + 0.3*0.3 + 0.4*0.3)]])

        test([(v1, [1]), (v2, [0])],
             num_sums,
             None,
             [0.4, 0.6],
             {v1: [[0.1, 0.2]],
              v2: [[0.3, 0.4]]},
             [[(0.2*0.4 + 0.3*0.6)]])

        test([v1, v2],
             num_sums,
             ivs,
             [0.2, 0.2, 0.3, 0.3],
             {v1: [[0.1, 0.2]],
              v2: [[0.3, 0.4]],
              ivs: [[-1]]},
             [[(0.1*0.2 + 0.2*0.2 + 0.3*0.3 + 0.4*0.3)]])

        test([v1, v2],
             num_sums,
             ivs,
             [0.2, 0.2, 0.3, 0.3],
             {v1: [[0.1, 0.2]],
              v2: [[0.3, 0.4]],
              ivs: [[1]]},
             [[(0.2*0.2)]])

        test([(v3, [2, 0]), (v4, [1, 3])],
             num_sums,
             ivs,
             [0.1, 0.2, 0.3, 0.4],
             {v3: [[0.0, 0.01, 0.02, 0.03]],
              v4: [[0.2, 0.21, 0.22, 0.23]],
              ivs: [[3]]},
             [[(0.23*0.4)]])

        # Single input with 1 value, single-element batch
        test([v3],
             num_sums,
             None,
             [0.1, 0.2, 0.3, 0.4],
             {v3: [[0.1, 0.2, 0.3, 0.4]]},
             [[(0.1*0.1 + 0.2*0.2 + 0.3*0.3 + 0.4*0.4)]])

        test([(v3, [3, 1])],
             num_sums,
             None,
             [0.4, 0.6],
             {v3: [[0.1, 0.2, 0.3, 0.4]]},
             [[(0.4*0.4 + 0.2*0.6)]])

        test([v3],
             num_sums,
             ivs,
             [0.1, 0.2, 0.3, 0.4],
             {v3: [[0.1, 0.2, 0.3, 0.4]],
              ivs: [[-1]]},
             [[(0.1*0.1 + 0.2*0.2 + 0.3*0.3 + 0.4*0.4)]])

        test([(v3, [3, 1, 0, 2])],
             num_sums,
             ivs,
             [0.1, 0.2, 0.3, 0.4],
             {v3: [[0.1, 0.2, 0.3, 0.4]],
              ivs: [[2]]},
             [[(0.1*0.3)]])

    def test_compute_mpe_value(self):
        """Calculating MPE value of Sums."""
        def test(values, num_sums_or_sizes, ivs, weights, feed, output):
            with self.subTest(values=values, num_sums_or_sizes=num_sums_or_sizes, ivs=ivs,
                              weights=weights, feed=feed):
                n = spn.SumsLayer(*values, ivs=ivs, num_sums_or_sizes=num_sums_or_sizes)
                n.generate_weights(weights)
                op = n.get_value(spn.InferenceType.MPE)
                op_log = n.get_log_value(spn.InferenceType.MPE)
                with tf.Session() as sess:
                    spn.initialize_weights(n).run()
                    out = sess.run(op, feed_dict=feed)
                    out_log = sess.run(tf.exp(op_log), feed_dict=feed)

                np.testing.assert_array_almost_equal(
                    out,
                    np.array(output, dtype=spn.conf.dtype.as_numpy_dtype()))
                np.testing.assert_array_almost_equal(
                    out_log,
                    np.array(output, dtype=spn.conf.dtype.as_numpy_dtype()))

        # MULTIPLE SUM NODES
        # -------------------
        num_sums = 2
        ivs = spn.IVs(num_vars=num_sums, num_vals=4)

        # Create inputs
        v1 = spn.ContVars(num_vars=4, name="ContVars1")
        v2 = spn.ContVars(num_vars=4, name="ContVars2")

        v3 = spn.ContVars(num_vars=8, name="ContVars3")
        v4 = spn.ContVars(num_vars=8, name="ContVars4")

        # Multiple inputs, multi-element batch
        test([v1, v2],
             None,
             None,
             [0.2, 0.2, 0.3, 0.3, 0.1, 0.2, 0.3, 0.4],
             {v1: [[0.1, 0.2, 0.3, 0.4],
                   [0.11, 0.12, 0.13, 0.14]],
              v2: [[0.5, 0.6, 0.7, 0.8],
                   [0.15, 0.16, 0.17, 0.18]]},
             [[(0.4*0.3),  (0.8*0.4)],
              [(0.14*0.3), (0.18*0.4)]])

        test([(v1, [1, 2]), (v2, [0, 3])],
             None,
             None,
             [0.4, 0.6, 0.2, 0.8],
             {v1: [[0.1, 0.2, 0.3, 0.4],
                   [0.11, 0.12, 0.13, 0.14]],
              v2: [[0.5, 0.6, 0.7, 0.8],
                   [0.15, 0.16, 0.17, 0.18]]},
             [[(0.3*0.6),  (0.8*0.8)],
              [(0.13*0.6), (0.18*0.8)]])

        test([v1, v2],
             None,
             ivs,
             [0.2, 0.2, 0.3, 0.3, 0.1, 0.2, 0.3, 0.4],
             {v1: [[0.1, 0.2, 0.3, 0.4],
                   [0.11, 0.12, 0.13, 0.14]],
              v2: [[0.5, 0.6, 0.7, 0.8],
                   [0.15, 0.16, 0.17, 0.18]],
              ivs: [[-1, -1],
                    [-1, -1]]},
             [[(0.4*0.3),  (0.8*0.4)],
              [(0.14*0.3), (0.18*0.4)]])

        test([v1, v2],
             None,
             ivs,
             [0.2, 0.2, 0.3, 0.3, 0.1, 0.2, 0.3, 0.4],
             {v1: [[0.1, 0.2, 0.3, 0.4],
                   [0.11, 0.12, 0.13, 0.14]],
              v2: [[0.5, 0.6, 0.7, 0.8],
                   [0.15, 0.16, 0.17, 0.18]],
              ivs: [[-1, 2],
                    [1, -1]]},
             [[(0.4*0.3),  (0.7*0.3)],
              [(0.12*0.2), (0.18*0.4)]])

        test([v1, v2],
             None,
             ivs,
             [0.2, 0.2, 0.3, 0.3, 0.1, 0.2, 0.3, 0.4],
             {v1: [[0.1, 0.2, 0.3, 0.4],
                   [0.11, 0.12, 0.13, 0.14]],
              v2: [[0.5, 0.6, 0.7, 0.8],
                   [0.15, 0.16, 0.17, 0.18]],
              ivs: [[3, 2],
                    [0, 1]]},
             [[(0.4*0.3),  (0.7*0.3)],
              [(0.11*0.2), (0.16*0.2)]])

        test([(v3, [7, 5, 3, 1]), (v4, [0, 2, 4, 6])],
             None,
             ivs,
             [0.2, 0.2, 0.3, 0.3, 0.1, 0.2, 0.3, 0.4],
             {v3: [[0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07],
                   [0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17]],
              v4: [[0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27],
                   [0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37]],
              ivs: [[1, 0],
                    [0, 1]]},
             [[(0.05*0.2), (0.2*0.1)],
              [(0.17*0.2), (0.32*0.2)]])

        # Single input with 1 value, multi-element batch
        test([v3],
             [4, 4],
             None,
             [0.2, 0.2, 0.3, 0.3, 0.1, 0.2, 0.3, 0.4],
             {v3: [[0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07],
                   [0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17]]},
             [[(0.03*0.3), (0.07*0.4)],
              [(0.13*0.3), (0.17*0.4)]])

        test([(v3, [7, 5, 3, 1])],
             2,
             None,
             [0.4, 0.6, 0.2, 0.8],
             {v3: [[0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07],
                   [0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17]]},
             [[(0.05*0.6), (0.01*0.8)],
              [(0.15*0.6), (0.11*0.8)]])

        test([v3],
             [4, 4],
             ivs,
             [0.2, 0.2, 0.3, 0.3, 0.1, 0.2, 0.3, 0.4],
             {v3: [[0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07],
                   [0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17]],
              ivs: [[-1, -1],
                    [-1, -1]]},
             [[(0.03*0.3), (0.07*0.4)],
              [(0.13*0.3), (0.17*0.4)]])

        test([v3],
             [4, 4],
             ivs,
             [0.2, 0.2, 0.3, 0.3, 0.1, 0.2, 0.3, 0.4],
             {v3: [[0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07],
                   [0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17]],
              ivs: [[-1, 3],
                    [2, -1]]},
             [[(0.03*0.3), (0.07*0.4)],
              [(0.12*0.3), (0.17*0.4)]])

        test([v3],
             [4, 4],
             ivs,
             [0.2, 0.2, 0.3, 0.3, 0.1, 0.2, 0.3, 0.4],
             {v3: [[0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07],
                   [0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17]],
              ivs: [[1, 3],
                    [2, 0]]},
             [[(0.01*0.2), (0.07*0.4)],
              [(0.12*0.3), (0.14*0.1)]])

        test([(v3, [7, 5, 3, 1, 0, 2, 4, 6])],
             2,
             ivs,
             [0.2, 0.2, 0.3, 0.3, 0.1, 0.2, 0.3, 0.4],
             {v3: [[0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07],
                   [0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17]],
              ivs: [[3, 1],
                    [0, 2]]},
             [[(0.01*0.3), (0.02*0.2)],
              [(0.17*0.2), (0.14*0.3)]])

        # Multiple inputs, single-element batch
        test([v1, v2],
             None,
             None,
             [0.2, 0.2, 0.3, 0.3, 0.1, 0.2, 0.3, 0.4],
             {v1: [[0.1, 0.2, 0.3, 0.4]],
              v2: [[0.5, 0.6, 0.7, 0.8]]},
             [[(0.4*0.3), (0.8*0.4)]])

        test([(v1, [1, 2]), (v2, [0, 3])],
             None,
             None,
             [0.4, 0.6, 0.2, 0.8],
             {v1: [[0.1, 0.2, 0.3, 0.4]],
              v2: [[0.5, 0.6, 0.7, 0.8]]},
             [[(0.3*0.6), (0.8*0.8)]])

        test([v1, v2],
             None,
             ivs,
             [0.2, 0.2, 0.3, 0.3, 0.1, 0.2, 0.3, 0.4],
             {v1: [[0.1, 0.2, 0.3, 0.4]],
              v2: [[0.5, 0.6, 0.7, 0.8]],
              ivs: [[-1, -1]]},
             [[(0.4*0.3), (0.8*0.4)]])

        test([v1, v2],
             None,
             ivs,
             [0.2, 0.2, 0.3, 0.3, 0.1, 0.2, 0.3, 0.4],
             {v1: [[0.1, 0.2, 0.3, 0.4]],
              v2: [[0.5, 0.6, 0.7, 0.8]],
              ivs: [[-1, 2]]},
             [[(0.4*0.3), (0.7*0.3)]])

        test([v1, v2],
             None,
             ivs,
             [0.2, 0.2, 0.3, 0.3, 0.1, 0.2, 0.3, 0.4],
             {v1: [[0.1, 0.2, 0.3, 0.4]],
              v2: [[0.5, 0.6, 0.7, 0.8]],
              ivs: [[3, 2]]},
             [[(0.4*0.3), (0.7*0.3)]])

        test([(v3, [7, 5, 3, 1]), (v4, [0, 2, 4, 6])],
             None,
             ivs,
             [0.2, 0.2, 0.3, 0.3, 0.1, 0.2, 0.3, 0.4],
             {v3: [[0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]],
              v4: [[0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27]],
              ivs: [[1, 0]]},
             [[(0.05*0.2), (0.2*0.1)]])

        # Single input with 1 value, single-element batch
        test([v3],
             [4, 4],
             None,
             [0.2, 0.2, 0.3, 0.3, 0.1, 0.2, 0.3, 0.4],
             {v3: [[0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]]},
             [[(0.03*0.3), (0.07*0.4)]])

        test([(v3, [7, 5, 3, 1])],
             2,
             None,
             [0.4, 0.6, 0.2, 0.8],
             {v3: [[0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]]},
             [[(0.05*0.6), (0.01*0.8)]])

        test([v3],
             2,
             ivs,
             [0.2, 0.2, 0.3, 0.3, 0.1, 0.2, 0.3, 0.4],
             {v3: [[0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]],
              ivs: [[-1, -1]]},
             [[(0.03*0.3), (0.07*0.4)]])

        test([v3],
             [4, 4],
             ivs,
             [0.2, 0.2, 0.3, 0.3, 0.1, 0.2, 0.3, 0.4],
             {v3: [[0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]],
              ivs: [[-1, 3]]},
             [[(0.03*0.3), (0.07*0.4)]])

        test([v3],
             [4, 4],
             ivs,
             [0.2, 0.2, 0.3, 0.3, 0.1, 0.2, 0.3, 0.4],
             {v3: [[0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]],
              ivs: [[1, 3]]},
             [[(0.01*0.2), (0.07*0.4)]])

        test([(v3, [7, 5, 3, 1, 0, 2, 4, 6])],
             [4, 4],
             ivs,
             [0.2, 0.2, 0.3, 0.3, 0.1, 0.2, 0.3, 0.4],
             {v3: [[0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]],
              ivs: [[3, 1]]},
             [[(0.01*0.3), (0.02*0.2)]])

        # SINGLE SUM NODES
        # ----------------
        num_sums = 1
        ivs = spn.IVs(num_vars=num_sums, num_vals=4)

        # Create inputs
        v1 = spn.ContVars(num_vars=2, name="ContVars1")
        v2 = spn.ContVars(num_vars=2, name="ContVars2")

        v3 = spn.ContVars(num_vars=4, name="ContVars3")
        v4 = spn.ContVars(num_vars=4, name="ContVars4")

        # Multiple inputs, multi-element batch
        test([v1, v2],
             num_sums,
             None,
             [0.2, 0.2, 0.3, 0.3],
             {v1: [[0.1, 0.2],
                   [0.11, 0.12]],
              v2: [[0.3, 0.4],
                   [0.13, 0.14]]},
             [[(0.4*0.3)],
              [(0.14*0.3)]])

        test([(v1, [1]), (v2, [0])],
             num_sums,
             None,
             [0.4, 0.6],
             {v1: [[0.1, 0.2],
                   [0.11, 0.12]],
              v2: [[0.3, 0.4],
                   [0.13, 0.14]]},
             [[(0.3*0.6)],
              [(0.13*0.6)]])

        test([v1, v2],
             num_sums,
             ivs,
             [0.2, 0.2, 0.3, 0.3],
             {v1: [[0.1, 0.2],
                   [0.11, 0.12]],
              v2: [[0.3, 0.4],
                   [0.13, 0.14]],
              ivs: [[-1],
                    [-1]]},
             [[(0.4*0.3)],
              [(0.14*0.3)]])

        test([v1, v2],
             num_sums,
             ivs,
             [0.2, 0.2, 0.3, 0.3],
             {v1: [[0.1, 0.2],
                   [0.11, 0.12]],
              v2: [[0.3, 0.4],
                   [0.13, 0.14]],
              ivs: [[1],
                    [-1]]},
             [[(0.2*0.2)],
              [(0.14*0.3)]])

        test([v1, v2],
             num_sums,
             ivs,
             [0.2, 0.2, 0.3, 0.3],
             {v1: [[0.1, 0.2],
                   [0.11, 0.12]],
              v2: [[0.3, 0.4],
                   [0.13, 0.14]],
              ivs: [[2],
                    [1]]},
             [[(0.3*0.3)],
              [(0.12*0.2)]])

        test([(v3, [2, 0]), (v4, [1, 3])],
             num_sums,
             ivs,
             [0.1, 0.2, 0.3, 0.4],
             {v3: [[0.0, 0.01, 0.02, 0.03],
                   [0.1, 0.11, 0.12, 0.13]],
              v4: [[0.2, 0.21, 0.22, 0.23],
                   [0.3, 0.31, 0.32, 0.33]],
              ivs: [[3],
                    [1]]},
             [[(0.23*0.4)],
              [(0.1*0.2)]])

        # Single input with 1 value, multi-element batch
        test([v3],
             num_sums,
             None,
             [0.1, 0.2, 0.3, 0.4],
             {v3: [[0.1, 0.2, 0.3, 0.4],
                   [0.11, 0.12, 0.13, 0.14]]},
             [[(0.4*0.4)],
              [(0.14*0.4)]])

        test([(v3, [3, 1])],
             num_sums,
             None,
             [0.4, 0.6],
             {v3: [[0.1, 0.2, 0.3, 0.4],
                   [0.11, 0.12, 0.13, 0.14]]},
             [[(0.4*0.4)],
              [(0.12*0.6)]])

        test([v3],
             num_sums,
             ivs,
             [0.1, 0.2, 0.3, 0.4],
             {v3: [[0.1, 0.2, 0.3, 0.4],
                   [0.11, 0.12, 0.13, 0.14]],
              ivs: [[-1],
                    [-1]]},
             [[(0.4*0.4)],
              [(0.14*0.4)]])

        test([v3],
             num_sums,
             ivs,
             [0.1, 0.2, 0.3, 0.4],
             {v3: [[0.1, 0.2, 0.3, 0.4],
                   [0.11, 0.12, 0.13, 0.14]],
              ivs: [[-1],
                    [2]]},
             [[(0.4*0.4)],
              [(0.13*0.3)]])

        test([v3],
             num_sums,
             ivs,
             [0.1, 0.2, 0.3, 0.4],
             {v3: [[0.1, 0.2, 0.3, 0.4],
                   [0.11, 0.12, 0.13, 0.14]],
              ivs: [[0],
                    [2]]},
             [[(0.1*0.1)],
              [(0.13*0.3)]])

        test([(v3, [3, 1, 0, 2])],
             num_sums,
             ivs,
             [0.1, 0.2, 0.3, 0.4],
             {v3: [[0.1, 0.2, 0.3, 0.4],
                   [0.11, 0.12, 0.13, 0.14]],
              ivs: [[2],
                    [1]]},
             [[(0.1*0.3)],
              [(0.12*0.2)]])

        # Multiple inputs, single-element batch
        test([v1, v2],
             num_sums,
             None,
             [0.2, 0.2, 0.3, 0.3],
             {v1: [[0.1, 0.2]],
              v2: [[0.3, 0.4]]},
             [[(0.4*0.3)]])

        test([(v1, [1]), (v2, [0])],
             num_sums,
             None,
             [0.4, 0.6],
             {v1: [[0.1, 0.2]],
              v2: [[0.3, 0.4]]},
             [[(0.3*0.6)]])

        test([v1, v2],
             num_sums,
             ivs,
             [0.2, 0.2, 0.3, 0.3],
             {v1: [[0.1, 0.2]],
              v2: [[0.3, 0.4]],
              ivs: [[-1]]},
             [[(0.4*0.3)]])

        test([v1, v2],
             num_sums,
             ivs,
             [0.2, 0.2, 0.3, 0.3],
             {v1: [[0.1, 0.2]],
              v2: [[0.3, 0.4]],
              ivs: [[1]]},
             [[(0.2*0.2)]])

        test([(v3, [2, 0]), (v4, [1, 3])],
             num_sums,
             ivs,
             [0.1, 0.2, 0.3, 0.4],
             {v3: [[0.0, 0.01, 0.02, 0.03]],
              v4: [[0.2, 0.21, 0.22, 0.23]],
              ivs: [[3]]},
             [[(0.23*0.4)]])

        # Single input with 1 value, single-element batch
        test([v3],
             num_sums,
             None,
             [0.1, 0.2, 0.3, 0.4],
             {v3: [[0.1, 0.2, 0.3, 0.4]]},
             [[(0.4*0.4)]])

        test([(v3, [3, 1])],
             num_sums,
             None,
             [0.4, 0.6],
             {v3: [[0.1, 0.2, 0.3, 0.4]]},
             [[(0.4*0.4)]])

        test([v3],
             num_sums,
             ivs,
             [0.1, 0.2, 0.3, 0.4],
             {v3: [[0.1, 0.2, 0.3, 0.4]],
              ivs: [[-1]]},
             [[(0.4*0.4)]])

        test([(v3, [3, 1, 0, 2])],
             num_sums,
             ivs,
             [0.1, 0.2, 0.3, 0.4],
             {v3: [[0.1, 0.2, 0.3, 0.4]],
              ivs: [[2]]},
             [[(0.1*0.3)]])

    def test_compute_mpe_value_varsize(self):
        """Calculating MPE value of Sums."""
        def test(values, num_sums_or_sizes, ivs, weights, feed, output):
            with self.subTest(values=values, num_sums_or_sizes=num_sums_or_sizes, ivs=ivs,
                              weights=weights, feed=feed):
                n = spn.SumsLayer(*values, ivs=ivs, num_sums_or_sizes=num_sums_or_sizes)
                n.generate_weights(weights)
                op = n.get_value(spn.InferenceType.MPE)
                op_log = n.get_log_value(spn.InferenceType.MPE)
                with tf.Session() as sess:
                    spn.initialize_weights(n).run()
                    out = sess.run(op, feed_dict=feed)
                    out_log = sess.run(tf.exp(op_log), feed_dict=feed)

                np.testing.assert_array_almost_equal(
                    out,
                    np.array(output, dtype=spn.conf.dtype.as_numpy_dtype()))
                np.testing.assert_array_almost_equal(
                    out_log,
                    np.array(output, dtype=spn.conf.dtype.as_numpy_dtype()))

        # MULTIPLE SUM NODES
        # -------------------
        num_sums = 2
        ivs = spn.IVs(num_vars=num_sums, num_vals=5)

        # Create inputs
        v1 = spn.ContVars(num_vars=4, name="ContVars1")
        v2 = spn.ContVars(num_vars=4, name="ContVars2")

        # Multiple inputs, multi-element batch
        test([v1, v2],
             [3, 5],
             None,
             [0.2, 0.2, 0.3, 0.3, 0.1, 0.2, 0.3, 0.4],
             {v1: [[0.1, 0.2, 0.3, 0.4],
                   [0.11, 0.12, 0.13, 0.14]],
              v2: [[0.5, 0.6, 0.7, 0.8],
                   [0.15, 0.16, 0.17, 0.18]]},
             [[(0.3*0.3/0.7),  (0.8*0.4/1.3)],
              [(0.13*0.3/0.7), (0.18*0.4/1.3)]])

        test([(v1, [1, 2]), (v2, [0, 3])],
             [3, 1],
             None,
             [0.4, 0.6, 0.2, 0.8],
             {v1: [[0.1, 0.2, 0.3, 0.4],
                   [0.11, 0.12, 0.13, 0.14]],
              v2: [[0.5, 0.6, 0.7, 0.8],
                   [0.15, 0.16, 0.17, 0.18]]},
             [[(0.3*0.6)/1.2,  (0.8*0.8)/0.8],
              [(0.13*0.6)/1.2, (0.18*0.8)/0.8]])

        test([v1, v2],
             [3, 5],
             ivs,
             [0.2, 0.2, 0.3, 0.3, 0.1, 0.2, 0.3, 0.4],
             {v1: [[0.1, 0.2, 0.3, 0.4],
                   [0.11, 0.12, 0.13, 0.14]],
              v2: [[0.5, 0.6, 0.7, 0.8],
                   [0.15, 0.16, 0.17, 0.18]],
              ivs: [[-1, -1],
                    [-1, -1]]},
             [[(0.3*0.3/0.7),  (0.8*0.4/1.3)],
              [(0.13*0.3/0.7), (0.18*0.4/1.3)]])

        test([v1, v2],
             [3, 5],
             ivs,
             [0.2, 0.2, 0.3, 0.3, 0.1, 0.2, 0.3, 0.4],
             {v1: [[0.1, 0.2, 0.3, 0.4],
                   [0.11, 0.12, 0.13, 0.14]],
              v2: [[0.5, 0.6, 0.7, 0.8],
                   [0.15, 0.16, 0.17, 0.18]],
              ivs: [[-1, 2],
                    [1, -1]]},
             [[(0.3*0.3)/0.7,  (0.6*0.2)/1.3],
              [(0.12*0.2)/0.7, (0.18*0.4)/1.3]])

    def test_compute_scope(self):
        """Calculating scope of Sums"""
        # Create a graph
        v12 = spn.IVs(num_vars=2, num_vals=4, name="V12")
        v34 = spn.ContVars(num_vars=3, name="V34")

        scopes_per_node = {
            v12: [spn.Scope(v12, 0), spn.Scope(v12, 0), spn.Scope(v12, 0), spn.Scope(v12, 0),
                  spn.Scope(v12, 1), spn.Scope(v12, 1), spn.Scope(v12, 1), spn.Scope(v12, 1)],
            v34: [spn.Scope(v34, 0), spn.Scope(v34, 1), spn.Scope(v34, 2)]
        }

        def generate_scopes_from_inputs(node, inputs, num_sums_or_sizes, ivs=False):
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
            if isinstance(num_sums_or_sizes, int):
                num_sums_or_sizes = num_sums_or_sizes * [size // num_sums_or_sizes]

            new_scope = []
            offset = 0
            # For each sum generate the scope based on its size
            for i, s in enumerate(num_sums_or_sizes):
                scope = flat_scopes[offset]
                for j in range(1, s):
                    scope |= flat_scopes[j + offset]
                offset += s
                if ivs:
                    scope |= spn.Scope(node.ivs.node, i)
                new_scope.append(scope)
            scopes_per_node[node] = new_scope

        def sums_layer_and_test(inputs, num_sums_or_sizes, name, ivs=False):
            """ Create a sums layer, generate its correct scope and test """
            sums_layer = spn.SumsLayer(*inputs, num_sums_or_sizes=num_sums_or_sizes, name=name)
            if ivs:
                sums_layer.generate_ivs()
            generate_scopes_from_inputs(sums_layer, inputs, num_sums_or_sizes, ivs=ivs)
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
            [(v12, [0, 1, 2, 3]), (v12, [1, 2, 5, 6]), (v12, [4, 5, 6, 7])], 3, "Ss1", ivs=True)

        ss2 = sums_layer_and_test([(v12, [6, 7]), (v34, 0)], num_sums_or_sizes=[1, 2], name="Ss2")
        ss3 = sums_layer_and_test([(v12, [3, 7]), (v34, 1), (v12, [4, 5, 6]), v34],
                                  num_sums_or_sizes=[1, 2, 2, 2, 2], name="Ss3")

        s1 = sums_layer_and_test([(v34, [1, 2])], num_sums_or_sizes=1, name="S1", ivs=True)
        concat_layer_and_test([(ss1, [0, 2]), (ss2, 0)], name="N1")
        concat_layer_and_test([(ss1, 1), ss3, s1], name="N2")
        n = concat_layer_and_test([(ss1, 0), ss2, (ss3, [0, 1]), s1], name="N3")
        sums_layer_and_test([(ss1, [1, 2]), ss2], num_sums_or_sizes=[2, 1, 1], name="Ss4")
        sums_layer_and_test([(ss1, [0, 2]), (n, [0, 1]), (ss3, [4, 2])],
                            num_sums_or_sizes=[3, 2, 1], name="Ss5")

    def test_compute_valid(self):
        """Calculating validity of Sums"""
        # Without IVs
        v12 = spn.IVs(num_vars=2, num_vals=4)
        v34 = spn.ContVars(num_vars=2)
        s1 = spn.SumsLayer((v12, [0, 1, 2, 3]), (v12, [0, 1, 2, 3]),
                           (v12, [0, 1, 2, 3]), num_sums_or_sizes=3)
        self.assertTrue(s1.is_valid())

        s2 = spn.SumsLayer((v12, [0, 1, 2, 4]), name="S2")
        s2b = spn.SumsLayer((v12, [0, 1, 2, 4]), num_sums_or_sizes=[3, 1], name="S2b")
        self.assertTrue(s2b.is_valid())
        self.assertFalse(s2.is_valid())

        s3 = spn.SumsLayer((v12, [0, 1, 2, 3]), (v34, 0), (v12, [0, 1, 2, 3]),
                           (v34, 0), num_sums_or_sizes=2)
        s3b = spn.SumsLayer((v12, [0, 1, 2, 3]), (v34, 0), (v12, [0, 1, 2, 3]),
                            (v34, 0), num_sums_or_sizes=[4, 1, 4, 1])
        s3c = spn.SumsLayer((v12, [0, 1, 2, 3]), (v34, 0), (v12, [0, 1, 2, 3]),
                            (v34, 0), num_sums_or_sizes=[4, 1, 5])
        self.assertFalse(s3.is_valid())
        self.assertTrue(s3b.is_valid())
        self.assertFalse(s3c.is_valid())

        p1 = spn.Product((v12, [0, 5]), (v34, 0))
        p2 = spn.Product((v12, [1, 6]), (v34, 0))
        p3 = spn.Product((v12, [1, 6]), (v34, 1))

        s4 = spn.SumsLayer(p1, p2, p1, p2, num_sums_or_sizes=2)
        s5 = spn.SumsLayer(p1, p3, p1, p3, p1, p3, num_sums_or_sizes=3)
        s6 = spn.SumsLayer(p1, p2, p3, num_sums_or_sizes=[2, 1])
        s7 = spn.SumsLayer(p1, p2, p3, num_sums_or_sizes=[1, 2])
        s8 = spn.SumsLayer(p1, p2, p3, p2, p1, num_sums_or_sizes=[2, 1, 2])
        self.assertTrue(s4.is_valid())
        self.assertFalse(s5.is_valid())  # p1 and p3 different scopes
        self.assertTrue(s6.is_valid())
        self.assertFalse(s7.is_valid())  # p2 and p3 different scopes
        self.assertTrue(s8.is_valid())
        # With IVS
        s6 = spn.SumsLayer(p1, p2, p1, p2, p1, p2, num_sums_or_sizes=3)
        s6.generate_ivs()
        self.assertTrue(s6.is_valid())

        s7 = spn.SumsLayer(p1, p2, num_sums_or_sizes=1)
        s7.set_ivs(spn.ContVars(num_vars=2))
        self.assertFalse(s7.is_valid())

        s7 = spn.SumsLayer(p1, p2, p3, num_sums_or_sizes=3)
        s7.set_ivs(spn.ContVars(num_vars=3))
        self.assertTrue(s7.is_valid())

        s7 = spn.SumsLayer(p1, p2, p3, num_sums_or_sizes=[2, 1])
        s7.set_ivs(spn.ContVars(num_vars=3))
        self.assertFalse(s7.is_valid())

        s8 = spn.SumsLayer(p1, p2, p1, p2, num_sums_or_sizes=2)
        s8.set_ivs(spn.IVs(num_vars=3, num_vals=2))
        with self.assertRaises(spn.StructureError):
            s8.is_valid()

        s9 = spn.SumsLayer(p1, p2, p1, p2, num_sums_or_sizes=[1, 3])
        s9.set_ivs(spn.ContVars(num_vars=2))
        with self.assertRaises(spn.StructureError):
            s9.is_valid()

        s9 = spn.SumsLayer(p1, p2, p1, p2, num_sums_or_sizes=[1, 3])
        s9.set_ivs(spn.ContVars(num_vars=3))
        with self.assertRaises(spn.StructureError):
            s9.is_valid()

        s9 = spn.SumsLayer(p1, p2, p1, p2, num_sums_or_sizes=2)
        s9.set_ivs(spn.IVs(num_vars=1, num_vals=4))
        self.assertTrue(s9.is_valid())

        s9 = spn.SumsLayer(p1, p2, p1, p2, num_sums_or_sizes=[1, 3])
        s9.set_ivs(spn.IVs(num_vars=1, num_vals=4))
        self.assertTrue(s9.is_valid())

        s9 = spn.SumsLayer(p1, p2, p1, p2, num_sums_or_sizes=[1, 3])
        s9.set_ivs(spn.IVs(num_vars=2, num_vals=2))
        self.assertFalse(s9.is_valid())

        s9 = spn.SumsLayer(p1, p2, p1, p2, num_sums_or_sizes=2)
        s9.set_ivs(spn.IVs(num_vars=2, num_vals=2))
        self.assertTrue(s9.is_valid())

        s9 = spn.SumsLayer(p1, p2, p1, p2, num_sums_or_sizes=[1, 2, 1])
        s9.set_ivs(spn.IVs(num_vars=2, num_vals=2))
        self.assertFalse(s9.is_valid())

        s10 = spn.SumsLayer(p1, p2, p1, p2, num_sums_or_sizes=2)
        s10.set_ivs((v12, [0, 3, 5, 7]))
        self.assertTrue(s10.is_valid())

        s10 = spn.SumsLayer(p1, p2, p1, p2, num_sums_or_sizes=[1, 2, 1])
        s10.set_ivs((v12, [0, 3, 5, 7]))
        self.assertFalse(s10.is_valid())

    @parameterized.expand([('Non-log', False), ('Log', True)])
    def test_compute_mpe_path_noivs_single_sum(self, name, log):
        # TODO reconfigure...
        v12 = spn.IVs(num_vars=2, num_vals=4)
        v34 = spn.ContVars(num_vars=2)
        v5 = spn.ContVars(num_vars=1)
        s = spn.SumsLayer((v12, [0, 5]), v34, (v12, [3]), v5, num_sums_or_sizes=1)
        w = s.generate_weights()
        counts = tf.placeholder(tf.float32, shape=(None, 1))
        if log:
            op = s._compute_log_mpe_path(tf.identity(counts),
                                         w.get_log_value(),
                                         None,
                                         v12.get_log_value(),
                                         v34.get_log_value(),
                                         v12.get_log_value(),
                                         v5.get_log_value())
        else:
            op = s._compute_mpe_path(tf.identity(counts),
                                     w.get_value(),
                                     None,
                                     v12.get_value(),
                                     v34.get_value(),
                                     v12.get_value(),
                                     v5.get_value())

        # Index 2 to skip weights and IVs +2 because of index of repeated input tensor
        self.assertEqual(op[2+2], None)
        op = [o for o in op if o is not None]

        init = w.initialize()
        counts_feed = [[10],
                       [11],
                       [12],
                       [13]]
        v12_feed = [[0, 1],
                    [1, 1],
                    [0, 0],
                    [3, 3]]
        v34_feed = [[0.1, 0.2],
                    [1.2, 0.2],
                    [0.1, 0.2],
                    [0.9, 0.8]]
        v5_feed = [[0.5],
                   [0.5],
                   [1.2],
                   [0.9]]

        with tf.Session() as sess:
            sess.run(init)
            # Skip the IVs op
            out = sess.run(op, feed_dict={counts: counts_feed,
                                          v12: v12_feed,
                                          v34: v34_feed,
                                          v5: v5_feed})
        # Weights
        np.testing.assert_array_almost_equal(
            out[0], np.transpose(np.array([[[10., 0., 0., 0., 0., 0.],
                                            [0., 0., 11., 0., 0., 0.],
                                            [0., 0., 0., 0., 0., 12.],
                                            [0., 0., 0., 0., 13., 0.]]],
                                          dtype=np.float32), [1, 0, 2]))

        np.testing.assert_array_almost_equal(
            out[1], np.array([[10., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 13., 0., 0., 0., 0.]],
                             dtype=np.float32))

        np.testing.assert_array_almost_equal(
            out[2], np.array([[0., 0.],
                              [11., 0.],
                              [0., 0.],
                              [0., 0.]],
                             dtype=np.float32))

        np.testing.assert_array_almost_equal(
            out[3], np.array([[0.],
                              [0.],
                              [12.],
                              [0.]],
                             dtype=np.float32))

    @parameterized.expand([tuple(s) for s in itertools.product([False, True], [False, True])])
    def test_compute_mpe_path_noivs_multi_sums(self, log, matmul):
        v12 = spn.IVs(num_vars=2, num_vals=4)
        v34 = spn.ContVars(num_vars=2)
        v5 = spn.ContVars(num_vars=1)
        s = spn.SumsLayer((v12, [0, 5]), v34, (v12, [3]), v5, (v12, [0, 5]), v34,
                          (v12, [3]), v5, num_sums_or_sizes=2)
        w = s.generate_weights()
        counts = tf.placeholder(tf.float32, shape=(None, 2))

        spn.conf.sumslayer_count_with_matmul = matmul

        if log:
            op = s._compute_log_mpe_path(tf.identity(counts),
                                         w.get_log_value(),
                                         None,
                                         v12.get_log_value(),
                                         v34.get_log_value(),
                                         v12.get_log_value(),
                                         v5.get_log_value(),
                                         v12.get_log_value(),
                                         v34.get_log_value(),
                                         v12.get_log_value(),
                                         v5.get_log_value())
        else:
            op = s._compute_mpe_path(tf.identity(counts),
                                     w.get_value(),
                                     None,
                                     v12.get_value(),
                                     v34.get_value(),
                                     v12.get_value(),
                                     v5.get_value(),
                                     v12.get_value(),
                                     v34.get_value(),
                                     v12.get_value(),
                                     v5.get_value())

        # Index 2 to skip weights and IVs + i because of index of repeated input tensor
        print(op)
        [self.assertEqual(op[2+i], None) for i in [2, 4, 5, 6, 7]]
        op = [o for o in op if o is not None]

        init = w.initialize()
        counts_feed = [[10, 20],
                       [11, 21],
                       [12, 22],
                       [13, 23]]
        v12_feed = [[0, 1],
                    [1, 1],
                    [0, 0],
                    [3, 3]]
        v34_feed = [[0.1, 0.2],
                    [1.2, 0.2],
                    [0.1, 0.2],
                    [0.9, 0.8]]
        v5_feed = [[0.5],
                   [0.5],
                   [1.2],
                   [0.9]]

        with tf.Session() as sess:
            sess.run(init)
            # Skip the IVs op
            out = sess.run(op, feed_dict={counts: counts_feed,
                                          v12: v12_feed,
                                          v34: v34_feed,
                                          v5: v5_feed})
        # Weights
        np.testing.assert_array_almost_equal(
            out[0], np.transpose(np.array([[[10., 0., 0., 0., 0., 0.],
                                            [0., 0., 11., 0., 0., 0.],
                                            [0., 0., 0., 0., 0., 12.],
                                            [0., 0., 0., 0., 13., 0.]],
                                           [[20., 0., 0., 0., 0., 0.],
                                            [0., 0., 21., 0., 0., 0.],
                                            [0., 0., 0., 0., 0., 22.],
                                            [0., 0., 0., 0., 23., 0.]]],
                                          dtype=np.float32), [1, 0, 2]))

        np.testing.assert_array_almost_equal(
            out[1], np.array([[30., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 36., 0., 0., 0., 0.]],
                             dtype=np.float32))

        np.testing.assert_array_almost_equal(
            out[2], np.array([[0., 0.],
                              [32., 0.],
                              [0., 0.],
                              [0., 0.]],
                             dtype=np.float32))

        np.testing.assert_array_almost_equal(
            out[3], np.array([[0.],
                              [0.],
                              [34.],
                              [0.]],
                             dtype=np.float32))

    @parameterized.expand([('Non-log', False), ('Log', True)])
    def test_compute_mpe_path_ivs_single_sum(self, name, log):
        v12 = spn.IVs(num_vars=2, num_vals=4)
        v34 = spn.ContVars(num_vars=2)
        v5 = spn.ContVars(num_vars=1)
        s = spn.SumsLayer((v12, [0, 5]), v34, (v12, [3]), v5, num_sums_or_sizes=1)
        iv = s.generate_ivs()
        print(iv.get_value().shape)
        w = s.generate_weights()
        counts = tf.placeholder(tf.float32, shape=(None, 1))
        if log:
            op = s._compute_log_mpe_path(tf.identity(counts),
                                         w.get_log_value(),
                                         iv.get_log_value(),
                                         v12.get_log_value(),
                                         v34.get_log_value(),
                                         v12.get_log_value(),
                                         v5.get_log_value())
        else:
            op = s._compute_mpe_path(tf.identity(counts),
                                     w.get_value(),
                                     iv.get_value(),
                                     v12.get_value(),
                                     v34.get_value(),
                                     v12.get_value(),
                                     v5.get_value())
        [self.assertEqual(op[2+i], None) for i in [2]]
        op = [o for o in op if o is not None]

        init = w.initialize()
        counts_feed = [[10],
                       [11],
                       [12],
                       [13],
                       [14],
                       [15],
                       [16],
                       [17]]
        v12_feed = [[0, 1],
                    [1, 1],
                    [0, 0],
                    [3, 3],
                    [0, 1],
                    [1, 1],
                    [0, 0],
                    [3, 3]]
        v34_feed = [[0.1, 0.2],
                    [1.2, 0.2],
                    [0.1, 0.2],
                    [0.9, 0.8],
                    [0.1, 0.2],
                    [1.2, 0.2],
                    [0.1, 0.2],
                    [0.9, 0.8]]
        v5_feed = [[0.5],
                   [0.5],
                   [1.2],
                   [0.9],
                   [0.5],
                   [0.5],
                   [1.2],
                   [0.9]]
        ivs_feed = [[-1], [-1], [-1], [-1], [1], [2], [3], [1]]

        with tf.Session() as sess:
            sess.run(init)
            out = sess.run(op, feed_dict={counts: counts_feed,
                                          iv: ivs_feed,
                                          v12: v12_feed,
                                          v34: v34_feed,
                                          v5: v5_feed})
        # Weights
        np.testing.assert_array_almost_equal(
            out[0], np.transpose(np.array([[[10., 0., 0., 0., 0., 0.],
                                            [0., 0., 11., 0., 0., 0.],
                                            [0., 0., 0., 0., 0., 12.],
                                            [0., 0., 0., 0., 13., 0.],
                                            [0., 14., 0., 0., 0., 0.],
                                            [0., 0., 15., 0., 0., 0.],
                                            [0., 0., 0., 16., 0., 0.],
                                            [17., 0., 0., 0., 0., 0.]]],
                                          dtype=np.float32), [1, 0, 2]))

        # IVs
        np.testing.assert_array_almost_equal(
            out[1], np.array([[[10., 0., 0., 0., 0., 0.],
                              [0., 0., 11., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 12.],
                              [0., 0., 0., 0., 13., 0.],
                              [0., 14., 0., 0., 0., 0.],
                              [0., 0., 15., 0., 0., 0.],
                              [0., 0., 0., 16., 0., 0.],
                              [17., 0., 0., 0., 0., 0.]]],
                             dtype=np.float32).transpose(1, 0, 2))

        np.testing.assert_array_almost_equal(
            out[2], np.array([[10., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 13., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 14., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0.],
                              [17., 0., 0., 0., 0., 0., 0., 0.]],
                             dtype=np.float32))

        np.testing.assert_array_almost_equal(
            out[3], np.array([[0., 0.],
                              [11., 0.],
                              [0., 0.],
                              [0., 0.],
                              [0., 0.],
                              [15., 0.],
                              [0., 16.],
                              [0., 0.]],
                             dtype=np.float32))

        np.testing.assert_array_almost_equal(
            out[4], np.array([[0.],
                              [0.],
                              [12.],
                              [0.],
                              [0.],
                              [0.],
                              [0.],
                              [0.]],
                             dtype=np.float32))

    @parameterized.expand([('Non-log', False), ('Log', True)])
    def test_compute_mpe_path_ivs_multi_sums(self, name, log):
        v12 = spn.IVs(num_vars=2, num_vals=4)
        v34 = spn.ContVars(num_vars=2)
        v5 = spn.ContVars(num_vars=1)
        s = spn.SumsLayer((v12, [0, 5]), v34, (v12, [3]), v5, (v12, [0, 5]), v34,
                          (v12, [3]), v5, num_sums_or_sizes=2)
        iv = s.generate_ivs()
        w = s.generate_weights()
        counts = tf.placeholder(tf.float32, shape=(None, 2))
        if log:
            op = s._compute_log_mpe_path(tf.identity(counts),
                                         w.get_log_value(),
                                         iv.get_log_value(),
                                         v12.get_log_value(),
                                         v34.get_log_value(),
                                         v12.get_log_value(),
                                         v5.get_log_value(),
                                         v12.get_log_value(),
                                         v34.get_log_value(),
                                         v12.get_log_value(),
                                         v5.get_log_value())
        else:
            op = s._compute_mpe_path(tf.identity(counts),
                                     w.get_value(),
                                     iv.get_value(),
                                     v12.get_value(),
                                     v34.get_value(),
                                     v12.get_value(),
                                     v5.get_value(),
                                     v12.get_value(),
                                     v34.get_value(),
                                     v12.get_value(),
                                     v5.get_value())

        [self.assertEqual(op[2 + i], None) for i in [2, 4, 5, 6, 7]]
        op = [o for o in op if o is not None]

        init = w.initialize()
        counts_feed = [[10, 20],
                       [11, 21],
                       [12, 22],
                       [13, 23],
                       [14, 24],
                       [15, 25],
                       [16, 26],
                       [17, 27]]
        v12_feed = [[0, 1],
                    [1, 1],
                    [0, 0],
                    [3, 3],
                    [0, 1],
                    [1, 1],
                    [0, 0],
                    [3, 3]]
        v34_feed = [[0.1, 0.2],
                    [1.2, 0.2],
                    [0.1, 0.2],
                    [0.9, 0.8],
                    [0.1, 0.2],
                    [1.2, 0.2],
                    [0.1, 0.2],
                    [0.9, 0.8]]
        v5_feed = [[0.5],
                   [0.5],
                   [1.2],
                   [0.9],
                   [0.5],
                   [0.5],
                   [1.2],
                   [0.9]]
        ivs_feed = [[-1, -1],
                    [-1, -1],
                    [-1, -1],
                    [-1, -1],
                    [1, 1],
                    [2, 2],
                    [3, 3],
                    [1, 1]]

        with tf.Session() as sess:
            sess.run(init)
            out = sess.run(op, feed_dict={counts: counts_feed,
                                          iv: ivs_feed,
                                          v12: v12_feed,
                                          v34: v34_feed,
                                          v5: v5_feed})
        # Weights
        np.testing.assert_array_almost_equal(
            out[0], np.transpose(np.array([[[10., 0., 0., 0., 0., 0.],
                                            [0., 0., 11., 0., 0., 0.],
                                            [0., 0., 0., 0., 0., 12.],
                                            [0., 0., 0., 0., 13., 0.],
                                            [0., 14., 0., 0., 0., 0.],
                                            [0., 0., 15., 0., 0., 0.],
                                            [0., 0., 0., 16., 0., 0.],
                                            [17., 0., 0., 0., 0., 0.]],
                                           [[20., 0., 0., 0., 0., 0.],
                                            [0., 0., 21., 0., 0., 0.],
                                            [0., 0., 0., 0., 0., 22.],
                                            [0., 0., 0., 0., 23., 0.],
                                            [0., 24., 0., 0., 0., 0.],
                                            [0., 0., 25., 0., 0., 0.],
                                            [0., 0., 0., 26., 0., 0.],
                                            [27., 0., 0., 0., 0., 0.]]],
                                          dtype=np.float32), [1, 0, 2]))

        # IVs
        np.testing.assert_array_almost_equal(
            out[1], np.array([[[10., 0., 0., 0., 0., 0., 20., 0., 0., 0., 0., 0.],
                              [0., 0., 11., 0., 0., 0., 0., 0., 21., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 12., 0., 0., 0., 0., 0., 22.],
                              [0., 0., 0., 0., 13., 0., 0., 0., 0., 0., 23., 0.],
                              [0., 14., 0., 0., 0., 0., 0., 24., 0., 0., 0., 0.],
                              [0., 0., 15., 0., 0., 0., 0., 0., 25., 0., 0., 0.],
                              [0., 0., 0., 16., 0., 0., 0., 0., 0., 26., 0., 0.],
                              [17., 0., 0., 0., 0., 0., 27., 0., 0., 0., 0., 0.]]],
                             dtype=np.float32).transpose(1, 0, 2).reshape((8, 2, 6)))

        np.testing.assert_array_almost_equal(
            out[2], np.array([[30., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 36., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 38., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0., 0., 0.],
                              [44., 0., 0., 0., 0., 0., 0., 0.]],
                             dtype=np.float32))

        np.testing.assert_array_almost_equal(
            out[3], np.array([[0., 0.],
                              [32., 0.],
                              [0., 0.],
                              [0., 0.],
                              [0., 0.],
                              [40., 0.],
                              [0., 42.],
                              [0., 0.]],
                             dtype=np.float32))

        np.testing.assert_array_almost_equal(
            out[4], np.array([[0.],
                              [0.],
                              [34.],
                              [0.],
                              [0.],
                              [0.],
                              [0.],
                              [0.]],
                             dtype=np.float32))


if __name__ == '__main__':
    unittest.main()
