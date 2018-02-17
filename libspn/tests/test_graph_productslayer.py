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
import random
import collections
from context import libspn as spn


class TestNodesProductsLayer(unittest.TestCase):

    def tearDown(self):
        tf.reset_default_graph()

    def test_compute_vals(self):
        """Calculating value of ProductsLayer"""

        def test(input_sizes, num_or_size_prods, batch_size=7, indexed=True):
            self.tearDown()
            with self.subTest(input_sizes=input_sizes, batch_size=batch_size,
                              num_or_size_prods=num_or_size_prods,
                              indexed=indexed):
                # Calculate num-prods or prod-input-size, based on type and
                # values of the parameter 'num_or_size_prods'
                if isinstance(num_or_size_prods, int):
                    num_prods = num_or_size_prods
                    prod_input_sizes = [sum(input_sizes) // num_prods] * num_prods
                elif isinstance(num_or_size_prods, list):
                    num_prods = len(num_or_size_prods)
                    prod_input_sizes = num_or_size_prods

                max_prod_input_sizes = max(prod_input_sizes)

                inputs = []
                indices = []
                inputs_feed = collections.OrderedDict()
                # Create ContVar inputs of random size each
                for i_size in input_sizes:
                    if indexed:
                        # Choose a random size for an input node
                        num_vars = np.random.randint(low=i_size, high=i_size+5)
                    else:
                        num_vars = i_size

                    # Create a ContVars input node
                    inputs.append(spn.ContVars(num_vars=num_vars))
                    if indexed:
                        # Generate random indices for the created input
                        indices.append(random.sample(range(num_vars), k=i_size))
                    else:
                        indices.append(None)

                    # Generate random values for the inputs feed
                    inputs_feed[inputs[-1]] = \
                        np.random.random((batch_size, num_vars))

                # Create the ProductsLayer node
                p = spn.ProductsLayer(*list(zip(inputs, indices)),
                                      num_or_size_prods=num_or_size_prods)

                # Create value ops
                op_marg = p.get_value(spn.InferenceType.MARGINAL)
                op_log_marg = p.get_log_value(spn.InferenceType.MARGINAL)
                op_mpe = p.get_value(spn.InferenceType.MPE)
                op_log_mpe = p.get_log_value(spn.InferenceType.MPE)

                # Create an empty outputs ordered-dictionary
                outputs = collections.OrderedDict()

                # Create a session and execute the generated op
                with tf.Session() as sess:
                    outputs[op_marg] = sess.run(op_marg, feed_dict=inputs_feed)
                    outputs[op_log_marg] = sess.run(tf.exp(op_log_marg),
                                                    feed_dict=inputs_feed)
                    outputs[op_mpe] = sess.run(op_mpe, feed_dict=inputs_feed)
                    outputs[op_log_marg] = sess.run(tf.exp(op_log_mpe),
                                                    feed_dict=inputs_feed)

                # Calculate true-output
                # Concat all inputs_feed into a single matrix
                concatenated_inputs = np.hstack([(i_feed[:, ind] if indexed else
                                                  i_feed) for (key, i_feed), ind in
                                                 zip(inputs_feed.items(), indices)])
                # Split concatenated-inputs based on product-size
                inputs_per_prod = np.hsplit(concatenated_inputs, np.cumsum(prod_input_sizes))
                del inputs_per_prod[-1]

                # Pad those inputs, with value '1.', whose size < max-prod-input-size
                padded_values = \
                    np.stack([np.pad(inp, ((0, 0), (0, max_prod_input_sizes-inp.shape[1])),
                                     'constant', constant_values=(1.0)) for
                             inp in inputs_per_prod])

                # Reduce-prod and transpose to calculate true-output
                true_output = np.transpose(np.prod(padded_values, axis=-1))

                # Assert outputs with true_outputs
                for _, out in outputs.items():
                    np.testing.assert_array_almost_equal(
                        out, np.array(true_output, dtype=spn.conf.dtype.as_numpy_dtype()))

        def unit_value_product_test_cases():
            """Test cases with single input, with unit size, and a single product
               modelled """
            num_or_size_prods = [1, [1]]
            batch_sizes = [1, 7]
            for n_prod in num_or_size_prods:
                for b_size in batch_sizes:
                    test([1], n_prod, b_size, indexed=True)
                    test([1], n_prod, b_size, indexed=False)

        def multiple_values_products_test_cases():
            """Test cases with combinations of multiple inputs, with varying
               sizes, and multiple products modelled """
            input_sizes = [[10],
                           [1] * 10,
                           [2, 2, 2, 2, 2],
                           [5, 5],
                           [1, 2, 3, 4],
                           [4, 3, 2, 1],
                           [2, 1, 2, 3, 1, 1]]
            num_prods = [1, 10, 5, 2]
            size_prods = [[10],
                          [1] * 10,
                          [2, 2, 2, 2, 2],
                          [5, 5],
                          [1, 2, 3, 4],
                          [4, 3, 2, 1],
                          [2, 1, 2, 3, 1, 1]]
            num_or_size_prods = num_prods + size_prods
            batch_sizes = [1, 7]

            for n_prod in num_or_size_prods:
                for i_size in input_sizes:
                    for b_size in batch_sizes:
                        test(i_size, n_prod, b_size, indexed=True)
                        test(i_size, n_prod, b_size, indexed=False)

        # Run all test case combinations
        unit_value_product_test_cases()
        multiple_values_products_test_cases()

    # def test_comput_scope(self):
    #     """Calculating scope of ProductsLayer"""
    #     # Create graph
    #     v12 = spn.IVs(num_vars=2, num_vals=4, name="V12")
    #     v34 = spn.ContVars(num_vars=2, name="V34")
    #     s1 = spn.Sum((v12, [0, 1, 2, 3]), name="S1")
    #     s1.generate_ivs()
    #     s2 = spn.Sum((v12, [4, 5, 6, 7]), name="S2")
    #     ps1 = spn.ProductsLayer((v12, [0, 7]), (v12, [3, 4]), v34, num_or_size_prods=3,
    #                        name="Ps1")
    #     n1 = spn.Concat(s1, s2, (ps1, [2]), name="N1")
    #     n2 = spn.Concat((ps1, [0]), (ps1, [1]), name="N2")
    #     s3 = spn.Sum(ps1, name="S3")
    #     s3.generate_ivs()
    #     ps2 = spn.ProductsLayer((n1, [0, 1]), (n1, 2), (n2, 0), n2, (n2, 1), s3,
    #                        num_or_size_prods=4, name="Ps2")
    #     s4 = spn.Sum((ps2, 0), n2, name="S4")
    #     s5 = spn.Sum(ps2, name="S5")
    #     s6 = spn.Sum((ps2, [1, 3]), name="S6")
    #     s6.generate_ivs()
    #     ps3 = spn.ProductsLayer(s4, (n1, 2), name="Ps3")
    #     ps4 = spn.ProductsLayer(s4, s5, s6, s4, s5, s6, num_or_size_prods=2, name="Ps4")
    #     # Test
    #     self.assertListEqual(v12.get_scope(),
    #                          [spn.Scope(v12, 0), spn.Scope(v12, 0),
    #                           spn.Scope(v12, 0), spn.Scope(v12, 0),
    #                           spn.Scope(v12, 1), spn.Scope(v12, 1),
    #                           spn.Scope(v12, 1), spn.Scope(v12, 1)])
    #     self.assertListEqual(v34.get_scope(),
    #                          [spn.Scope(v34, 0), spn.Scope(v34, 1)])
    #     self.assertListEqual(s1.get_scope(),
    #                          [spn.Scope(v12, 0) | spn.Scope(s1.ivs.node, 0)])
    #     self.assertListEqual(s2.get_scope(),
    #                          [spn.Scope(v12, 1)])
    #     self.assertListEqual(ps1.get_scope(),
    #                          [spn.Scope(v12, 0) | spn.Scope(v12, 1),
    #                           spn.Scope(v12, 0) | spn.Scope(v12, 1),
    #                           spn.Scope(v34, 0) | spn.Scope(v34, 1)])
    #     self.assertListEqual(n1.get_scope(),
    #                          [spn.Scope(v12, 0) | spn.Scope(s1.ivs.node, 0),
    #                           spn.Scope(v12, 1),
    #                           spn.Scope(v34, 0) | spn.Scope(v34, 1)])
    #     self.assertListEqual(n2.get_scope(),
    #                          [spn.Scope(v12, 0) | spn.Scope(v12, 1),
    #                           spn.Scope(v12, 0) | spn.Scope(v12, 1)])
    #     self.assertListEqual(s3.get_scope(),
    #                          [spn.Scope(v12, 0) | spn.Scope(v12, 1) |
    #                           spn.Scope(v34, 0) | spn.Scope(v34, 1) |
    #                           spn.Scope(s3.ivs.node, 0)])
    #     self.assertListEqual(ps2.get_scope(),
    #                          [spn.Scope(v12, 0) | spn.Scope(v12, 1) |
    #                           spn.Scope(s1.ivs.node, 0),
    #                           spn.Scope(v12, 0) | spn.Scope(v12, 1) |
    #                           spn.Scope(v34, 0) | spn.Scope(v34, 1),
    #                           spn.Scope(v12, 0) | spn.Scope(v12, 1),
    #                           spn.Scope(v12, 0) | spn.Scope(v12, 1) |
    #                           spn.Scope(v34, 0) | spn.Scope(v34, 1) |
    #                           spn.Scope(s3.ivs.node, 0)])
    #     self.assertListEqual(s4.get_scope(),
    #                          [spn.Scope(v12, 0) | spn.Scope(v12, 1) |
    #                           spn.Scope(s1.ivs.node, 0)])
    #     self.assertListEqual(s5.get_scope(),
    #                          [spn.Scope(v12, 0) | spn.Scope(v12, 1) |
    #                           spn.Scope(s1.ivs.node, 0) | spn.Scope(v34, 0) |
    #                           spn.Scope(v34, 1) | spn.Scope(s3.ivs.node, 0)])
    #     self.assertListEqual(s6.get_scope(),
    #                          [spn.Scope(v12, 0) | spn.Scope(v12, 1) |
    #                           spn.Scope(v34, 0) | spn.Scope(v34, 1) |
    #                           spn.Scope(s3.ivs.node, 0) |
    #                           spn.Scope(s6.ivs.node, 0)])
    #     self.assertListEqual(ps3.get_scope(),
    #                          [spn.Scope(v12, 0) | spn.Scope(v12, 1) |
    #                           spn.Scope(s1.ivs.node, 0) |
    #                           spn.Scope(v34, 0) | spn.Scope(v34, 1)])
    #     self.assertListEqual(ps4.get_scope(),
    #                          [spn.Scope(v12, 0) | spn.Scope(v12, 1) |
    #                           spn.Scope(v34, 0) | spn.Scope(v34, 1) |
    #                           spn.Scope(s1.ivs.node, 0) |
    #                           spn.Scope(s3.ivs.node, 0) |
    #                           spn.Scope(s6.ivs.node, 0),
    #                           spn.Scope(v12, 0) | spn.Scope(v12, 1) |
    #                           spn.Scope(v34, 0) | spn.Scope(v34, 1) |
    #                           spn.Scope(s1.ivs.node, 0) |
    #                           spn.Scope(s3.ivs.node, 0) |
    #                           spn.Scope(s6.ivs.node, 0)])
    #
    # def test_compute_valid(self):
    #     """Calculating validity of ProductsLayer"""
    #     v12 = spn.IVs(num_vars=2, num_vals=3)
    #     v345 = spn.IVs(num_vars=3, num_vals=3)
    #     v678 = spn.ContVars(num_vars=3)
    #     v910 = spn.ContVars(num_vars=2)
    #     p1 = spn.ProductsLayer((v12, 0), (v12, 4), (v12, 0), (v12, 5),
    #                       (v12, 1), (v12, 4), (v12, 1), (v12, 5), num_or_size_prods=4)
    #     p2 = spn.ProductsLayer((v12, 3), (v345, 0), (v12, 3), (v345, 1), (v12, 3), (v345, 2),
    #                       (v12, 5), (v345, 0), (v12, 5), (v345, 1), (v12, 5), (v345, 2),
    #                       num_or_size_prods=6)
    #     p3 = spn.ProductsLayer((v345, 0), (v345, 3), (v345, 6), (v345, 0), (v345, 3),
    #                       (v345, 7), (v345, 0), (v345, 3), (v345, 8), (v345, 0),
    #                       (v345, 4), (v345, 6), (v345, 0), (v345, 4), (v345, 7),
    #                       (v345, 0), (v345, 4), (v345, 8), (v345, 0), (v345, 5),
    #                       (v345, 6), (v345, 0), (v345, 5), (v345, 7), (v345, 0),
    #                       (v345, 5), (v345, 8), (v345, 1), (v345, 3), (v345, 6),
    #                       (v345, 1), (v345, 3), (v345, 7), (v345, 1), (v345, 3),
    #                       (v345, 8), (v345, 1), (v345, 4), (v345, 6), (v345, 1),
    #                       (v345, 4), (v345, 7), (v345, 1), (v345, 4), (v345, 8),
    #                       (v345, 1), (v345, 5), (v345, 6), (v345, 1), (v345, 5),
    #                       (v345, 7), (v345, 1), (v345, 5), (v345, 8), (v345, 2),
    #                       (v345, 3), (v345, 6), (v345, 2), (v345, 3), (v345, 7),
    #                       (v345, 2), (v345, 3), (v345, 8), (v345, 2), (v345, 4),
    #                       (v345, 6), (v345, 2), (v345, 4), (v345, 7), (v345, 2),
    #                       (v345, 4), (v345, 8), (v345, 2), (v345, 5), (v345, 6),
    #                       (v345, 2), (v345, 5), (v345, 7), (v345, 2), (v345, 5),
    #                       (v345, 8), num_or_size_prods=27)
    #     p4 = spn.ProductsLayer((v345, 6), (v678, 0), (v345, 6), (v678, 1),
    #                       (v345, 8), (v678, 0), (v345, 8), (v678, 1), num_or_size_prods=4)
    #     p5 = spn.ProductsLayer((v678, 1), (v910, 0), (v678, 1), (v910, 1), num_or_size_prods=2)
    #     p6_1 = spn.ProductsLayer((v678, 0), (v910, 0), (v678, 0), (v910, 1),
    #                         (v678, 1), (v910, 0), (v678, 1), (v910, 1),
    #                         (v678, 2), (v910, 0), (v678, 2), (v910, 1), num_or_size_prods=6)
    #     p6_2 = spn.ProductsLayer((v678, 0), (v910, 0), (v678, 1), (v910, 0),
    #                         (v678, 1), (v910, 1), (v678, 2), (v910, 0), num_or_size_prods=4)
    #     p7_1 = spn.ProductsLayer((v678, [0, 1, 2]))
    #     p7_2 = spn.ProductsLayer(v678)
    #     p8_1 = spn.ProductsLayer((v910, [0]), (v910, [1]))
    #     p8_2 = spn.ProductsLayer(v910, num_or_size_prods=2)
    #     self.assertTrue(p1.is_valid())
    #     self.assertTrue(p2.is_valid())
    #     self.assertTrue(p3.is_valid())
    #     self.assertTrue(p4.is_valid())
    #     self.assertTrue(p5.is_valid())
    #     self.assertTrue(p6_1.is_valid())
    #     self.assertTrue(p6_2.is_valid())
    #     self.assertTrue(p7_1.is_valid())
    #     self.assertTrue(p7_2.is_valid())
    #     self.assertTrue(p8_1.is_valid())
    #     self.assertTrue(p8_2.is_valid())
    #     p9 = spn.ProductsLayer((v12, 0), (v12, 1), (v12, 0), (v12, 2),
    #                       (v12, 1), (v12, 1), (v12, 1), (v12, 2), num_or_size_prods=4)
    #     p10 = spn.ProductsLayer((v12, 3), (v345, 0), (v345, 0),
    #                        (v12, 3), (v345, 0), (v345, 1),
    #                        (v12, 3), (v345, 0), (v345, 2),
    #                        (v12, 4), (v345, 0), (v345, 0),
    #                        (v12, 4), (v345, 0), (v345, 1),
    #                        (v12, 4), (v345, 0), (v345, 2),
    #                        (v12, 5), (v345, 0), (v345, 0),
    #                        (v12, 5), (v345, 0), (v345, 1),
    #                        (v12, 5), (v345, 0), (v345, 2), num_or_size_prods=9)
    #     p11_1 = spn.ProductsLayer((v345, 3), (v678, 0), (v678, 0),
    #                          (v345, 5), (v678, 0), (v678, 0), num_or_size_prods=2)
    #     p11_2 = spn.ProductsLayer((v345, 3), (v678, 0), (v678, 0),
    #                          (v345, 5), (v678, 0), (v678, 0), num_or_size_prods=3)
    #     p12 = spn.ProductsLayer((v910, 1), (v910, 1), num_or_size_prods=1)
    #     p13_1 = spn.ProductsLayer((v910, 0), (v910, 0), (v910, 0), (v910, 1),
    #                          (v910, 1), (v910, 0), (v910, 1), (v910, 1), num_or_size_prods=4)
    #     p13_2 = spn.ProductsLayer((v910, 0), (v910, 0), (v910, 0), (v910, 1),
    #                          (v910, 1), (v910, 0), (v910, 1), (v910, 1), num_or_size_prods=1)
    #     self.assertFalse(p9.is_valid())
    #     self.assertFalse(p10.is_valid())
    #     self.assertFalse(p11_1.is_valid())
    #     self.assertFalse(p11_2.is_valid())
    #     self.assertFalse(p12.is_valid())
    #     self.assertFalse(p13_1.is_valid())
    #     self.assertFalse(p13_2.is_valid())

    def test_compute_mpe_path(self):
        """Calculating path of ProductsLayer"""
        num_vals = 2

        def test(input_sizes, num_or_size_prods, batch_size=7, indexed=True):
            self.tearDown()
            with self.subTest(input_sizes=input_sizes, num_or_size_prods=num_or_size_prods):
                # Calculate num-prods or prod-input-size, based on type and
                # values of the parameter 'num_or_size_prods'
                if isinstance(num_or_size_prods, int):
                    num_prods = num_or_size_prods
                    prod_input_sizes = [sum(input_sizes) // num_prods] * num_prods
                elif isinstance(num_or_size_prods, list):
                    num_prods = len(num_or_size_prods)
                    prod_input_sizes = num_or_size_prods

                inputs = []
                indices = []
                true_outputs = []
                # Create randon (IVs or ContVar) inputs or random size each
                for i_size in input_sizes:
                    if indexed:
                        # Choose a random size for an input node
                        num_vars = np.random.randint(low=i_size, high=i_size+5)
                    else:
                        num_vars = i_size
                    # Randomly choose to create either a IVs or a ContVar input
                    if random.choice([True, False]):
                        # Create an IVs input node
                        inputs.append(spn.IVs(num_vars=num_vars,
                                              num_vals=num_vals))
                        # Generate random indices for the created input
                        indices.append(random.sample(range(num_vars*num_vals),
                                                     k=i_size))
                        # Create a Zeros matrix, with the same size as the created
                        # input node
                        true_outputs.append(np.zeros((batch_size, num_vars*num_vals)))
                    else:
                        # Create anContVars input node
                        inputs.append(spn.ContVars(num_vars=num_vars))
                        # Generate random indices for the created input
                        indices.append(random.sample(range(num_vars), k=i_size))
                        # Create a Zeros matrix, with the same size as the created
                        # input node
                        true_outputs.append(np.zeros((batch_size, num_vars)))

                # Create the ProductsLayer node
                p = spn.ProductsLayer(*list(zip(inputs, indices)),
                                      num_or_size_prods=num_or_size_prods)

                # Create counts as a placeholder
                counts = tf.placeholder(tf.float32, shape=(None, num_prods))

                # Create mpe path op
                op = p._compute_mpe_path(tf.identity(counts),
                                         *[inp.get_value() for inp in inputs])

                # Generate random counts feed
                coutnts_feed = np.random.randint(100, size=(batch_size, num_prods))

                # Create a session and execute the generated op
                with tf.Session() as sess:
                    outputs = sess.run(op, feed_dict={counts: coutnts_feed})

                # Calculate true-output
                # (1) Split counts into num_prod sub-matrices
                split_counts = np.split(coutnts_feed, num_prods, axis=1)

                # (2) Tile each counts matrix to the size of prod-input
                tiled_counts_list = [np.tile(sc, [1, p_inp_size])
                                     for sc, p_inp_size in zip(split_counts,
                                                               prod_input_sizes)]

                # (3) Concat all tiled counts to a single matrix
                concatenated_counts = np.concatenate(tiled_counts_list, axis=1)

                # (4) Split the concatenated counts matrix into input-sizes
                value_counts = np.split(concatenated_counts, np.cumsum(input_sizes),
                                        axis=1)

                # (5) Scatter each count matrix to inputs
                for t_out, i_indices, v_counts in zip(true_outputs, indices,
                                                      value_counts):
                    t_out[:, i_indices] = v_counts

                # Assert outputs with true_outputs
                for out, t_out in zip(outputs, true_outputs):
                    np.testing.assert_array_almost_equal(out, t_out)

        def unit_value_product_test_cases():
            """Test cases with single input, with unit size, and a single product
               modelled """
            num_or_size_prods = [1, [1]]
            batch_sizes = [1, 7]
            for n_prod in num_or_size_prods:
                for b_size in batch_sizes:
                    test([1], n_prod, b_size, indexed=True)
                    test([1], n_prod, b_size, indexed=False)

        def multiple_values_products_test_cases():
            """Test cases with combinations of multiple inputs, with varying
               sizes, and multiple products modelled """
            input_sizes = [[10],
                           [1] * 10,
                           [2, 2, 2, 2, 2],
                           [5, 5],
                           [1, 2, 3, 4],
                           [4, 3, 2, 1],
                           [2, 1, 2, 3, 1, 1]]
            num_prods = [1, 10, 5, 2]
            size_prods = [[10],
                          [1] * 10,
                          [2, 2, 2, 2, 2],
                          [5, 5],
                          [1, 2, 3, 4],
                          [4, 3, 2, 1],
                          [2, 1, 2, 3, 1, 1]]
            num_or_size_prods = num_prods + size_prods
            batch_sizes = [1, 7]

            for n_prod in num_or_size_prods:
                for i_size in input_sizes:
                    for b_size in batch_sizes:
                        test(i_size, n_prod, b_size, indexed=True)
                        test(i_size, n_prod, b_size, indexed=False)

        # Run all test case combinations
        unit_value_product_test_cases()
        multiple_values_products_test_cases()


if __name__ == '__main__':
    unittest.main()
