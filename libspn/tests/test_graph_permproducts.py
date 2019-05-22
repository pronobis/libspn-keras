#!/usr/bin/env python3

import unittest
import tensorflow as tf
import numpy as np
from context import libspn as spn


class TestNodesPermuteProducts(tf.test.TestCase):

    def test_compute_value(self):
        """Calculating value of PermuteProducts"""
        def test(inputs, feed, output):
            with self.subTest(inputs=inputs, feed=feed):
                n = spn.PermuteProducts(*inputs)
                op = n.get_value(spn.InferenceType.MARGINAL)
                op_log = n.get_log_value(spn.InferenceType.MARGINAL)
                op_mpe = n.get_value(spn.InferenceType.MPE)
                op_log_mpe = n.get_log_value(spn.InferenceType.MPE)
                with self.test_session() as sess:
                    out = sess.run(op, feed_dict=feed)
                    out_log = sess.run(tf.exp(op_log), feed_dict=feed)
                    out_mpe = sess.run(op_mpe, feed_dict=feed)
                    out_log_mpe = sess.run(tf.exp(op_log_mpe), feed_dict=feed)
                np.testing.assert_array_almost_equal(
                    out,
                    np.array(output, dtype=spn.conf.dtype.as_numpy_dtype()))
                np.testing.assert_array_almost_equal(
                    out_log,
                    np.array(output, dtype=spn.conf.dtype.as_numpy_dtype()))
                np.testing.assert_array_almost_equal(
                    out_mpe,
                    np.array(output, dtype=spn.conf.dtype.as_numpy_dtype()))
                np.testing.assert_array_almost_equal(
                    out_log_mpe,
                    np.array(output, dtype=spn.conf.dtype.as_numpy_dtype()))

        # Create inputs
        v1 = spn.RawLeaf(num_vars=3)
        v2 = spn.RawLeaf(num_vars=3)
        v3 = spn.RawLeaf(num_vars=3)

        # Multiple Product nodes - Common input Sizes
        # -------------------------------------------

        # Case 1: No. of inputs > Input sizes
        # No. of inputs = 3
        # Input sizes = [2, 2, 2] --> {O O | O O | O O}

        # Multi-element batch
        test([(v1, [0, 1]), (v2, [1, 2]), (v3, [0, 2])],
             {v1: [[0.1, 0.2, 0.3],       # 0.1  0.2
                   [0.4, 0.5, 0.6]],      # 0.4  0.5
              v2: [[0.7, 0.8, 0.9],       # 0.8  0.9
                   [0.11, 0.12, 0.13]],   # 0.12 0.13
              v3: [[0.14, 0.15, 0.16],    # 0.14 0.16
                   [0.17, 0.18, 0.19]]},  # 0.17 0.19
             [[(0.1 * 0.8 * 0.14), (0.1 * 0.8 * 0.16), (0.1 * 0.9 * 0.14),
               (0.1 * 0.9 * 0.16), (0.2 * 0.8 * 0.14), (0.2 * 0.8 * 0.16),
               (0.2 * 0.9 * 0.14), (0.2 * 0.9 * 0.16)],
              [(0.4 * 0.12 * 0.17), (0.4 * 0.12 * 0.19), (0.4 * 0.13 * 0.17),
               (0.4 * 0.13 * 0.19), (0.5 * 0.12 * 0.17), (0.5 * 0.12 * 0.19),
               (0.5 * 0.13 * 0.17), (0.5 * 0.13 * 0.19)]])

        # Single-element batch
        test([(v1, [0, 1]), (v2, [1, 2]), (v3, [0, 2])],
             {v1: [[0.1, 0.2, 0.3]],      # 0.1  0.2
              v2: [[0.7, 0.8, 0.9]],      # 0.8  0.9
              v3: [[0.14, 0.15, 0.16]]},   # 0.14 0.16
             [[(0.1 * 0.8 * 0.14), (0.1 * 0.8 * 0.16), (0.1 * 0.9 * 0.14),
               (0.1 * 0.9 * 0.16), (0.2 * 0.8 * 0.14), (0.2 * 0.8 * 0.16),
               (0.2 * 0.9 * 0.14), (0.2 * 0.9 * 0.16)]])

        # Case 2: No. of inputs < Input sizes
        # No. of inputs = 2
        # Input sizes = [3, 3] --> {O O O | O O O}

        # Multi-element batch
        test([v1, v2],
             {v1: [[0.1, 0.2, 0.3],
                   [0.4, 0.5, 0.6]],
              v2: [[0.7, 0.8, 0.9],
                   [0.11, 0.12, 0.13]]},
             [[(0.1 * 0.7), (0.1 * 0.8), (0.1 * 0.9),
               (0.2 * 0.7), (0.2 * 0.8), (0.2 * 0.9),
               (0.3 * 0.7), (0.3 * 0.8), (0.3 * 0.9)],
              [(0.4 * 0.11), (0.4 * 0.12), (0.4 * 0.13),
               (0.5 * 0.11), (0.5 * 0.12), (0.5 * 0.13),
               (0.6 * 0.11), (0.6 * 0.12), (0.6 * 0.13)]])

        # Single-element batch
        test([v1, v2],
             {v1: [[0.1, 0.2, 0.3]],
              v2: [[0.7, 0.8, 0.9]]},
             [[(0.1 * 0.7), (0.1 * 0.8), (0.1 * 0.9),
               (0.2 * 0.7), (0.2 * 0.8), (0.2 * 0.9),
               (0.3 * 0.7), (0.3 * 0.8), (0.3 * 0.9)]])

        # Case 3: No. of inputs == Input sizes
        # No. of inputs = 3
        # Input sizes = [3, 3, 3] --> {O O O | O O O | O O O}

        # Multi-element batch
        test([v1, v2, v3],
             {v1: [[0.1, 0.2, 0.3],
                   [0.4, 0.5, 0.6]],
              v2: [[0.7, 0.8, 0.9],
                   [0.11, 0.12, 0.13]],
              v3: [[0.14, 0.15, 0.16],
                   [0.17, 0.18, 0.19]]},
             [[(0.1 * 0.7 * 0.14), (0.1 * 0.7 * 0.15), (0.1 * 0.7 * 0.16),
               (0.1 * 0.8 * 0.14), (0.1 * 0.8 * 0.15), (0.1 * 0.8 * 0.16),
               (0.1 * 0.9 * 0.14), (0.1 * 0.9 * 0.15), (0.1 * 0.9 * 0.16),
               (0.2 * 0.7 * 0.14), (0.2 * 0.7 * 0.15), (0.2 * 0.7 * 0.16),
               (0.2 * 0.8 * 0.14), (0.2 * 0.8 * 0.15), (0.2 * 0.8 * 0.16),
               (0.2 * 0.9 * 0.14), (0.2 * 0.9 * 0.15), (0.2 * 0.9 * 0.16),
               (0.3 * 0.7 * 0.14), (0.3 * 0.7 * 0.15), (0.3 * 0.7 * 0.16),
               (0.3 * 0.8 * 0.14), (0.3 * 0.8 * 0.15), (0.3 * 0.8 * 0.16),
               (0.3 * 0.9 * 0.14), (0.3 * 0.9 * 0.15), (0.3 * 0.9 * 0.16)],
              [(0.4 * 0.11 * 0.17), (0.4 * 0.11 * 0.18), (0.4 * 0.11 * 0.19),
               (0.4 * 0.12 * 0.17), (0.4 * 0.12 * 0.18), (0.4 * 0.12 * 0.19),
               (0.4 * 0.13 * 0.17), (0.4 * 0.13 * 0.18), (0.4 * 0.13 * 0.19),
               (0.5 * 0.11 * 0.17), (0.5 * 0.11 * 0.18), (0.5 * 0.11 * 0.19),
               (0.5 * 0.12 * 0.17), (0.5 * 0.12 * 0.18), (0.5 * 0.12 * 0.19),
               (0.5 * 0.13 * 0.17), (0.5 * 0.13 * 0.18), (0.5 * 0.13 * 0.19),
               (0.6 * 0.11 * 0.17), (0.6 * 0.11 * 0.18), (0.6 * 0.11 * 0.19),
               (0.6 * 0.12 * 0.17), (0.6 * 0.12 * 0.18), (0.6 * 0.12 * 0.19),
               (0.6 * 0.13 * 0.17), (0.6 * 0.13 * 0.18), (0.6 * 0.13 * 0.19)]])

        # Single-element batch
        test([v1, v2, v3],
             {v1: [[0.1, 0.2, 0.3]],
              v2: [[0.7, 0.8, 0.9]],
              v3: [[0.14, 0.15, 0.16]]},
             [[(0.1 * 0.7 * 0.14), (0.1 * 0.7 * 0.15), (0.1 * 0.7 * 0.16),
               (0.1 * 0.8 * 0.14), (0.1 * 0.8 * 0.15), (0.1 * 0.8 * 0.16),
               (0.1 * 0.9 * 0.14), (0.1 * 0.9 * 0.15), (0.1 * 0.9 * 0.16),
               (0.2 * 0.7 * 0.14), (0.2 * 0.7 * 0.15), (0.2 * 0.7 * 0.16),
               (0.2 * 0.8 * 0.14), (0.2 * 0.8 * 0.15), (0.2 * 0.8 * 0.16),
               (0.2 * 0.9 * 0.14), (0.2 * 0.9 * 0.15), (0.2 * 0.9 * 0.16),
               (0.3 * 0.7 * 0.14), (0.3 * 0.7 * 0.15), (0.3 * 0.7 * 0.16),
               (0.3 * 0.8 * 0.14), (0.3 * 0.8 * 0.15), (0.3 * 0.8 * 0.16),
               (0.3 * 0.9 * 0.14), (0.3 * 0.9 * 0.15), (0.3 * 0.9 * 0.16)]])

        # Multiple Product nodes - Varying input Sizes
        # --------------------------------------------

        # Case 4: Ascending input sizes (2 inputs)
        # No. of inputs = 2
        # Input sizes = [2, 3] --> {O O | O O O}

        # Multi-element batch
        test([(v1, [0, 2]), v2],
             {v1: [[0.1, 0.2, 0.3],      # 0.1  0.3
                   [0.4, 0.5, 0.6]],     # 0.4  0.6
              v2: [[0.7, 0.8, 0.9],
                   [0.11, 0.12, 0.13]]},
             [[(0.1 * 0.7), (0.1 * 0.8), (0.1 * 0.9),
               (0.3 * 0.7), (0.3 * 0.8), (0.3 * 0.9)],
              [(0.4 * 0.11), (0.4 * 0.12), (0.4 * 0.13),
               (0.6 * 0.11), (0.6 * 0.12), (0.6 * 0.13)]])

        # Single-element batch
        test([(v1, [0, 2]), v2],
             {v1: [[0.1, 0.2, 0.3]],      # 0.1  0.3
              v2: [[0.7, 0.8, 0.9]]},
             [[(0.1 * 0.7), (0.1 * 0.8), (0.1 * 0.9),
               (0.3 * 0.7), (0.3 * 0.8), (0.3 * 0.9)]])

        # Case 5: Descending input sizes (2 inputs)
        # No. of inputs = 2
        # Input sizes = [3, 2] --> {O O O | O O}

        # Multi-element batch
        test([v1, (v2, [0, 1])],
             {v1: [[0.1, 0.2, 0.3],
                   [0.4, 0.5, 0.6]],
              v2: [[0.7, 0.8, 0.9],       # 0.7  0.8
                   [0.11, 0.12, 0.13]]},  # 0.11 0.12
             [[(0.1 * 0.7), (0.1 * 0.8),
               (0.2 * 0.7), (0.2 * 0.8),
               (0.3 * 0.7), (0.3 * 0.8)],
              [(0.4 * 0.11), (0.4 * 0.12),
               (0.5 * 0.11), (0.5 * 0.12),
               (0.6 * 0.11), (0.6 * 0.12)]])

        # Single-element batch
        test([v1, (v2, [0, 1])],
             {v1: [[0.1, 0.2, 0.3]],
              v2: [[0.7, 0.8, 0.9]]},       # 0.7  0.8
             [[(0.1 * 0.7), (0.1 * 0.8),
               (0.2 * 0.7), (0.2 * 0.8),
               (0.3 * 0.7), (0.3 * 0.8)]])

        # Case 6: Ascending input sizes (3 inputs)
        # No. of inputs = 3
        # Input sizes = [1, 2, 3] --> {O | O O | O O O}

        # Multi-element batch
        test([(v1, [0]), (v2, [1, 2]), v3],
             {v1: [[0.1, 0.2, 0.3],       # 0.1
                   [0.4, 0.5, 0.6]],      # 0.4
              v2: [[0.7, 0.8, 0.9],       # 0.8  0.9
                   [0.11, 0.12, 0.13]],   # 0.12 0.13
              v3: [[0.14, 0.15, 0.16],
                   [0.17, 0.18, 0.19]]},
             [[(0.1 * 0.8 * 0.14), (0.1 * 0.8 * 0.15), (0.1 * 0.8 * 0.16),
               (0.1 * 0.9 * 0.14), (0.1 * 0.9 * 0.15), (0.1 * 0.9 * 0.16)],
              [(0.4 * 0.12 * 0.17), (0.4 * 0.12 * 0.18), (0.4 * 0.12 * 0.19),
               (0.4 * 0.13 * 0.17), (0.4 * 0.13 * 0.18), (0.4 * 0.13 * 0.19)]])

        # Single-element batch
        test([(v1, [0]), (v2, [1, 2]), v3],
             {v1: [[0.1, 0.2, 0.3]],       # 0.1
              v2: [[0.7, 0.8, 0.9]],       # 0.8  0.9
              v3: [[0.14, 0.15, 0.16]]},
             [[(0.1 * 0.8 * 0.14), (0.1 * 0.8 * 0.15), (0.1 * 0.8 * 0.16),
               (0.1 * 0.9 * 0.14), (0.1 * 0.9 * 0.15), (0.1 * 0.9 * 0.16)]])

        # Case 7: Descending input sizes (3 inputs)
        # No. of inputs = 3
        # Input sizes = [3, 2, 1] --> {O O O | O O | O}

        # Multi-element batch
        test([v1, (v2, [0, 2]), (v3, [1])],
             {v1: [[0.1, 0.2, 0.3],
                   [0.4, 0.5, 0.6]],
              v2: [[0.7, 0.8, 0.9],       # 0.7  0.9
                   [0.11, 0.12, 0.13]],   # 0.11 0.13
              v3: [[0.14, 0.15, 0.16],    # 0.15
                   [0.17, 0.18, 0.19]]},  # 0.18
             [[(0.1 * 0.7 * 0.15),  (0.1 * 0.9 * 0.15),  (0.2 * 0.7 * 0.15),
               (0.2 * 0.9 * 0.15),  (0.3 * 0.7 * 0.15),  (0.3 * 0.9 * 0.15)],
              [(0.4 * 0.11 * 0.18), (0.4 * 0.13 * 0.18), (0.5 * 0.11 * 0.18),
               (0.5 * 0.13 * 0.18), (0.6 * 0.11 * 0.18), (0.6 * 0.13 * 0.18)]])

        # Single-element batch
        test([v1, (v2, [0, 2]), (v3, [1])],
             {v1: [[0.1, 0.2, 0.3]],
              v2: [[0.7, 0.8, 0.9]],      # 0.7  0.9
              v3: [[0.14, 0.15, 0.16]]},  # 0.18
             [[(0.1 * 0.7 * 0.15), (0.1 * 0.9 * 0.15), (0.2 * 0.7 * 0.15),
               (0.2 * 0.9 * 0.15), (0.3 * 0.7 * 0.15), (0.3 * 0.9 * 0.15)]])

        # Single Product node
        # -------------------

        # Case 8: Multiple inputs, each with size 1
        # No. of inputs = 3
        # Input sizes = [1, 1, 1] --> {O | O | O}

        # Multi-element batch
        test([(v1, [1]), (v2, [2]), (v3, [0])],
             {v1: [[0.1, 0.2, 0.3],       # 0.2
                   [0.4, 0.5, 0.6]],      # 0.5
              v2: [[0.7, 0.8, 0.9],       # 0.9
                   [0.11, 0.12, 0.13]],   # 0.13
              v3: [[0.14, 0.15, 0.16],    # 0.14
                   [0.17, 0.18, 0.19]]},  # 0.17
             [[(0.2 * 0.9 * 0.14)],
              [(0.5 * 0.13 * 0.17)]])

        # Single-element batch
        test([(v1, [1]), (v2, [2]), (v3, [0])],
             {v1: [[0.1, 0.2, 0.3]],      # 0.2
              v2: [[0.7, 0.8, 0.9]],      # 0.9
              v3: [[0.14, 0.15, 0.16]]},  # 0.14
             [[(0.2 * 0.9 * 0.14)]])

        # Case 9: Single input with size > 1
        # No. of inputs = 1
        # Input sizes = [3] --> {O O O}

        # Multi-element batch
        test([v1],
             {v1: [[0.1, 0.2, 0.3],
                   [0.4, 0.5, 0.6]]},
             [[(0.1 * 0.2 * 0.3)],
              [(0.4 * 0.5 * 0.6)]])

        # Single-element batch
        test([v1],
             {v1: [[0.1, 0.2, 0.3]]},
             [[(0.1 * 0.2 * 0.3)]])

        # Case 10: Single input with size = 1
        # No. of inputs = 1
        # Input sizes = [1] --> {O}

        # Multi-element batch
        test([(v2, [1])],
             {v2: [[0.7, 0.8, 0.9],       # 0.8
                   [0.11, 0.12, 0.13]]},  # 0.12
             [[0.8],
              [0.12]])

        # Single-element batch
        test([(v2, [2])],
             {v2: [[0.7, 0.8, 0.9]]},       # 0.9
             [[0.9]])

    def test_comput_scope(self):
        """Calculating scope of PermuteProducts"""
        # Create graph
        v12 = spn.IndicatorLeaf(num_vars=2, num_vals=4, name="V12")
        v34 = spn.RawLeaf(num_vars=2, name="V34")
        s1 = spn.Sum((v12, [0, 1, 2, 3]), name="S1")
        s1.generate_latent_indicators()
        s2 = spn.Sum((v12, [4, 5, 6, 7]), name="S2")
        p1 = spn.Product((v12, [0, 7]), name="P1")
        p2 = spn.Product((v12, [3, 4]), name="P2")
        p3 = spn.Product(v34, name="P3")
        n1 = spn.Concat(s1, s2, p3, name="N1")
        n2 = spn.Concat(p1, p2, name="N2")
        pp1 = spn.PermuteProducts(n1, n2, name="PP1")  # num_prods = 6
        pp2 = spn.PermuteProducts((n1, [0, 1]), (n2, [0]), name="PP2")  # num_prods = 2
        pp3 = spn.PermuteProducts(n2, p3, name="PP3")  # num_prods = 2
        pp4 = spn.PermuteProducts(p2, p3, name="PP4")  # num_prods = 1
        pp5 = spn.PermuteProducts((n2, [0, 1]), name="PP5")  # num_prods = 1
        pp6 = spn.PermuteProducts(p3, name="PP6")  # num_prods = 1
        n3 = spn.Concat((pp1, [0, 2, 3]), pp2, pp4, name="N3")
        s3 = spn.Sum((pp1, [0, 2, 4]), (pp1, [1, 3, 5]), pp2, pp3, (pp4, 0),
                     pp5, pp6, name="S3")
        s3.generate_latent_indicators()
        n4 = spn.Concat((pp3, [0, 1]), pp5, (pp6, 0), name="N4")
        pp7 = spn.PermuteProducts(n3, s3, n4, name="PP7")  # num_prods = 24
        pp8 = spn.PermuteProducts(n3, name="PP8")  # num_prods = 1
        pp9 = spn.PermuteProducts((n4, [0, 1, 2, 3]), name="PP9")  # num_prods = 1
        # Test
        self.assertListEqual(v12.get_scope(),
                             [spn.Scope(v12, 0), spn.Scope(v12, 0),
                              spn.Scope(v12, 0), spn.Scope(v12, 0),
                              spn.Scope(v12, 1), spn.Scope(v12, 1),
                              spn.Scope(v12, 1), spn.Scope(v12, 1)])
        self.assertListEqual(v34.get_scope(),
                             [spn.Scope(v34, 0), spn.Scope(v34, 1)])
        self.assertListEqual(s1.get_scope(),
                             [spn.Scope(v12, 0) | spn.Scope(s1.latent_indicators.node, 0)])
        self.assertListEqual(s2.get_scope(),
                             [spn.Scope(v12, 1)])
        self.assertListEqual(p1.get_scope(),
                             [spn.Scope(v12, 0) | spn.Scope(v12, 1)])
        self.assertListEqual(p2.get_scope(),
                             [spn.Scope(v12, 0) | spn.Scope(v12, 1)])
        self.assertListEqual(p3.get_scope(),
                             [spn.Scope(v34, 0) | spn.Scope(v34, 1)])
        self.assertListEqual(n1.get_scope(),
                             [spn.Scope(v12, 0) | spn.Scope(s1.latent_indicators.node, 0),
                              spn.Scope(v12, 1),
                              spn.Scope(v34, 0) | spn.Scope(v34, 1)])
        self.assertListEqual(n2.get_scope(),
                             [spn.Scope(v12, 0) | spn.Scope(v12, 1),
                              spn.Scope(v12, 0) | spn.Scope(v12, 1)])
        self.assertListEqual(pp1.get_scope(),
                             [spn.Scope(v12, 0) | spn.Scope(v12, 1) |
                              spn.Scope(s1.latent_indicators.node, 0),
                              spn.Scope(v12, 0) | spn.Scope(v12, 1) |
                              spn.Scope(s1.latent_indicators.node, 0),
                              spn.Scope(v12, 0) | spn.Scope(v12, 1),
                              spn.Scope(v12, 0) | spn.Scope(v12, 1),
                              spn.Scope(v12, 0) | spn.Scope(v12, 1) |
                              spn.Scope(v34, 0) | spn.Scope(v34, 1),
                              spn.Scope(v12, 0) | spn.Scope(v12, 1) |
                              spn.Scope(v34, 0) | spn.Scope(v34, 1)])
        self.assertListEqual(pp2.get_scope(),
                             [spn.Scope(v12, 0) | spn.Scope(v12, 1) |
                              spn.Scope(s1.latent_indicators.node, 0),
                              spn.Scope(v12, 0) | spn.Scope(v12, 1)])
        self.assertListEqual(pp3.get_scope(),
                             [spn.Scope(v12, 0) | spn.Scope(v12, 1) |
                              spn.Scope(v34, 0) | spn.Scope(v34, 1),
                              spn.Scope(v12, 0) | spn.Scope(v12, 1) |
                              spn.Scope(v34, 0) | spn.Scope(v34, 1)])
        self.assertListEqual(pp4.get_scope(),
                             [spn.Scope(v12, 0) | spn.Scope(v12, 1) |
                              spn.Scope(v34, 0) | spn.Scope(v34, 1)])
        self.assertListEqual(pp5.get_scope(),
                             [spn.Scope(v12, 0) | spn.Scope(v12, 1)])
        self.assertListEqual(pp6.get_scope(),
                             [spn.Scope(v34, 0) | spn.Scope(v34, 1)])
        self.assertListEqual(n3.get_scope(),
                             [spn.Scope(v12, 0) | spn.Scope(v12, 1) |
                              spn.Scope(s1.latent_indicators.node, 0),
                              spn.Scope(v12, 0) | spn.Scope(v12, 1),
                              spn.Scope(v12, 0) | spn.Scope(v12, 1),
                              spn.Scope(v12, 0) | spn.Scope(v12, 1) |
                              spn.Scope(s1.latent_indicators.node, 0),
                              spn.Scope(v12, 0) | spn.Scope(v12, 1),
                              spn.Scope(v12, 0) | spn.Scope(v12, 1) |
                              spn.Scope(v34, 0) | spn.Scope(v34, 1)])
        self.assertListEqual(s3.get_scope(),
                             [spn.Scope(v12, 0) | spn.Scope(v12, 1) |
                              spn.Scope(v34, 0) | spn.Scope(v34, 1) |
                              spn.Scope(s1.latent_indicators.node, 0) |
                              spn.Scope(s3.latent_indicators.node, 0)])
        self.assertListEqual(n4.get_scope(),
                             [spn.Scope(v12, 0) | spn.Scope(v12, 1) |
                              spn.Scope(v34, 0) | spn.Scope(v34, 1),
                              spn.Scope(v12, 0) | spn.Scope(v12, 1) |
                              spn.Scope(v34, 0) | spn.Scope(v34, 1),
                              spn.Scope(v12, 0) | spn.Scope(v12, 1),
                              spn.Scope(v34, 0) | spn.Scope(v34, 1)])
        self.assertListEqual(pp7.get_scope(),
                             [spn.Scope(v12, 0) | spn.Scope(v12, 1) |
                              spn.Scope(v34, 0) | spn.Scope(v34, 1) |
                              spn.Scope(s1.latent_indicators.node, 0) |
                              spn.Scope(s3.latent_indicators.node, 0)] * 24)
        self.assertListEqual(pp8.get_scope(),
                             [spn.Scope(v12, 0) | spn.Scope(v12, 1) |
                              spn.Scope(s1.latent_indicators.node, 0) | spn.Scope(v34, 0) |
                              spn.Scope(v34, 1)])
        self.assertListEqual(pp9.get_scope(),
                             [spn.Scope(v12, 0) | spn.Scope(v12, 1) |
                              spn.Scope(v34, 0) | spn.Scope(v34, 1)])

    def test_compute_valid(self):
        """Calculating validity of PermuteProducts"""
        v12 = spn.IndicatorLeaf(num_vars=2, num_vals=3)
        v345 = spn.IndicatorLeaf(num_vars=3, num_vals=3)
        v678 = spn.RawLeaf(num_vars=3)
        v910 = spn.RawLeaf(num_vars=2)
        p1 = spn.PermuteProducts((v12, [0, 1]), (v12, [4, 5]))
        p2 = spn.PermuteProducts((v12, [3, 5]), (v345, [0, 1, 2]))
        p3 = spn.PermuteProducts((v345, [0, 1, 2]), (v345, [3, 4, 5]), (v345, [6, 7, 8]))
        p4 = spn.PermuteProducts((v345, [6, 8]), (v678, [0, 1]))
        p5 = spn.PermuteProducts((v678, [1]), v910)
        p6 = spn.PermuteProducts(v678, v910)
        p7 = spn.PermuteProducts((v678, [0, 1, 2]))
        p8 = spn.PermuteProducts((v910, [0]), (v910, [1]))
        self.assertTrue(p1.is_valid())
        self.assertTrue(p2.is_valid())
        self.assertTrue(p3.is_valid())
        self.assertTrue(p4.is_valid())
        self.assertTrue(p5.is_valid())
        self.assertTrue(p6.is_valid())
        self.assertTrue(p7.is_valid())
        self.assertTrue(p8.is_valid())
        p9 = spn.PermuteProducts((v12, [0, 1]), (v12, [1, 2]))
        p10 = spn.PermuteProducts((v12, [3, 4, 5]), (v345, [0]), (v345, [0, 1, 2]))
        p11 = spn.PermuteProducts((v345, [3, 5]), (v678, [0]), (v678, [0]))
        p12 = spn.PermuteProducts((v910, [1]), (v910, [1]))
        p13 = spn.PermuteProducts(v910, v910)
        p14 = spn.PermuteProducts((v12, [0]), (v12, [1]))
        self.assertFalse(p9.is_valid())
        self.assertFalse(p10.is_valid())
        self.assertFalse(p11.is_valid())
        self.assertFalse(p12.is_valid())
        self.assertFalse(p13.is_valid())
        self.assertEqual(p14.num_prods, 1)
        self.assertFalse(p14.is_valid())

    def test_compute_mpe_path(self):
        """Calculating MPE path of PermuteProducts"""
        def test(counts, inputs, feed, output):
            with self.subTest(counts=counts, inputs=inputs, feed=feed):
                p = spn.PermuteProducts(*inputs)
                op = p._compute_log_mpe_path(tf.identity(counts),
                                         *[i[0].get_value() for i in inputs])
                with self.test_session() as sess:
                    out = sess.run(op, feed_dict=feed)

                for o, t in zip(out, output):
                    np.testing.assert_array_almost_equal(
                        o,
                        np.array(t, dtype=spn.conf.dtype.as_numpy_dtype()))

        # Create inputs
        v1 = spn.RawLeaf(num_vars=6)
        v2 = spn.RawLeaf(num_vars=8)
        v3 = spn.RawLeaf(num_vars=5)

        # Multiple Product nodes - Common input Sizes
        # -------------------------------------------

        # Case 1: No. of inputs = Input sizes
        # No. of inputs = 2
        # Input sizes = [2, 2] --> {O O | O O}
        counts = tf.placeholder(tf.float32, shape=(None, 4))
        test(counts,
             [(v1, [4, 5]), (v2, [1, 2])],
             {counts: [[1, 2, 3, 4],
                       [11, 12, 13, 14],
                       [21, 22, 23, 24]]},
             [[[0.0, 0.0, 0.0, 0.0, 3.0, 7.0],
               [0.0, 0.0, 0.0, 0.0, 23.0, 27.0],
               [0.0, 0.0, 0.0, 0.0, 43.0, 47.0]],
              [[0.0, 4.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 24.0, 26.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 44.0, 46.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])

        # Case 2: No. of inputs < Input sizes
        # No. of inputs = 2
        # Input sizes = [3, 3] --> {O O O | O O O}
        counts = tf.placeholder(tf.float32, shape=(None, 9))
        test(counts,
             [(v1, [0, 2, 4]), (v2, [1, 4, 7])],
             {counts: [[1, 2, 3, 4, 5, 6, 7, 8, 9],
                       [11, 12, 13, 14, 15, 16, 17, 18, 19],
                       [21, 22, 23, 24, 25, 26, 27, 28, 29]]},
             [[[6.0, 0.0, 15.0, 0.0, 24.0, 0.0],
               [36.0, 0.0, 45.0, 0.0, 54.0, 0.0],
               [66.0, 0.0, 75.0, 0.0, 84.0, 0.0]],
              [[0.0, 12.0, 0.0, 0.0, 15.0, 0.0, 0.0, 18.0],
               [0.0, 42.0, 0.0, 0.0, 45.0, 0.0, 0.0, 48.0],
               [0.0, 72.0, 0.0, 0.0, 75.0, 0.0, 0.0, 78.0]]])

        # Case 3: No. of inputs > Input sizes
        # No. of inputs = 3
        # Input sizes = [2, 2, 2] --> {O O | O O | O O}
        counts = tf.placeholder(tf.float32, shape=(None, 8))
        test(counts,
             [(v1, [0, 2]), (v2, [3, 5]), (v3, [1, 4])],
             {counts: [list(range(1, 9)),
                       list(range(11, 19)),
                       list(range(21, 29))]},
             [[[10.0, 0.0, 26.0, 0.0, 0.0, 0.0],
               [50.0, 0.0, 66.0, 0.0, 0.0, 0.0],
               [90.0, 0.0, 106.0, 0.0, 0.0, 0.0]],
              [[0.0, 0.0, 0.0, 14.0, 0.0, 22.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 54.0, 0.0, 62.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 94.0, 0.0, 102.0, 0.0, 0.0]],
              [[0.0, 16.0, 0.0, 0.0, 20.0],
               [0.0, 56.0, 0.0, 0.0, 60.0],
               [0.0, 96.0, 0.0, 0.0, 100.0]]])

        # Case 4: No. of inputs = Input sizes
        # No. of inputs = 3
        # Input sizes = [3, 3, 3] --> {O O O | O O O | O O O}
        counts = tf.placeholder(tf.float32, shape=(None, 27))
        test(counts,
             [(v1, [1, 3, 5]), (v2, [2, 4, 6]), (v3, [0, 1, 4])],
             {counts: [list(range(1, 28)),
                       list(range(101, 128)),
                       list(range(201, 228))]},
             [[[0.0, 45.0, 0.0, 126.0, 0.0, 207.0],
               [0.0, 945.0, 0.0, 1026.0, 0.0, 1107.0],
               [0.0, 1845.0, 0.0, 1926.0, 0.0, 2007.0]],
              [[0.0, 0.0, 99.0, 0.0, 126.0, 0.0, 153.0, 0.0],
               [0.0, 0.0, 999.0, 0.0, 1026.0, 0.0, 1053.0, 0.0],
               [0.0, 0.0, 1899.0, 0.0, 1926.0, 0.0, 1953.0, 0.0]],
              [[117.0, 126.0, 0.0, 0.0, 135.0],
               [1017.0, 1026.0, 0.0, 0.0, 1035.0],
               [1917.0, 1926.0, 0.0, 0.0, 1935.0]]])

        # Multiple Product nodes - Varying input Sizes
        # --------------------------------------------

        # Case 5: Ascending input sizes
        # No. of inputs = 3
        # Input sizes = [1, 2, 3] --> {O | O O | O O O}
        counts = tf.placeholder(tf.float32, shape=(None, 6))
        test(counts,
             [(v1, [3]), (v2, [4, 6]), (v3, [1, 2, 3])],
             {counts: [[1, 2, 3, 4, 5, 6],
                       [11, 12, 13, 14, 15, 16],
                       [21, 22, 23, 24, 25, 26]]},
             [[[0.0, 0.0, 0.0, 21.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 81.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 141.0, 0.0, 0.0]],
              [[0.0, 0.0, 0.0, 0.0, 6.0, 0.0, 15.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 36.0, 0.0, 45.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 66.0, 0.0, 75.0, 0.0]],
              [[0.0, 5.0, 7.0, 9.0, 0.0],
               [0.0, 25.0, 27.0, 29.0, 0.0],
               [0.0, 45.0, 47.0, 49.0, 0.0]]])

        # Case 6: Descending input sizes
        # No. of inputs = 3
        # Input sizes = [3, 2, 1] --> {O O O | O O | O}
        counts = tf.placeholder(tf.float32, shape=(None, 6))
        test(counts,
             [(v1, [2, 3, 4]), (v2, [0, 7]), (v3, [2])],
             {counts: [[1, 2, 3, 4, 5, 6],
                       [11, 12, 13, 14, 15, 16],
                       [21, 22, 23, 24, 25, 26]]},
             [[[0.0, 0.0, 3.0, 7.0, 11.0, 0.0],
               [0.0, 0.0, 23.0, 27.0, 31.0, 0.0],
               [0.0, 0.0, 43.0, 47.0, 51.0, 0.0]],
              [[9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 12.0],
               [39.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 42.0],
               [69.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 72.0]],
              [[0.0, 0.0, 21.0, 0.0, 0.0],
               [0.0, 0.0, 81.0, 0.0, 0.0],
               [0.0, 0.0, 141.0, 0.0, 0.0]]])

        # Case 7: Mixed input sizes - 1
        # No. of inputs = 3
        # Input sizes = [3, 2, 3] --> {O O O | O O | O O O}
        counts = tf.placeholder(tf.float32, shape=(None, 18))
        test(counts,
             [(v1, [0, 2, 5]), (v2, [3, 6]), (v3, [1, 2, 4])],
             {counts: [list(range(1, 19)),
                       list(range(21, 39)),
                       list(range(41, 59))]},
             [[[21.0, 0.0, 57.0, 0.0, 0.0, 93.0],
               [141.0, 0.0, 177.0, 0.0, 0.0, 213.0],
               [261.0, 0.0, 297.0, 0.0, 0.0, 333.0]],
              [[0.0, 0.0, 0.0, 72.0, 0.0, 0.0, 99.0, 0.0],
               [0.0, 0.0, 0.0, 252.0, 0.0, 0.0, 279.0, 0.0],
               [0.0, 0.0, 0.0, 432.0, 0.0, 0.0, 459.0, 0.0]],
              [[0.0, 51.0, 57.0, 0.0, 63.0],
               [0.0, 171.0, 177.0, 0.0, 183.0],
               [0.0, 291.0, 297.0, 0.0, 303.0]]])

        # Case 8: Mixed input sizes - 2
        # No. of inputs = 3
        # Input sizes = [1, 2, §] --> {O | O O O | O}
        counts = tf.placeholder(tf.float32, shape=(None, 3))
        test(counts,
             [(v1, [0]), (v2, [1, 2, 3]), (v3, [4])],
             {counts: [[1, 2, 3],
                       [11, 12, 13],
                       [21, 22, 23]]},
             [[[6.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [36.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [66.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
              [[0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 11.0, 12.0, 13.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 21.0, 22.0, 23.0, 0.0, 0.0, 0.0, 0.0]],
              [[0.0, 0.0, 0.0, 0.0, 6.0],
               [0.0, 0.0, 0.0, 0.0, 36.0],
               [0.0, 0.0, 0.0, 0.0, 66.0]]])

        # Single Product node
        # -------------------

        # Case 9: Multiple inputs, each with size 1
        # No. of inputs = 3
        # Input sizes = [1, 1, 1] --> {O | O | O}
        counts = tf.placeholder(tf.float32, shape=(None, 1))
        test(counts,
             [(v1, [3]), (v2, [2]), (v3, [1])],
             {counts: [[123],
                       [123],
                       [123]]},
             [[[0.0, 0.0, 0.0, 123.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 123.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 123.0, 0.0, 0.0]],
              [[0.0, 0.0, 123.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 123.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 123.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
              [[0.0, 123.0, 0.0, 0.0, 0.0],
               [0.0, 123.0, 0.0, 0.0, 0.0],
               [0.0, 123.0, 0.0, 0.0, 0.0]]])

        # Case 10: Single input with size > 1
        # No. of inputs = 1
        # Input sizes = [3] --> {O O O}
        counts = tf.placeholder(tf.float32, shape=(None, 1))
        test(counts,
             [(v3, [1, 2, 3])],
             {counts: [[3],
                       [3],
                       [3]]},
             [[[0.0, 3.0, 3.0, 3.0, 0.0],
               [0.0, 3.0, 3.0, 3.0, 0.0],
               [0.0, 3.0, 3.0, 3.0, 0.0]]])

        # Case 11:  Single input with size = 1
        # No. of inputs = 1
        # Input sizes = [1] --> {O}
        counts = tf.placeholder(tf.float32, shape=(None, 1))
        test(counts,
             [(v3, [4])],
             {counts: [[1],
                       [1],
                       [1]]},
             [[[0.0, 0.0, 0.0, 0.0, 1.0],
               [0.0, 0.0, 0.0, 0.0, 1.0],
               [0.0, 0.0, 0.0, 0.0, 1.0]]])


if __name__ == '__main__':
    unittest.main()
