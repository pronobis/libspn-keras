#!/usr/bin/env python3

from context import libspn as spn
from test import TestCase
import tensorflow as tf
import numpy as np


class TestGraph(TestCase):

    def test_input_conversion(self):
        """Conversion and verification of input specs in Input"""
        v1 = spn.RawLeaf(num_vars=5)
        # None
        inpt = spn.Input()
        self.assertIs(inpt.node, None)
        self.assertIs(inpt.indices, None)
        self.assertFalse(inpt)
        inpt = spn.Input(None)
        self.assertIs(inpt.node, None)
        self.assertIs(inpt.indices, None)
        self.assertFalse(inpt)
        inpt = spn.Input(None, [1, 2, 3])
        self.assertIs(inpt.node, None)
        self.assertIs(inpt.indices, None)
        self.assertFalse(inpt)
        inpt = spn.Input.as_input(None)
        self.assertIs(inpt.node, None)
        self.assertIs(inpt.indices, None)
        self.assertFalse(inpt)
        inpt = spn.Input.as_input((None, [1, 2, 3]))
        self.assertIs(inpt.node, None)
        self.assertIs(inpt.indices, None)
        self.assertFalse(inpt)
        # Node
        inpt = spn.Input(v1)
        self.assertIs(inpt.node, v1)
        self.assertIs(inpt.indices, None)
        self.assertTrue(inpt)
        inpt = spn.Input.as_input(v1)
        self.assertIs(inpt.node, v1)
        self.assertIs(inpt.indices, None)
        self.assertTrue(inpt)
        # (Node, None)
        inpt = spn.Input(v1, None)
        self.assertIs(inpt.node, v1)
        self.assertIs(inpt.indices, None)
        self.assertTrue(inpt)
        inpt = spn.Input.as_input((v1, None))
        self.assertIs(inpt.node, v1)
        self.assertIs(inpt.indices, None)
        self.assertTrue(inpt)
        # (Node, index)
        inpt = spn.Input(v1, 10)
        self.assertIs(inpt.node, v1)
        self.assertListEqual(inpt.indices, [10])
        self.assertTrue(inpt)
        inpt = spn.Input.as_input((v1, 10))
        self.assertIs(inpt.node, v1)
        self.assertListEqual(inpt.indices, [10])
        self.assertTrue(inpt)
        # (Node, indices)
        inpt = spn.Input(v1, [10])
        self.assertIs(inpt.node, v1)
        self.assertListEqual(inpt.indices, [10])
        self.assertTrue(inpt)
        inpt = spn.Input.as_input((v1, [10]))
        self.assertIs(inpt.node, v1)
        self.assertListEqual(inpt.indices, [10])
        self.assertTrue(inpt)
        inpt = spn.Input(v1, [10, 1, 20])
        self.assertIs(inpt.node, v1)
        self.assertListEqual(inpt.indices, [10, 1, 20])
        self.assertTrue(inpt)
        inpt = spn.Input.as_input((v1, [10, 1, 20]))
        self.assertIs(inpt.node, v1)
        self.assertListEqual(inpt.indices, [10, 1, 20])
        self.assertTrue(inpt)

        # Checking type of input
        with self.assertRaises(TypeError):
            inpt = spn.Input(set())
        with self.assertRaises(TypeError):
            inpt = spn.Input.as_input(set())
        with self.assertRaises(TypeError):
            inpt = spn.Input.as_input((set()))
        with self.assertRaises(TypeError):
            inpt = spn.Input.as_input(tuple())
        with self.assertRaises(TypeError):
            inpt = spn.Input(v1, set())
        with self.assertRaises(TypeError):
            inpt = spn.Input(v1, set())
        with self.assertRaises(TypeError):
            inpt = spn.Input.as_input((v1, set()))
        with self.assertRaises(TypeError):
            inpt = spn.Input.as_input((v1,))
        # Detecting empty list
        with self.assertRaises(ValueError):
            inpt = spn.Input(v1, [])
        with self.assertRaises(ValueError):
            inpt = spn.Input.as_input((v1, []))
        # Detecting incorrect indices
        with self.assertRaises(ValueError):
            inpt = spn.Input(v1, [1, set(), 2])
        with self.assertRaises(ValueError):
            inpt = spn.Input.as_input((v1, [1, set(), 2]))
        with self.assertRaises(ValueError):
            inpt = spn.Input(v1, [1, -1, 2])
        with self.assertRaises(ValueError):
            inpt = spn.Input.as_input((v1, [1, -1, 2]))

    def test_input_flags(self):
        """Detection of different types of inputs"""
        inpt = spn.Input()
        self.assertFalse(inpt)
        self.assertFalse(inpt.is_op)
        self.assertFalse(inpt.is_var)
        self.assertFalse(inpt.is_param)

        n = spn.Sum()
        inpt = spn.Input(n)
        self.assertTrue(inpt)
        self.assertTrue(inpt.is_op)
        self.assertFalse(inpt.is_var)
        self.assertFalse(inpt.is_param)

        n = spn.RawLeaf()
        inpt = spn.Input(n)
        self.assertTrue(inpt)
        self.assertFalse(inpt.is_op)
        self.assertTrue(inpt.is_var)
        self.assertFalse(inpt.is_param)

        n = spn.Weights()
        inpt = spn.Input(n)
        self.assertTrue(inpt)
        self.assertFalse(inpt.is_op)
        self.assertFalse(inpt.is_var)
        self.assertTrue(inpt.is_param)

    def test_opnode_varnode_abstract(self):
        """Can OpNode, VarNode and ParamNode be instantiated?"""
        with self.assertRaises(TypeError):
            spn.OpNode()
        with self.assertRaises(TypeError):
            spn.VarNode()
        with self.assertRaises(TypeError):
            spn.ParamNode()

    def test_get_nodes(self):
        """Obtaining the list of nodes in the SPN graph"""
        # Generate graph
        v1 = spn.RawLeaf(num_vars=1)
        v2 = spn.RawLeaf(num_vars=1)
        v3 = spn.RawLeaf(num_vars=1)
        s1 = spn.Sum(v1, v1, v2)  # v1 included twice
        s2 = spn.Sum(v1, v3)
        s3 = spn.Sum(v2, v3, v3)  # v3 included twice
        s4 = spn.Sum(s1, v1)
        s5 = spn.Sum(s2, v3, s3)
        s6 = spn.Sum(s4, s2, s5, s4, s5)  # s4 and s5 included twice
        spn.generate_weights(s6)

        # Test
        nodes = v1.get_nodes(skip_params=True)
        self.assertListEqual(nodes, [v1])
        nodes = v1.get_nodes(skip_params=False)
        self.assertListEqual(nodes, [v1])

        nodes = v2.get_nodes(skip_params=True)
        self.assertListEqual(nodes, [v2])
        nodes = v2.get_nodes(skip_params=False)
        self.assertListEqual(nodes, [v2])

        nodes = v3.get_nodes(skip_params=True)
        self.assertListEqual(nodes, [v3])
        nodes = v3.get_nodes(skip_params=False)
        self.assertListEqual(nodes, [v3])

        nodes = s1.get_nodes(skip_params=True)
        self.assertListEqual(nodes, [s1, v1, v2])
        nodes = s1.get_nodes(skip_params=False)
        self.assertListEqual(nodes, [s1, s1.weights.node, v1, v2])

        nodes = s2.get_nodes(skip_params=True)
        self.assertListEqual(nodes, [s2, v1, v3])
        nodes = s2.get_nodes(skip_params=False)
        self.assertListEqual(nodes, [s2, s2.weights.node, v1, v3])

        nodes = s3.get_nodes(skip_params=True)
        self.assertListEqual(nodes, [s3, v2, v3])
        nodes = s3.get_nodes(skip_params=False)
        self.assertListEqual(nodes, [s3, s3.weights.node, v2, v3])

        nodes = s4.get_nodes(skip_params=True)
        self.assertListEqual(nodes, [s4, s1, v1, v2])
        nodes = s4.get_nodes(skip_params=False)
        self.assertListEqual(nodes, [s4, s4.weights.node, s1, v1,
                                     s1.weights.node, v2])

        nodes = s5.get_nodes(skip_params=True)
        self.assertListEqual(nodes, [s5, s2, v3, s3, v1, v2])
        nodes = s5.get_nodes(skip_params=False)
        self.assertListEqual(nodes, [s5, s5.weights.node, s2, v3, s3,
                                     s2.weights.node, v1, s3.weights.node, v2])

        nodes = s6.get_nodes(skip_params=True)
        self.assertListEqual(nodes, [s6, s4, s2, s5, s1, v1, v3, s3, v2])
        nodes = s6.get_nodes(skip_params=False)
        self.assertListEqual(nodes, [s6, s6.weights.node, s4, s2, s5,
                                     s4.weights.node, s1, v1, s2.weights.node,
                                     v3, s5.weights.node, s3, s1.weights.node,
                                     v2, s3.weights.node])

    def test_get_num_nodes(self):
        """Computing the number of nodes in the SPN graph"""
        # Generate graph
        v1 = spn.RawLeaf(num_vars=1)
        v2 = spn.RawLeaf(num_vars=1)
        v3 = spn.RawLeaf(num_vars=1)
        s1 = spn.Sum(v1, v1, v2)  # v1 included twice
        s2 = spn.Sum(v1, v3)
        s3 = spn.Sum(v2, v3, v3)  # v3 included twice
        s4 = spn.Sum(s1, v1)
        s5 = spn.Sum(s2, v3, s3)
        s6 = spn.Sum(s4, s2, s5, s4, s5)  # s4 and s5 included twice
        spn.generate_weights(s6)

        # Test
        num = v1.get_num_nodes(skip_params=True)
        self.assertEqual(num, 1)
        num = v1.get_num_nodes(skip_params=False)
        self.assertEqual(num, 1)

        num = v2.get_num_nodes(skip_params=True)
        self.assertEqual(num, 1)
        num = v2.get_num_nodes(skip_params=False)
        self.assertEqual(num, 1)

        num = v3.get_num_nodes(skip_params=True)
        self.assertEqual(num, 1)
        num = v3.get_num_nodes(skip_params=False)
        self.assertEqual(num, 1)

        num = s1.get_num_nodes(skip_params=True)
        self.assertEqual(num, 3)
        num = s1.get_num_nodes(skip_params=False)
        self.assertEqual(num, 4)

        num = s2.get_num_nodes(skip_params=True)
        self.assertEqual(num, 3)
        num = s2.get_num_nodes(skip_params=False)
        self.assertEqual(num, 4)

        num = s3.get_num_nodes(skip_params=True)
        self.assertEqual(num, 3)
        num = s3.get_num_nodes(skip_params=False)
        self.assertEqual(num, 4)

        num = s4.get_num_nodes(skip_params=True)
        self.assertEqual(num, 4)
        num = s4.get_num_nodes(skip_params=False)
        self.assertEqual(num, 6)

        num = s5.get_num_nodes(skip_params=True)
        self.assertEqual(num, 6)
        num = s5.get_num_nodes(skip_params=False)
        self.assertEqual(num, 9)

        num = s6.get_num_nodes(skip_params=True)
        self.assertEqual(num, 9)
        num = s6.get_num_nodes(skip_params=False)
        self.assertEqual(num, 15)

    def test_get_out_size(self):
        """Computing the sizes of the outputs of nodes in SPN graph"""
        # Generate graph
        v1 = spn.RawLeaf(num_vars=5)
        v2 = spn.RawLeaf(num_vars=5)
        v3 = spn.RawLeaf(num_vars=5)
        s1 = spn.Sum((v1, [1, 3]), (v1, [1, 4]), v2)  # v1 included twice
        s2 = spn.Sum(v1, (v3, [0, 1, 2, 3, 4]))
        s3 = spn.Sum(v2, v3, v3)  # v3 included twice
        n4 = spn.Concat(s1, v1)
        n5 = spn.Concat((v3, [0, 4]), s3)
        n6 = spn.Concat(n4, s2, n5, (n4, [0]), (n5, [1]))  # n4 and n5 included twice

        # Test
        num = v1.get_out_size()
        self.assertEqual(num, 5)
        num = v2.get_out_size()
        self.assertEqual(num, 5)
        num = v3.get_out_size()
        self.assertEqual(num, 5)
        num = s1.get_out_size()
        self.assertEqual(num, 1)
        num = s2.get_out_size()
        self.assertEqual(num, 1)
        num = s3.get_out_size()
        self.assertEqual(num, 1)
        num = n4.get_out_size()
        self.assertEqual(num, 6)
        num = n5.get_out_size()
        self.assertEqual(num, 3)
        num = n6.get_out_size()
        self.assertEqual(num, 12)

    def test_scope(self):
        """Scope creation and operations"""
        # Creation
        s1 = spn.Scope("a", 1)
        self.assertListEqual([i for i in s1],
                             [("a", 1)])
        s2 = spn.Scope("b", 1)
        s3 = spn.Scope("b", 2)
        # Merging
        s = spn.Scope.merge_scopes([s1, s2, s3])
        self.assertListEqual(sorted([i for i in s]),
                             [("a", 1), ("b", 1), ("b", 2)])
        # Set operations
        s = s1 | s2 | s3
        self.assertListEqual(sorted([i for i in s]),
                             [("a", 1), ("b", 1), ("b", 2)])
        s = s1.union(s2).union(s3)
        self.assertListEqual(sorted([i for i in s]),
                             [("a", 1), ("b", 1), ("b", 2)])
        s = (s1 | s2 | s3) & spn.Scope("b", 1)
        self.assertListEqual(sorted([i for i in s]),
                             [("b", 1)])
        s = (s1 | s2 | s3).intersection(spn.Scope("b", 1))
        self.assertListEqual(sorted([i for i in s]),
                             [("b", 1)])
        s = (s1 | s2 | s3) - spn.Scope("b", 1)
        self.assertListEqual(sorted([i for i in s]),
                             [("a", 1), ("b", 2)])
        s = (s1 | s2 | s3).difference(spn.Scope("b", 1))
        self.assertListEqual(sorted([i for i in s]),
                             [("a", 1), ("b", 2)])
        s = (s1 | s2) ^ (spn.Scope("b", 1) | spn.Scope("c", 1))
        self.assertListEqual(sorted([i for i in s]),
                             [("a", 1), ("c", 1)])
        s = (s1 | s2).symmetric_difference(
            (spn.Scope("b", 1) | spn.Scope("c", 1)))
        self.assertListEqual(sorted([i for i in s]),
                             [("a", 1), ("c", 1)])
        # Hash
        self.assertIsNotNone(hash(s))
        # Used in dicts
        d = {s1: 10, s: 12}
        self.assertEqual(d[spn.Scope("a", 1)], 10)
        self.assertEqual(d[spn.Scope("a", 1) | spn.Scope("c", 1)], 12)
        # Subsets
        self.assertTrue(s.issuperset(spn.Scope("c", 1)))
        self.assertTrue(spn.Scope("c", 1).issubset(s))
        self.assertFalse(s.issuperset(spn.Scope("d", 1)))
        self.assertFalse(spn.Scope("d", 1).issubset(s))
        self.assertTrue(s.isdisjoint(spn.Scope("d", 1)))
        self.assertFalse(s.isdisjoint(spn.Scope("a", 1)))
        # Test for empty scope
        self.assertFalse(spn.Scope("a", 1) & spn.Scope("b", 1))
        self.assertTrue(spn.Scope("a", 1) | spn.Scope("b", 1))
        self.assertEqual(len(spn.Scope("a", 1) & spn.Scope("b", 1)), 0)
        self.assertEqual(len(spn.Scope("a", 1) | spn.Scope("b", 1)), 2)

    def test_gather_input_scopes(self):
        v12 = spn.IndicatorLeaf(num_vars=2, num_vals=4, name="V12")
        v34 = spn.RawLeaf(num_vars=2, name="V34")
        s1 = spn.Sum(v12, v12, v34, (v12, [7, 3, 1, 0]), (v34, 0), name="S1")
        scopes_v12 = v12._compute_scope()
        scopes_v34 = v34._compute_scope()
        # Note: weights/latent_indicators are disconnected, so None should be output these
        scopes = s1._gather_input_scopes(None, None, None, scopes_v12, scopes_v34,
                                         scopes_v12, scopes_v34)
        self.assertTupleEqual(scopes,
                              (None, None, None, scopes_v12, scopes_v34,
                               [scopes_v12[7], scopes_v12[3],
                                scopes_v12[1], scopes_v12[0]],
                               [scopes_v34[0]]))

    def test_get_scope(self):
        """Computing the scope of nodes of the SPN graph"""
        # Create graph
        v12 = spn.IndicatorLeaf(num_vars=2, num_vals=4, name="V12")
        v34 = spn.RawLeaf(num_vars=2, name="V34")
        s1 = spn.Sum((v12, [0, 1, 2, 3]), name="S1")
        s2 = spn.Sum((v12, [4, 5, 6, 7]), name="S2")
        p1 = spn.Product((v12, [0, 7]), name="P1")
        p2 = spn.Product((v12, [3, 4]), name="P2")
        p3 = spn.Product(v34, name="P3")
        n1 = spn.Concat(s1, s2, p3, name="N1")
        n2 = spn.Concat(p1, p2, name="N2")
        p4 = spn.Product((n1, [0]), (n1, [1]), name="P4")
        p5 = spn.Product((n2, [0]), (n1, [2]), name="P5")
        s3 = spn.Sum(p4, n2, name="S3")
        p6 = spn.Product(s3, (n1, [2]), name="P6")
        s4 = spn.Sum(p5, p6, name="S4")
        # Test
        self.assertListEqual(v12.get_scope(),
                             [spn.Scope(v12, 0), spn.Scope(v12, 0),
                              spn.Scope(v12, 0), spn.Scope(v12, 0),
                              spn.Scope(v12, 1), spn.Scope(v12, 1),
                              spn.Scope(v12, 1), spn.Scope(v12, 1)])
        self.assertListEqual(v34.get_scope(),
                             [spn.Scope(v34, 0), spn.Scope(v34, 1)])
        self.assertListEqual(s1.get_scope(),
                             [spn.Scope(v12, 0)])
        self.assertListEqual(s2.get_scope(),
                             [spn.Scope(v12, 1)])
        self.assertListEqual(p1.get_scope(),
                             [spn.Scope(v12, 0) | spn.Scope(v12, 1)])
        self.assertListEqual(p2.get_scope(),
                             [spn.Scope(v12, 0) | spn.Scope(v12, 1)])
        self.assertListEqual(p3.get_scope(),
                             [spn.Scope(v34, 0) | spn.Scope(v34, 1)])
        self.assertListEqual(n1.get_scope(),
                             [spn.Scope(v12, 0),
                              spn.Scope(v12, 1),
                              spn.Scope(v34, 0) | spn.Scope(v34, 1)])
        self.assertListEqual(n2.get_scope(),
                             [spn.Scope(v12, 0) | spn.Scope(v12, 1),
                              spn.Scope(v12, 0) | spn.Scope(v12, 1)])
        self.assertListEqual(p4.get_scope(),
                             [spn.Scope(v12, 0) | spn.Scope(v12, 1)])
        self.assertListEqual(p5.get_scope(),
                             [spn.Scope(v12, 0) | spn.Scope(v12, 1) |
                              spn.Scope(v34, 0) | spn.Scope(v34, 1)])
        self.assertListEqual(s3.get_scope(),
                             [spn.Scope(v12, 0) | spn.Scope(v12, 1)])
        self.assertListEqual(p6.get_scope(),
                             [spn.Scope(v12, 0) | spn.Scope(v12, 1) |
                              spn.Scope(v34, 0) | spn.Scope(v34, 1)])
        self.assertListEqual(s4.get_scope(),
                             [spn.Scope(v12, 0) | spn.Scope(v12, 1) |
                              spn.Scope(v34, 0) | spn.Scope(v34, 1)])

    def test_is_valid_true(self):
        """Checking validity of the SPN"""
        # Create graph
        v12 = spn.IndicatorLeaf(num_vars=2, num_vals=4, name="V12")
        v34 = spn.RawLeaf(num_vars=2, name="V34")
        s1 = spn.Sum((v12, [0, 1, 2, 3]), name="S1")
        s2 = spn.Sum((v12, [4, 5, 6, 7]), name="S2")
        p1 = spn.Product((v12, [0, 7]), name="P1")
        p2 = spn.Product((v12, [3, 4]), name="P2")
        p3 = spn.Product(v34, name="P3")
        n1 = spn.Concat(s1, s2, p3, name="N1")
        n2 = spn.Concat(p1, p2, name="N2")
        p4 = spn.Product((n1, [0]), (n1, [1]), name="P4")
        p5 = spn.Product((n2, [0]), (n1, [2]), name="P5")
        s3 = spn.Sum(p4, n2, name="S3")
        p6 = spn.Product(s3, (n1, [2]), name="P6")
        s4 = spn.Sum(p5, p6, name="S4")
        # Test
        self.assertTrue(v12.is_valid())
        self.assertTrue(v34.is_valid())
        self.assertTrue(s1.is_valid())
        self.assertTrue(s2.is_valid())
        self.assertTrue(s3.is_valid())
        self.assertTrue(s4.is_valid())
        self.assertTrue(p1.is_valid())
        self.assertTrue(p2.is_valid())
        self.assertTrue(p3.is_valid())
        self.assertTrue(p4.is_valid())
        self.assertTrue(p5.is_valid())
        self.assertTrue(p6.is_valid())
        self.assertTrue(n1.is_valid())
        self.assertTrue(n2.is_valid())

    def test_is_valid_false(self):
        """Checking validity of the SPN"""
        # Create graph
        v12 = spn.IndicatorLeaf(num_vars=2, num_vals=4, name="V12")
        v34 = spn.RawLeaf(num_vars=2, name="V34")
        s1 = spn.Sum((v12, [0, 1, 2, 3]), name="S1")
        s2 = spn.Sum((v12, [4, 5, 6, 7]), name="S2")
        p1 = spn.Product((v12, [0, 7]), name="P1")
        p2 = spn.Product((v12, [2, 3, 4]), name="P2")
        p3 = spn.Product(v34, name="P3")
        n1 = spn.Concat(s1, s2, p3, name="N1")
        n2 = spn.Concat(p1, p2, name="N2")
        p4 = spn.Product((n1, [0]), (n1, [1]), name="P4")
        p5 = spn.Product((n2, [0]), (n1, [2]), name="P5")
        s3 = spn.Sum(p4, n2, name="S3")
        p6 = spn.Product(s3, (n1, [2]), name="P6")
        s4 = spn.Sum(p5, p6, name="S4")
        # Test
        self.assertTrue(v12.is_valid())
        self.assertTrue(v34.is_valid())
        self.assertTrue(s1.is_valid())
        self.assertTrue(s2.is_valid())
        self.assertTrue(p1.is_valid())
        self.assertTrue(p3.is_valid())
        self.assertTrue(p4.is_valid())
        self.assertTrue(n1.is_valid())
        self.assertFalse(p2.is_valid())
        self.assertFalse(n2.is_valid())
        self.assertFalse(s3.is_valid())
        self.assertFalse(s4.is_valid())
        self.assertFalse(p5.is_valid())
        self.assertFalse(p6.is_valid())

    def test_gather_input_tensors(self):
        def test(inpt, feed, true_output):
            with self.subTest(inputs=inpt, feed=feed):
                n = spn.Concat(inpt)
                op, = n._gather_input_tensors(n.inputs[0].node.get_value())
                with self.test_session() as sess:
                    out = sess.run(op, feed_dict=feed)
                np.testing.assert_array_equal(out, np.array(true_output))

        v1 = spn.RawLeaf(num_vars=3)
        v2 = spn.RawLeaf(num_vars=1)

        # Disconnected input
        n = spn.Concat(None)
        op, = n._gather_input_tensors(3)
        self.assertIs(op, None)

        # None input tensor
        n = spn.Concat((v1, 1))
        op, = n._gather_input_tensors(None)
        self.assertIs(op, None)

        # Gathering for indices specified
        test((v1, [0, 2, 1]),
             {v1: [[1, 2, 3],
                   [4, 5, 6]]},
             [[1.0, 3.0, 2.0],
              [4.0, 6.0, 5.0]])
        test((v1, [0, 2]),
             {v1: [[1, 2, 3],
                   [4, 5, 6]]},
             [[1.0, 3.0],
              [4.0, 6.0]])
        test((v1, [1]),
             {v1: [[1, 2, 3],
                   [4, 5, 6]]},
             [[2.0],
              [5.0]])
        test((v1, [0, 2, 1]),
             {v1: [[1, 2, 3]]},
             [[1.0, 3.0, 2.0]])
        test((v1, [0, 2]),
             {v1: [[1, 2, 3]]},
             [[1.0, 3.0]])
        test((v1, [1]),
             {v1: [[1, 2, 3]]},
             [[2.0]])

        # Test that if None indices, it passes the tensor directly
        n = spn.Concat(v1)
        t = tf.constant([1, 2, 3])
        op, = n._gather_input_tensors(t)
        self.assertIs(op, t)

        # Gathering for None indices
        test(v1,
             {v1: [[1, 2, 3],
                   [4, 5, 6]]},
             [[1.0, 2.0, 3.0],
              [4.0, 5.0, 6.0]])
        test((v1, None),
             {v1: [[1, 2, 3],
                   [4, 5, 6]]},
             [[1.0, 2.0, 3.0],
              [4.0, 5.0, 6.0]])
        test(v1,
             {v1: [[1, 2, 3]]},
             [[1.0, 2.0, 3.0]])
        test((v1, None),
             {v1: [[1, 2, 3]]},
             [[1.0, 2.0, 3.0]])

        # Gathering for single index specified
        test((v1, 1),
             {v1: [[1, 2, 3],
                   [4, 5, 6]]},
             [[2.0],
              [5.0]])
        test((v1, [1]),
             {v1: [[1, 2, 3],
                   [4, 5, 6]]},
             [[2.0],
              [5.0]])
        test((v1, 1),
             {v1: [[1, 2, 3]]},
             [[2.0]])
        test((v1, [1]),
             {v1: [[1, 2, 3]]},
             [[2.0]])

        # Gathering for one element input, index specified
        test((v2, 0),
             {v2: [[1],
                   [4]]},
             [[1.0],
              [4.0]])
        test((v2, [0]),
             {v2: [[1],
                   [4]]},
             [[1.0],
              [4.0]])
        test((v2, 0),
             {v2: [[1]]},
             [[1.0]])
        test((v2, [0]),
             {v2: [[1]]},
             [[1.0]])

        # Gathering for one element input, None indices
        test(v2,
             {v2: [[1],
                   [4]]},
             [[1.0],
              [4.0]])
        test((v2, None),
             {v2: [[1],
                   [4]]},
             [[1.0],
              [4.0]])
        test(v2,
             {v2: [[1]]},
             [[1.0]])
        test((v2, None),
             {v2: [[1]]},
             [[1.0]])


if __name__ == '__main__':
    tf.test.main()
