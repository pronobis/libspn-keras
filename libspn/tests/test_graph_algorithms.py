#!/usr/bin/env python3

from context import libspn as spn
from test import TestCase
import tensorflow as tf


class TestGraphAlgorithms(TestCase):

    def assertListAlmostEqual(self, list1, list2):
        self.assertEqual(len(list1), len(list2))
        for l1, l2 in zip(list1, list2):
            self.assertAlmostEqual(l1, l2)

    def test_compute_graph_up_noconst(self):
        """Computing value assuming no constant functions"""
        # Number of times val_fun was called
        # Use list to avoid creating local fun variable during assignment
        counter = [0]

        def val_fun(node, *inputs):
            counter[0] += 1
            if isinstance(node, spn.graph.node.VarNode):
                return 1
            elif isinstance(node, spn.graph.node.ParamNode):
                return 0.1
            else:
                weight_val, iv_val, *values = inputs
                return weight_val + sum(values) + 1

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

        # Calculate value
        val = spn.compute_graph_up(s6, val_fun)

        # Test
        self.assertAlmostEqual(val, 35.2)
        self.assertEqual(counter[0], 15)

    def test_compute_graph_up_const(self):
        """Computing value with constant function detection"""
        # Number of times val_fun was called
        # Use list to avoid creating local fun variable during assignment
        counter = [0]

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

        def val_fun(node, *inputs):
            counter[0] += 1
            # s3 is not needed for calculations since only parent is s5
            self.assertIsNot(node, s3)
            # Fixed value or compute using children
            if node == s5:
                return 16
            else:
                if isinstance(node, spn.graph.node.VarNode):
                    return 1
                else:
                    weight_val, iv_val, *values = inputs
                    return sum(values) + 1

        def const_fun(node):
            if node == s5:
                return True
            else:
                return False

        # Calculate value
        val = spn.compute_graph_up(s6, val_fun, const_fun)

        # Test
        self.assertEqual(val, 48)
        self.assertEqual(counter[0], 8)

    def test_compute_graph_down(self):
        counter = [0]
        parent_vals_saved = {}

        def fun(node, parent_vals):
            parent_vals_saved[node] = parent_vals
            val = sum(parent_vals) + 0.01
            counter[0] += 1
            if node.is_op:
                return [val + i for i, _ in enumerate(node.inputs)]
            else:
                return 101

        # Generate graph
        v1 = spn.RawLeaf(num_vars=1, name="v1")
        v2 = spn.RawLeaf(num_vars=1, name="v2")
        v3 = spn.RawLeaf(num_vars=1, name="v3")
        s1 = spn.Sum(v1, v1, v2, name="s1")  # v1 included twice
        s2 = spn.Sum(v1, v3, name="s2")
        s3 = spn.Sum(v2, v3, v3, name="s3")  # v3 included twice
        s4 = spn.Sum(s1, v1, name="s4")
        s5 = spn.Sum(s2, v3, s3, name="s5")
        s6 = spn.Sum(s4, s2, s5, s4, s5, name="s6")  # s4 and s5 included twice
        spn.generate_weights(s6)

        down_values = {}
        spn.compute_graph_up_down(s6, down_fun=fun, graph_input=5,
                                  down_values=down_values)

        self.assertEqual(counter[0], 15)
        # Using sorted since order is not guaranteed
        self.assertListAlmostEqual(sorted(parent_vals_saved[s6]), [5])
        self.assertListAlmostEqual(down_values[s6], [5.01, 6.01, 7.01, 8.01,
                                                     9.01, 10.01, 11.01])
        self.assertListAlmostEqual(sorted(parent_vals_saved[s5]), [9.01, 11.01])
        self.assertListAlmostEqual(down_values[s5], [20.03, 21.03, 22.03,
                                                     23.03, 24.03])
        self.assertListAlmostEqual(sorted(parent_vals_saved[s4]), [7.01, 10.01])
        self.assertListAlmostEqual(down_values[s4], [17.03, 18.03, 19.03, 20.03])
        self.assertListAlmostEqual(sorted(parent_vals_saved[s3]), [24.03])
        self.assertListAlmostEqual(down_values[s3], [24.04, 25.04, 26.04,
                                                     27.04, 28.04])
        self.assertListAlmostEqual(sorted(parent_vals_saved[s2]), [8.01, 22.03])
        self.assertListAlmostEqual(down_values[s2], [30.05, 31.05, 32.05, 33.05])
        self.assertListAlmostEqual(sorted(parent_vals_saved[s1]), [19.03])
        self.assertListAlmostEqual(down_values[s1], [19.04, 20.04, 21.04,
                                                     22.04, 23.04])

        self.assertListAlmostEqual(sorted(parent_vals_saved[v1]),
                                   [20.03, 21.04, 22.04, 32.05])
        self.assertEqual(down_values[v1], 101)
        self.assertListAlmostEqual(sorted(parent_vals_saved[v2]),
                                   [23.04, 26.04])
        self.assertEqual(down_values[v2], 101)
        self.assertListAlmostEqual(sorted(parent_vals_saved[v3]),
                                   [23.03, 27.04, 28.04, 33.05])
        self.assertEqual(down_values[v3], 101)

        # Test if the algorithm works on a VarNode and calls graph_input function
        down_values = {}
        parent_vals_saved = {}
        spn.compute_graph_up_down(v1, down_fun=fun, graph_input=lambda: 5,
                                  down_values=down_values)
        self.assertEqual(parent_vals_saved[v1][0], 5)
        self.assertEqual(down_values[v1], 101)

    def test_traverse_graph_nostop_params(self):
        """Traversing the whole graph including param nodes"""
        counter = [0]
        nodes = [None] * 15

        def fun(node):
            nodes[counter[0]] = node
            counter[0] += 1

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

        # Traverse
        spn.traverse_graph(s6, fun=fun, skip_params=False)

        # Test
        self.assertEqual(counter[0], 15)
        self.assertIs(nodes[0], s6)
        self.assertIs(nodes[1], s6.weights.node)
        self.assertIs(nodes[2], s4)
        self.assertIs(nodes[3], s2)
        self.assertIs(nodes[4], s5)
        self.assertIs(nodes[5], s4.weights.node)
        self.assertIs(nodes[6], s1)
        self.assertIs(nodes[7], v1)
        self.assertIs(nodes[8], s2.weights.node)
        self.assertIs(nodes[9], v3)
        self.assertIs(nodes[10], s5.weights.node)
        self.assertIs(nodes[11], s3)
        self.assertIs(nodes[12], s1.weights.node)
        self.assertIs(nodes[13], v2)
        self.assertIs(nodes[14], s3.weights.node)

    def test_traverse_graph_nostop_noparams(self):
        """Traversing the whole graph excluding param nodes"""
        counter = [0]
        nodes = [None] * 10

        def fun(node):
            nodes[counter[0]] = node
            counter[0] += 1

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

        # Traverse
        spn.traverse_graph(s6, fun=fun, skip_params=True)

        # Test
        self.assertEqual(counter[0], 9)
        self.assertIs(nodes[0], s6)
        self.assertIs(nodes[1], s4)
        self.assertIs(nodes[2], s2)
        self.assertIs(nodes[3], s5)
        self.assertIs(nodes[4], s1)
        self.assertIs(nodes[5], v1)
        self.assertIs(nodes[6], v3)
        self.assertIs(nodes[7], s3)
        self.assertIs(nodes[8], v2)

    def test_traverse_graph_stop(self):
        """Traversing the graph until fun returns True"""
        counter = [0]
        nodes = [None] * 9
        true_node_no = 4  # s5

        def fun(node):
            nodes[counter[0]] = node
            counter[0] += 1
            if counter[0] == true_node_no:
                return True

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

        # Traverse
        spn.traverse_graph(s6, fun=fun, skip_params=True)

        # Test
        self.assertEqual(counter[0], 4)
        self.assertIs(nodes[0], s6)
        self.assertIs(nodes[1], s4)
        self.assertIs(nodes[2], s2)
        self.assertIs(nodes[3], s5)
        self.assertIs(nodes[4], None)
        self.assertIs(nodes[5], None)
        self.assertIs(nodes[6], None)
        self.assertIs(nodes[7], None)
        self.assertIs(nodes[8], None)

    def test_traversing_on_dense(self):
        """Compare traversal algs on dense SPN"""
        def fun1(node, *args):
            counter[0] += 1

        def fun2(node, *args):
            counter[0] += 1
            if node.is_op:
                return [None] * len(node.inputs)

        # Generate dense graph
        v1 = spn.IndicatorLeaf(num_vars=3, num_vals=2, name="IndicatorLeaf1")
        v2 = spn.IndicatorLeaf(num_vars=3, num_vals=2, name="IndicatorLeaf2")

        gen = spn.DenseSPNGenerator(num_decomps=2,
                                    num_subsets=3,
                                    num_mixtures=2,
                                    input_dist=spn.DenseSPNGenerator.InputDist.MIXTURE,
                                    num_input_mixtures=None)
        root = gen.generate(v1, v2)
        spn.generate_weights(root)

        # Run traversal algs and count nodes
        counter = [0]
        spn.compute_graph_up_down(root, down_fun=fun2, graph_input=1)
        c1 = counter[0]

        counter = [0]
        spn.compute_graph_up(root, val_fun=fun1)
        c2 = counter[0]

        counter = [0]
        spn.traverse_graph(root, fun=fun1, skip_params=False)
        c3 = counter[0]

        # Compare
        self.assertEqual(c1, c3)
        self.assertEqual(c2, c3)


if __name__ == '__main__':
    tf.test.main()
