from collections.__init__ import defaultdict, OrderedDict
from itertools import product, chain

import numpy as np
import tensorflow as tf

from libspn.graph.op.sum import Sum
from libspn.graph.op.sumslayer import SumsLayer
from libspn.graph.op.parallel_sums import ParallelSums
from libspn.graph.op.products_layer import ProductsLayer
from libspn.graph.op.product import Product
from libspn.graph.op.permute_products import PermuteProducts
from libspn.graph.op.concat import Concat
from libspn.exceptions import StructureError
from libspn.graph.algorithms import traverse_graph


def convert_to_layer_nodes(root):
    """
    At each level in the SPN rooted in the 'root' node, model all the nodes
    as a single layer-node.

    Args:
        root (Node): The root of the SPN graph.

    Returns:
        root (Node): The root of the SPN graph, with each layer modelled as a
                     single layer-node.
    """

    parents = defaultdict(list)
    depths = defaultdict(list)
    node_to_depth = OrderedDict()
    node_to_depth[root] = 1

    def get_parents(node):
        # Add to Parents dict
        if node.is_op:
            for i in node.inputs:
                if (i and  # Input not empty
                        not(i.is_param or i.is_var)):
                    parents[i.node].append(node)
                    node_to_depth[i.node] = node_to_depth[node] + 1

    def permute_inputs(input_values, input_sizes):
        # For a given list of inputs and their corresponding sizes, create a
        # nested-list of (input, index) pairs.
        # E.g: input_values = [(A, [2, 5]), (B, None)]
        #      input_sizes = [2, 3]
        #      inputs = [[('A', 2), ('A', 5)],
        #                [('B', 0), ('B', 1), ('B', 2)]]
        inputs = [list(product([inp.node], inp.indices)) if inp and inp.indices
                  else list(product([inp.node], list(range(inp_size)))) for
                  inp, inp_size in zip(input_values, input_sizes)]

        # For a given nested-list of (input, index) pairs, permute over the inputs
        # E.g: permuted_inputs = [('A', 2), ('B', 0),
        #                         ('A', 2), ('B', 1),
        #                         ('A', 2), ('B', 2),
        #                         ('A', 5), ('B', 0),
        #                         ('A', 5), ('B', 1),
        #                         ('A', 5), ('B', 2)]
        permuted_inputs = list(product(*[inps for inps in inputs]))
        return list(chain(*permuted_inputs))

    # Create a parents dictionary of the SPN graph
    traverse_graph(root, fun=get_parents, skip_params=True)

    # Create a depth dictionary of the SPN graph
    for key, value in node_to_depth.items():
        depths[value].append(key)
    spn_depth = len(depths)

    # Iterate through each depth of the SPN, starting from the deepest layer,
    # moving up to the root node
    for depth in range(spn_depth, 1, -1):
        if isinstance(depths[depth][0], (Sum, ParallelSums)):  # A Sums Layer
            # Create a default SumsLayer node
            with tf.name_scope("Layer%s" % depth):
                sums_layer = SumsLayer(name="SumsLayer-%s.%s" % (depth, 1))
            # Initialize a counter for keeping track of number of sums
            # modelled in the layer node
            layer_num_sums = 0
            # Initialize an empty list for storing sum-input-sizes of sums
            # modelled in the layer node
            num_or_size_sums = []
            # Iterate through each node at the current depth of the SPN
            for node in depths[depth]:
                # TODO: To be replaced with node.num_sums once AbstractSums
                # class is introduced
                # No. of sums modelled by the current node
                node_num_sums = (1 if isinstance(node, Sum) else node.num_sums)
                # Add Input values of the current node to the SumsLayer node
                sums_layer.add_values(*node.values * node_num_sums)
                # Add sum-input-size, of each sum modelled in the current node,
                # to the list
                num_or_size_sums += [sum(node.get_input_sizes()[2:])] * node_num_sums
                # Visit each parent of the current node
                for parent in parents[node]:
                    try:
                        # 'Values' in case parent is an Op node
                        values = list(parent.values)
                    except AttributeError:
                        # 'Inputs' in case parent is a Concat node
                        values = list(parent.inputs)
                    # Iterate through each input value of the current parent node
                    for i, value in enumerate(values):
                        # If the value is the current node
                        if value.node == node:
                            # Check if it has indices
                            if value.indices is not None:
                                # If so, then just add the num-sums of the
                                # layer-op as offset
                                indices = (np.asarray(value.indices) +
                                           layer_num_sums).tolist()
                            else:
                                # If not, then create a list accrodingly
                                indices = list(range(layer_num_sums,
                                                     (layer_num_sums +
                                                      node_num_sums)))
                            # Replace previous (node) Input value in the
                            # current parent node, with the new layer-node value
                            values[i] = (sums_layer, indices)
                            break  # Once child-node found, don't have to search further
                    # Reset values of the current parent node, by including
                    # the new child (Layer-node)
                    try:
                        # set 'values' in case parent is an Op node
                        parent.set_values(*values)
                    except AttributeError:
                        # set 'inputs' in case parent is a Concat node
                        parent.set_inputs(*values)
                # Increment num-sums-counter of the layer-node
                layer_num_sums += node_num_sums
                # Disconnect
                node.disconnect_inputs()

            # After all nodes at a certain depth are modelled into a Layer-node,
            # set num-sums parameter accordingly
            sums_layer.set_sum_sizes(num_or_size_sums)
        elif isinstance(depths[depth][0], (Product, PermuteProducts)):  # A Products Layer
            with tf.name_scope("Layer%s" % depth):
                prods_layer = ProductsLayer(name="ProductsLayer-%s.%s" % (depth, 1))
            # Initialize a counter for keeping track of number of prods
            # modelled in the layer node
            layer_num_prods = 0
            # Initialize an empty list for storing prod-input-sizes of prods
            # modelled in the layer node
            num_or_size_prods = []
            # Iterate through each node at the current depth of the SPN
            for node in depths[depth]:
                # Get input values and sizes of the product node
                input_values = list(node.values)
                input_sizes = list(node.get_input_sizes())
                if isinstance(node, PermuteProducts):
                    # Permute over input-values to model permuted products
                    input_values = permute_inputs(input_values, input_sizes)
                    node_num_prods = node.num_prods
                    prod_input_size = len(input_values) // node_num_prods
                elif isinstance(node, Product):
                    node_num_prods = 1
                    prod_input_size = int(sum(input_sizes))

                # Add Input values of the current node to the ProductsLayer node
                prods_layer.add_values(*input_values)
                # Add prod-input-size, of each product modelled in the current
                # node, to the list
                num_or_size_prods += [prod_input_size] * node_num_prods
                # Visit each parent of the current node
                for parent in parents[node]:
                    values = list(parent.values)
                    # Iterate through each input value of the current parent node
                    for i, value in enumerate(values):
                        # If the value is the current node
                        if value.node == node:
                            # Check if it has indices
                            if value.indices is not None:
                                # If so, then just add the num-prods of the
                                # layer-op as offset
                                indices = value.indices + layer_num_prods
                            else:
                                # If not, then create a list accrodingly
                                indices = list(range(layer_num_prods,
                                                     (layer_num_prods +
                                                      node_num_prods)))
                            # Replace previous (node) Input value in the
                            # current parent node, with the new layer-node value
                            values[i] = (prods_layer, indices)
                    # Reset values of the current parent node, by including
                    # the new child (Layer-node)
                    parent.set_values(*values)
                # Increment num-prods-counter of the layer node
                layer_num_prods += node_num_prods
                # Disconnect
                node.disconnect_inputs()

            # After all nodes at a certain depth are modelled into a Layer-node,
            # set num-prods parameter accordingly
            prods_layer.set_prod_sizes(num_or_size_prods)

        elif isinstance(depths[depth][0], (SumsLayer, ProductsLayer, Concat)):  # A Concat node
            pass
        else:
            raise StructureError("Unknown node-type: {}".format(depths[depth][0]))

    return root