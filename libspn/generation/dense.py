from collections import deque
from libspn import utils
from libspn.graph.op.conv_products import ConvProducts
from libspn.graph.node import Input
from libspn.graph.op.sum import Sum
from libspn.graph.op.parallel_sums import ParallelSums
from libspn.graph.op.sumslayer import SumsLayer
from libspn.graph.op.product import Product
from libspn.graph.op.permute_products import PermuteProducts
from libspn.graph.op.products_layer import ProductsLayer
from libspn.graph.op.concat import Concat
from libspn.graph.op.conv_sums import ConvSums
from libspn.graph.op.local_sums import LocalSums
from libspn.graph.algorithms import traverse_graph
from libspn.log import get_logger
from libspn.exceptions import StructureError
from libspn.utils.enum import Enum
from collections import OrderedDict, defaultdict
from itertools import chain, product
import tensorflow as tf
import numpy as np
import random


class DenseSPNGenerator:
    """Generates a dense SPN according to the algorithm described in
    Poon&Domingos UAI'11.

    Attributes:
        num_decomps (int): Number of decompositions at each level.
        num_subsets (int): Number of variable sub-sets for each decomposition.
        num_mixtures (int): Number of mixtures (sums) for each variable subset.
        input_dist (InputDist): Determines how inputs sharing the same scope
                                (for instance IndicatorLeaf for different values of a
                                random variable) should be included into the
                                generated structure.
        num_input_mixtures (int): Number of mixtures used for combining all
                                  inputs sharing scope when ``input_dist`` is
                                  set to ``MIXTURE``. If set to ``None``,
                                  ``num_mixtures`` is used.
        balanced (bool): Use only balanced decompositions, into subsets of
                         similar cardinality (differing by max 1).
        node_type (NodeType): Determines the type of op-node - single (Sum, Product),
                              block (ParallelSums, PermuteProducts) or layer (SumsLayer,
                              ProductsLayer) - to be used in the generated structure.

    """

    __logger = get_logger()
    __debug1 = __logger.debug1
    __debug2 = __logger.debug2
    __debug3 = __logger.debug3

    class InputDist(Enum):
        """Determines how inputs sharing the same scope (for instance IndicatorLeaf for
        different values of a random variable) should be included into the
        generated structure."""

        RAW = 0
        """Each input is considered a different distribution over the scope and
        used directly instead of a mixture as an input to product nodes for
        singleton variable subsets."""

        MIXTURE = 1
        """``input_num_mixtures`` mixtures are created over all the inputs
        sharing a scope, effectively creating ``input_num_mixtures``
        distributions over the scope. These mixtures are then used as inputs
        to product nodes for singleton variable subsets."""

    class NodeType(Enum):
        """Determines the type of op-node - single (Sum, Product), block (ParallelSums,
        PermuteProducts) or layer (SumsLayer, ProductsLayer) - to be used in the
        generated structure."""

        SINGLE = 0
        BLOCK = 1
        LAYER = 2

    class SubsetInfo:
        """Stores information about a single subset to be decomposed.

        Attributes:
            level(int): Number of the SPN layer where the subset is decomposed.
            subset(list of tuple of tuple): Subset of inputs to decompose
                                            grouped by scope.
            parents(list of Sum): List of sum nodes mixing the outputs of the
                                  generated decompositions. Should be the root
                                  node at the very top.
        """

        def __init__(self, level, subset, parents):
            self.level = level
            self.subset = subset
            self.parents = parents

    def __init__(self, num_decomps, num_subsets, num_mixtures,
                 input_dist=InputDist.MIXTURE, num_input_mixtures=None,
                 balanced=True, node_type=NodeType.LAYER):
        # Args
        if not isinstance(num_decomps, int) or num_decomps < 1:
            raise ValueError("num_decomps must be a positive integer")
        if not isinstance(num_subsets, int) or num_subsets < 1:
            raise ValueError("num_subsets must be a positive integer")
        if not isinstance(num_mixtures, int) or num_mixtures < 1:
            raise ValueError("num_mixtures must be a positive integer")
        if input_dist not in DenseSPNGenerator.InputDist:
            raise ValueError("Incorrect input_dist: %s", input_dist)
        if (num_input_mixtures is not None and
                (not isinstance(num_input_mixtures, int)
                 or num_input_mixtures < 1)):
            raise ValueError("num_input_mixtures must be None"
                             " or a positive integer")

        # Attributes
        self.num_decomps = num_decomps
        self.num_subsets = num_subsets
        self.num_mixtures = num_mixtures
        self.input_dist = input_dist
        self.balanced = balanced
        self.node_type = node_type
        if num_input_mixtures is None:
            self.num_input_mixtures = num_mixtures
        else:
            self.num_input_mixtures = num_input_mixtures

        # Stirling numbers and ratios for partition sampling
        self.__stirling = utils.Stirling()

    def generate(self, *inputs, rnd=None, root_name=None):
        """Generate the SPN.

        Args:
            inputs (input_like): Inputs to the generated SPN.
            rnd (Random): Optional. A custom instance of a random number generator
                          ``random.Random`` that will be used instead of the
                          default global instance. This permits using a generator
                          with a custom state independent of the global one.
            root_name (str): Name of the root node of the generated SPN.

        Returns:
           Sum: Root node of the generated SPN.
        """
        self.__debug1(
            "Generating dense SPN (num_decomps=%s, num_subsets=%s,"
            " num_mixtures=%s, input_dist=%s, num_input_mixtures=%s)",
            self.num_decomps, self.num_subsets,
            self.num_mixtures, self.input_dist, self.num_input_mixtures)
        inputs = [Input.as_input(i) for i in inputs]
        input_set = self.__generate_set(inputs)
        self.__debug1("Found %s distinct input scopes",
                      len(input_set))

        # Create root
        root = Sum(name=root_name)

        # Subsets left to process
        subsets = deque()
        subsets.append(DenseSPNGenerator.SubsetInfo(level=1,
                                                              subset=input_set,
                                                              parents=[root]))

        # Process subsets layer by layer
        self.__decomp_id = 1  # Id number of a decomposition, for info only
        while subsets:
            # Process whole layer (all subsets at the same level)
            level = subsets[0].level
            self.__debug1("Processing level %s", level)
            while subsets and subsets[0].level == level:
                subset = subsets.popleft()
                new_subsets = self.__add_decompositions(subset, rnd)
                for s in new_subsets:
                    subsets.append(s)

        # If NodeType is LAYER, convert the generated graph with LayerNodes
        return (self.convert_to_layer_nodes(root) if self.node_type ==
                DenseSPNGenerator.NodeType.LAYER else root)

    def __generate_set(self, inputs):
        """Generate a set of inputs to the generated SPN grouped by scope.

        Args:
            inputs (list of Input): List of inputs.

        Returns:
           list of tuple of tuple: A list where each elements is a tuple of
               all inputs to the generated SPN which share the same scope.
               Each of that scopes is guaranteed to be unique. That tuple
               contains tuples ``(node, index)`` which uniquely identify
               specific inputs.
        """
        scope_dict = {}  # Dict indexed by scope

        def add_input(scope, node, index):
            try:
                # Try appending to existing scope
                scope_dict[scope].add((node, index))
            except KeyError:
                # Scope not in dict, check if it overlaps with other scopes
                for s in scope_dict:
                    if s & scope:
                        raise StructureError("Differing scopes of inputs overlap")
                # Add to dict
                scope_dict[scope] = set([(node, index)])

        # Process inputs and group by scope
        for inpt in inputs:
            node_scopes = inpt.node.get_scope()
            if inpt.indices is None:
                for index, scope in enumerate(node_scopes):
                    add_input(scope, inpt.node, index)
            else:
                for index in inpt.indices:
                    add_input(node_scopes[index], inpt.node, index)

        # Convert to hashable tuples and sort
        # Sorting might improve performance due to branch prediction
        return [tuple(sorted(i)) for i in scope_dict.values()]

    def __add_decompositions(self, subset_info: SubsetInfo, rnd: random.Random):
        """Add nodes for a single subset, i.e. an instance of ``num_decomps``
        decompositions of ``subset`` into ``num_subsets`` sub-subsets with
        ``num_mixures`` mixtures per sub-subset.

        Args:
            subset_info(SubsetInfo): Info about the subset being decomposed.
            rnd (Random): A custom instance of a random number generator or
                          ``None`` if default global instance should be used.

        Returns:
            list of SubsetInfo: Info about each new generated subset, which
            requires further decomposition.
        """

        def subsubset_to_inputs_list(subsubset):
            """Convert sub-subsets into a list of tuples, where each
               tuple contains an input and a list of indices
            """
            subsubset_list = list(next(iter(subsubset)))
            # Create a list of unique inputs from sub-subsets list
            unique_inputs = list(set(s_subset[0]
                                 for s_subset in subsubset_list))
            # For each unique input, collect all associated indices
            # into a single list, then create a list of tuples,
            # where each tuple contains an unique input and it's
            # list of indices
            inputs_list = []
            for unique_inp in unique_inputs:
                indices_list = []
                for s_subset in subsubset_list:
                    if s_subset[0] == unique_inp:
                        indices_list.append(s_subset[1])
                inputs_list.append(tuple((unique_inp, indices_list)))

            return inputs_list

        # Get subset partitions
        self.__debug3("Decomposing subset:\n%s", subset_info.subset)
        num_elems = len(subset_info.subset)
        num_subsubsets = min(num_elems, self.num_subsets)  # Requested num subsets
        partitions = utils.random_partitions(subset_info.subset, num_subsubsets,
                                             self.num_decomps,
                                             balanced=self.balanced,
                                             rnd=rnd,
                                             stirling=self.__stirling)
        self.__debug2("Randomized %s decompositions of a subset"
                      " of %s elements into %s sets",
                      len(partitions), num_elems, num_subsubsets)

        # Generate nodes for each decomposition/partition
        subsubset_infos = []
        for part in partitions:
            self.__debug2("Decomposition %s: into %s subsubsets of cardinality %s",
                          self.__decomp_id, len(part), [len(s) for s in part])
            self.__debug3("Decomposition %s subsubsets:\n%s",
                          self.__decomp_id, part)
            # Handle each subsubset
            sums_id = 1
            prod_inputs = []
            for subsubset in part:
                if self.node_type == DenseSPNGenerator.NodeType.SINGLE:
                    # Use single-nodes
                    if len(subsubset) > 1:  # Decomposable further
                        # Add mixtures
                        with tf.name_scope("Sums%s.%s" % (self.__decomp_id, sums_id)):
                            sums = [Sum(name="Sum%s" % (i + 1))
                                    for i in range(self.num_mixtures)]
                            sums_id += 1
                        # Register the mixtures as inputs of products
                        prod_inputs.append([(s, 0) for s in sums])
                        # Generate subsubset info
                        subsubset_infos.append(DenseSPNGenerator.SubsetInfo(
                            level=subset_info.level + 1, subset=subsubset,
                            parents=sums))
                    else:  # Non-decomposable
                        if self.input_dist == DenseSPNGenerator.InputDist.RAW:
                            # Register the content of subset as inputs to products
                            prod_inputs.append(next(iter(subsubset)))
                        elif self.input_dist == DenseSPNGenerator.InputDist.MIXTURE:
                            # Add mixtures
                            with tf.name_scope("Sums%s.%s" % (self.__decomp_id, sums_id)):
                                sums = [Sum(name="Sum%s" % (i + 1))
                                        for i in range(self.num_input_mixtures)]
                                sums_id += 1
                            # Register the mixtures as inputs of products
                            prod_inputs.append([(s, 0) for s in sums])
                            # Create an inputs list
                            inputs_list = subsubset_to_inputs_list(subsubset)
                            # Connect inputs to mixtures
                            for s in sums:
                                s.add_values(*inputs_list)
                else:  # Use multi-nodes
                    if len(subsubset) > 1:  # Decomposable further
                        # Add mixtures
                        with tf.name_scope("Sums%s.%s" % (self.__decomp_id, sums_id)):
                            sums = ParallelSums(num_sums=self.num_mixtures,
                                                name="ParallelSums%s.%s" %
                                           (self.__decomp_id, sums_id))
                            sums_id += 1
                        # Register the mixtures as inputs of PermProds
                        prod_inputs.append(sums)
                        # Generate subsubset info
                        subsubset_infos.append(DenseSPNGenerator.SubsetInfo(
                                               level=subset_info.level + 1,
                                               subset=subsubset, parents=[sums]))
                    else:  # Non-decomposable
                        if self.input_dist == DenseSPNGenerator.InputDist.RAW:
                            # Create an inputs list
                            inputs_list = subsubset_to_inputs_list(subsubset)
                            if len(inputs_list) > 1:
                                inputs_list = [Concat(*inputs_list)]
                            # Register the content of subset as inputs to PermProds
                            [prod_inputs.append(inp) for inp in inputs_list]
                        elif self.input_dist == DenseSPNGenerator.InputDist.MIXTURE:
                            # Create an inputs list
                            inputs_list = subsubset_to_inputs_list(subsubset)
                            # Add mixtures
                            with tf.name_scope("Sums%s.%s" % (self.__decomp_id, sums_id)):
                                sums = ParallelSums(*inputs_list,
                                                    num_sums=self.num_input_mixtures,
                                                    name="ParallelSums%s.%s" %
                                               (self.__decomp_id, sums_id))
                                sums_id += 1
                            # Register the mixtures as inputs of PermProds
                            prod_inputs.append(sums)
            # Add product nodes
            if self.node_type == DenseSPNGenerator.NodeType.SINGLE:
                products = self.__add_products(prod_inputs)
            else:
                products = ([PermuteProducts(*prod_inputs, name="PermuteProducts%s" % self.__decomp_id)]
                            if len(prod_inputs) > 1 else prod_inputs)
            # Connect products to each parent Sum
            for p in subset_info.parents:
                p.add_values(*products)
            # Increment decomposition id
            self.__decomp_id += 1
        return subsubset_infos

    def __add_products(self, prod_inputs):
        """
        Add product nodes for a single decomposition and connect them to their
        input nodes.

        Args:
            prod_inputs (list of list of Node): A list of lists of nodes
                being inputs to the products, grouped by scope.

        Returns:
            list of Product: A list of product nodes.
        """
        selected = [0 for _ in prod_inputs]  # Input selected for each scope
        cont = True
        products = []
        product_num = 1
        with tf.name_scope("Products%s" % self.__decomp_id):
            while cont:
                if len(prod_inputs) > 1:
                    # Add a product node
                    products.append(Product(*[pi[s] for (pi, s) in
                                              zip(prod_inputs, selected)],
                                            name="Product%s" % product_num))
                else:
                    products.append(*[pi[s] for (pi, s) in
                                      zip(prod_inputs, selected)])
                product_num += 1
                # Increment selected
                cont = False
                for i, group in enumerate(prod_inputs):
                    if selected[i] < len(group) - 1:
                        selected[i] += 1
                        for j in range(i):
                            selected[j] = 0
                        cont = True
                        break
        return products

    def convert_to_layer_nodes(self, root):
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
                            not(i.is_param or i.is_var or
                                isinstance(i.node, (SumsLayer, ProductsLayer, ConvSums,
                                                    ConvProducts, Concat, LocalSums)))):
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
                # After all nodes at a certain depth are modelled into a Layer-node,
                # set num-prods parameter accordingly
                prods_layer.set_prod_sizes(num_or_size_prods)
            elif isinstance(depths[depth][0], (SumsLayer, ProductsLayer, Concat)):  # A Concat node
                pass
            else:
                raise StructureError("Unknown node-type: {}".format(depths[depth][0]))

        return root
