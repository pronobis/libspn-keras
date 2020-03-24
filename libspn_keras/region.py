import itertools
import operator
from collections import deque, OrderedDict
import functools
import numpy as np

from libspn_keras import BackpropMode
from libspn_keras.layers import DenseSum, DenseProduct, RootSum
from libspn_keras.layers.to_regions import ToRegions
from libspn_keras.layers.permute_and_pad_scopes import PermuteAndPadScopes
import tensorflow as tf


class OverlappingScopesException(Exception):
    pass


class RegionVariable:

    def __init__(self, index):
        self.index = index
        self.scope = [self]

    def __repr__(self):
        return "x{}".format(self.index)


def _assert_no_scope_overlap(children):
    for c0, c1 in itertools.combinations(children, 2):
        if set(c0.scope) & set(c1.scope):
            raise OverlappingScopesException(
                "Children {} and {} have overlapping scopes".format(c0, c1))


class RegionNode:

    def __init__(self, children):
        _assert_no_scope_overlap(children)
        self.children = children
        self.scope = functools.reduce(operator.concat, [c.scope for c in children])

    def __repr__(self):
        return '{' + ", ".join(str(s) for s in self.children) + '}'


def generate_poon_domingos_region_graph(num_rows, num_cols):

    @functools.lru_cache()
    def create_region(i0, i1, j0, j1):
        if i1 - i0 == 1 and j1 - j0 == 1:
            return [RegionVariable(i0 * num_rows + j0)]
        sub_regions = []
        for ix in range(i0 + 1, i1):
            sub_regions += map(
                RegionNode,
                itertools.product(create_region(i0, ix, j0, j1), create_region(ix, i1, j0, j1))
            )

        for jx in range(j0 + 1, j1):
            sub_regions += map(
                RegionNode,
                itertools.product(create_region(i0, i1, j0, jx), create_region(i0, i1, jx, j1))
            )
        return sub_regions

    return create_region(0, num_rows, 0, num_cols)


def get_nodes_to_depth_mapping(root):
    node_to_depth = dict()
    node_to_depth[root] = 0

    def _node_fn(node):
        # Add to Parents dict
        if isinstance(node, RegionNode):
            for child in node.children:
                node_to_depth[child] = node_to_depth[node] + 1
    traverse_region_graph(root, _node_fn)
    return node_to_depth


def get_depth_to_nodes_mapping(root, node_to_depth_mapping=None):

    node_to_depth_mapping = node_to_depth_mapping or get_nodes_to_depth_mapping(root)
    depth_to_nodes = OrderedDict()

    for node, depth in sorted(node_to_depth_mapping.items(), key=operator.itemgetter(1)):
        if depth not in depth_to_nodes:
            depth_to_nodes[depth] = [node]
        else:
            depth_to_nodes[depth].append(node)

    return depth_to_nodes


def compute_max_num_children_by_depth(root, depth_to_nodes_mapping=None):
    depth_to_nodes_mapping = depth_to_nodes_mapping or get_depth_to_nodes_mapping(root)
    max_num_children_by_depth = OrderedDict()
    for depth, nodes in depth_to_nodes_mapping.items():
        if any(isinstance(n, RegionNode) for n in nodes):
            max_num_children_by_depth[depth] = max(
                len(node.children) for node in nodes if isinstance(node, RegionNode)
            )
    return max_num_children_by_depth


def collect_variable_nodes(root):
    variable_nodes = []

    def _gather(node):
        if isinstance(node, RegionVariable):
            variable_nodes.append(node)
        else:
            [_gather(child) for child in node.children]

    _gather(root)
    return variable_nodes


def region_graph_to_permutations_and_prods_per_depth(root):
    node_to_depth_mapping = get_nodes_to_depth_mapping(
        root
    )
    depth_to_nodes_mapping = get_depth_to_nodes_mapping(
        root, node_to_depth_mapping=node_to_depth_mapping
    )
    max_num_children_by_depth = compute_max_num_children_by_depth(
        root, depth_to_nodes_mapping=depth_to_nodes_mapping
    )
    max_num_children_by_depth_cum_prod = np.cumprod(
        [1] + list(reversed(max_num_children_by_depth.values())))
    max_depth = max(node_to_depth_mapping.values())

    permutations = []

    def recurse_permutation(node):
        depth = node_to_depth_mapping[node]
        if isinstance(node, RegionVariable) and depth < max_depth:
            permutations.append(node.index)
            for _ in range(max_num_children_by_depth_cum_prod[max_depth]
                           // max_num_children_by_depth_cum_prod[depth + 1]):
                permutations.append(-1)
        elif isinstance(node, RegionVariable) and depth == max_depth:
            permutations.append(node.index)
        elif isinstance(node, RegionNode):
            [recurse_permutation(child) for child in node.children]
            for _ in range(max_num_children_by_depth_cum_prod[depth] // max_num_children_by_depth_cum_prod[depth + 1] - len(node.children)):
                children_beneath = \
                    max_num_children_by_depth_cum_prod[max_depth] // \
                    max_num_children_by_depth_cum_prod[depth + 1]
                permutations.extend([-1] * children_beneath)

    recurse_permutation(root)

    return permutations, list(reversed(max_num_children_by_depth.values()))


def region_graph_to_dense_spn(
    root, leaf_node, num_sums_iterable, logspace_accumulators=False,
    backprop_mode=BackpropMode.GRADIENT,
    accumulator_initializer=None, linear_accumulator_constraint=None, product_first=True,
    num_classes=None,  with_root=True, return_weighted_child_logits=True
):
    """

    """
    permutation, num_factors_leaf_to_root = region_graph_to_permutations_and_prods_per_depth(root)

    sum_kwargs = dict(
        logspace_accumulators=logspace_accumulators, backprop_mode=backprop_mode,
        accumulator_initializer=accumulator_initializer,
        linear_accumulator_constraint=linear_accumulator_constraint
    )

    sum_product_stack = []
    if not product_first:
        sum_product_stack.append(
            DenseSum(num_sums=next(num_sums_iterable), **sum_kwargs)
        )
    for depth, num_factors in enumerate(num_factors_leaf_to_root):
        sum_product_stack.append(
            DenseProduct(num_factors=num_factors)
        )
        if depth == len(num_factors_leaf_to_root) - 1:
            break
        sum_product_stack.append(
            DenseSum(num_sums=next(num_sums_iterable), **sum_kwargs)
        )

    if num_classes is not None:
        sum_product_stack.append(DenseSum(num_sums=num_classes, **sum_kwargs))

    if with_root:
        sum_product_stack.append(RootSum(
            return_weighted_child_logits=return_weighted_child_logits,
            **sum_kwargs
        ))

    pre_stack = [
        ToRegions(num_decomps=1, input_shape=[len(collect_variable_nodes(root))]),
        leaf_node,
        PermuteAndPadScopes(num_decomps=1, permutations=np.asarray([permutation]))
    ]

    return tf.keras.Sequential(pre_stack + sum_product_stack)


def traverse_region_graph(root, fun):
    """Runs ``fun`` on descendants of ``root`` (including ``root``) by
    traversing the graph breadth-first until ``fun`` returns True.

    Args:
        root (Node): The root of the SPN graph.
        fun (function): A function ``fun(node)`` executed once for every node of
                        the graph. It should return ``True`` if traversing
                        should be stopped.
        skip_params (bool): If ``True``, the param nodes will not be traversed.

    Returns:
        Node: Returns the last traversed node (the one for which ``fun``
        returned True) or ``None`` if ``fun`` never returned ``True``.
    """
    visited_nodes = set()  # Set of visited nodes
    queue = deque()
    queue.append(root)

    while queue:
        next_node = queue.popleft()
        if next_node not in visited_nodes:
            fun(next_node)

            visited_nodes.add(next_node)

            if isinstance(next_node, RegionNode):
                for child in next_node.children:
                    queue.append(child)
