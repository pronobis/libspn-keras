import abc
from collections import deque, OrderedDict
import functools
import itertools
import operator
from typing import Callable, Dict, Iterator, List, Mapping, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.initializers import Initializer

from libspn_keras.layers import BaseLeaf, DenseProduct, DenseSum, RootSum
from libspn_keras.layers.flat_to_regions import FlatToRegions
from libspn_keras.layers.permute_and_pad_scopes import PermuteAndPadScopes
from libspn_keras.sum_ops import SumOpGradBackprop


class OverlappingScopesException(Exception):
    """Thrown when scopes of children overlap."""


class Region(abc.ABC):
    """
    Represents a region in the SPN.

    Regions graphs are DAGs just like SPNs, but they do not represent sums and products. They are merely
    used for defining the scope structure of the SPN.
    """

    @property
    @abc.abstractmethod
    def scope(self):  # noqa: ANN201
        """Obtain scope as a list of nodes."""

    @property
    @abc.abstractmethod
    def children(self):  # noqa: ANN201
        """Obtain children as a list of nodes."""

    @staticmethod
    def _assert_no_scope_overlap(children) -> None:  # noqa: ANN001
        """
        Ensure validity of the SPN represented by asserting none of the children overlap in their scopes.

        Args:
            children: List of Regions.

        Raises:
            OverlappingScopesException: the scopes of two children overlap.
        """
        for c0, c1 in itertools.combinations(children, 2):
            if set(c0.scope) & set(c1.scope):
                raise OverlappingScopesException(
                    "Children {} and {} have overlapping scopes".format(c0, c1)
                )


class RegionNode(Region):
    """
    Represents a region in the SPN.

    Regions graphs are DAGs just like SPNs, but they do not represent sums and products. They are merely
    used for defining the scope structure of the SPN.

    A RegionNode is a node with children, as opposed to a RegionVariable, which is a leaf node.

    Args:
          children (list of RegionNode or RegionVariable): children of this node. Scope of the
            resulting node is the union of the scopes of its children. The scopes of these children
            must not overlap (pairwise disjoint)
    """

    def __init__(self, children: List[Region]):
        self._assert_no_scope_overlap(children)
        self._children = children
        self._scope = list(
            functools.reduce(operator.concat, [c.scope for c in children])  # type: ignore
        )

    def __repr__(self):
        return "{" + ", ".join(str(s) for s in self._children) + "}"

    @property
    def children(self) -> List[Region]:
        """
        Obtain children as a list of nodes.

        Returns:
            A list of Regions
        """
        return self._children

    @property
    def scope(self) -> List[Region]:
        """
        Obtain scope as a list of nodes.

        Returns:
            A list of Regions
        """
        return self._scope


class RegionVariable(Region):
    """
    Represents a region in the SPN.

    Regions graphs are DAGs just like SPNs, but they do not represent sums and products. They are merely
    used for defining the scope structure of the SPN.

    A RegionVariable is a leaf node without children, as opposed to a RegionNode which is a node with
    children.

    Args:
        index: Index of the variable
    """

    def __init__(self, index: int):
        self.index = index

    def __repr__(self):
        return "x{}".format(self.index)

    @property
    def children(self) -> List[Region]:
        """
        Obtain children as a list of nodes.

        Returns:
            A list of Regions
        """
        return []

    @property
    def scope(self) -> List[Region]:
        """
        Obtain scope as a list of nodes.

        Returns:
            A list of Regions
        """
        return [self]


def region_graph_to_dense_spn(
    region_graph_root: RegionNode,
    leaf_node: BaseLeaf,
    num_sums_iterable: Iterator[int],
    return_weighted_child_logits: bool,
    logspace_accumulators: bool = False,
    accumulator_initializer: Optional[Initializer] = None,
    linear_accumulator_constraint: Optional[Constraint] = None,
    product_first: bool = True,
    num_classes: Optional[int] = None,
    with_root: bool = True,
) -> tf.keras.Sequential:
    """
    Convert a region graph (built from :class:`RegionNode` and :class:`RegionVar`) to a dense SPN.

    Args:
        region_graph_root: Root of the region graph
        leaf_node: Node to insert at the leaf of the SPN
        num_sums_iterable: Number of sums for all but the last root sum
            layer from bottom to top
        logspace_accumulators: Whether to represent accumulators of weights in logspace
            or not
        accumulator_initializer: Initializer for accumulators
        linear_accumulator_constraint: Constraint for linear
            accumulator, default: GreaterThanEpsilon
        product_first: Whether to start with a product layer
        num_classes: Number of classes at output. If ``None``, will not use 'latent' sums
            at the end but will instead directly connect the root to the final layer of the dense
            stack. This means if set to ``None`` the SPN cannot be used for classification.
        with_root: If ``True``, sets a ``RootSum`` as the final layer.
        return_weighted_child_logits: Whether to return weighted child logits. If ``

    Returns:
        A Sum-Product Network as a tf.keras.Sequential model.
    """
    (
        permutation,
        num_factors_leaf_to_root,
    ) = _region_graph_to_permutations_and_prods_per_depth(region_graph_root)

    sum_kwargs = dict(
        logspace_accumulators=logspace_accumulators,
        accumulator_initializer=accumulator_initializer,
        linear_accumulator_constraint=linear_accumulator_constraint,
        sum_op=SumOpGradBackprop(),
    )

    sum_product_stack: List[tf.keras.layers.Layer] = []
    if not product_first:
        sum_product_stack.append(
            DenseSum(num_sums=next(num_sums_iterable), **sum_kwargs)
        )
    for depth, num_factors in enumerate(num_factors_leaf_to_root):
        sum_product_stack.append(DenseProduct(num_factors=num_factors))
        if depth == len(num_factors_leaf_to_root) - 1:
            break
        sum_product_stack.append(
            DenseSum(num_sums=next(num_sums_iterable), **sum_kwargs)
        )

    if num_classes is not None:
        sum_product_stack.append(DenseSum(num_sums=num_classes, **sum_kwargs))  # type: ignore

    if with_root:
        sum_product_stack.append(
            RootSum(
                return_weighted_child_logits=return_weighted_child_logits, **sum_kwargs  # type: ignore
            )
        )

    pre_stack: List[tf.keras.layers.Layer] = [
        FlatToRegions(
            num_decomps=1, input_shape=[len(_collect_variable_nodes(region_graph_root))]
        ),
        leaf_node,
        PermuteAndPadScopes(permutations=np.asarray([permutation])),
    ]

    return tf.keras.Sequential(pre_stack + sum_product_stack)


def _get_nodes_to_depth_mapping(root: Region) -> Dict[Region, int]:
    node_to_depth = dict()
    node_to_depth[root] = 0

    def _node_fn(node: Region) -> None:
        # Add to Parents dict
        for child in node.children:
            node_to_depth[child] = node_to_depth[node] + 1

    _traverse_region_graph(root, _node_fn)
    return node_to_depth


def _get_depth_to_nodes_mapping(
    root: Region, node_to_depth_mapping: Optional[Mapping[Region, int]] = None,
) -> Dict[int, List[Region]]:

    node_to_depth_mapping = node_to_depth_mapping or _get_nodes_to_depth_mapping(root)
    depth_to_nodes = OrderedDict()

    for node, depth in sorted(
        node_to_depth_mapping.items(), key=operator.itemgetter(1)
    ):
        if depth not in depth_to_nodes:
            depth_to_nodes[depth] = [node]
        else:
            depth_to_nodes[depth].append(node)

    return depth_to_nodes


def _compute_max_num_children_by_depth(
    root: Region, depth_to_nodes_mapping: Mapping[int, List[Region]] = None,
) -> Dict[int, int]:
    depth_to_nodes_mapping = depth_to_nodes_mapping or _get_depth_to_nodes_mapping(root)
    max_num_children_by_depth = OrderedDict()
    for depth, nodes in depth_to_nodes_mapping.items():
        if any(isinstance(n, RegionNode) for n in nodes):
            max_num_children_by_depth[depth] = max(
                len(node.children) for node in nodes if isinstance(node, RegionNode)
            )
    return max_num_children_by_depth


def _collect_variable_nodes(root: RegionNode) -> List[Region]:
    variable_nodes: List[Region] = []

    def _gather(node: Region) -> None:
        if isinstance(node, RegionVariable):
            variable_nodes.append(node)
        else:
            for child in node.children:
                _gather(child)

    _gather(root)
    return variable_nodes


def _region_graph_to_permutations_and_prods_per_depth(
    root: RegionNode,
) -> Tuple[list, List[int]]:
    node_to_depth_mapping = _get_nodes_to_depth_mapping(root)
    depth_to_nodes_mapping = _get_depth_to_nodes_mapping(
        root, node_to_depth_mapping=node_to_depth_mapping
    )
    max_num_children_by_depth = _compute_max_num_children_by_depth(
        root, depth_to_nodes_mapping=depth_to_nodes_mapping
    )
    max_num_children_by_depth_cum_prod = np.cumprod(
        [1] + list(reversed(list(max_num_children_by_depth.values())))
    )
    max_depth = max(node_to_depth_mapping.values())

    permutations = []

    def recurse_permutation(node: Region) -> None:
        depth = node_to_depth_mapping[node]
        if isinstance(node, RegionVariable) and depth < max_depth:
            permutations.append(node.index)
            for _ in range(
                max_num_children_by_depth_cum_prod[max_depth]
                // max_num_children_by_depth_cum_prod[depth + 1]
            ):
                permutations.append(-1)
        elif isinstance(node, RegionVariable) and depth == max_depth:
            permutations.append(node.index)
        elif isinstance(node, RegionNode):
            for child in node.children:
                recurse_permutation(child)
            for _ in range(
                max_num_children_by_depth_cum_prod[depth]
                // max_num_children_by_depth_cum_prod[depth + 1]
                - len(node.children)
            ):
                children_beneath = (
                    max_num_children_by_depth_cum_prod[max_depth]
                    // max_num_children_by_depth_cum_prod[depth + 1]
                )
                permutations.extend([-1] * children_beneath)

    recurse_permutation(root)

    return permutations, list(reversed(list(max_num_children_by_depth.values())))


def _traverse_region_graph(root: Region, fun: Callable[[Region], None]) -> None:
    """
    Run ``fun`` on descendants of ``root`` (including ``root``) breadth-first.

    Args:
        root (Region): The root of the SPN graph.
        fun (function): A function ``fun(node)`` executed once for every node of
                        the graph. It should return ``True`` if traversing
                        should be stopped.
    """
    visited_nodes = set()  # Set of visited nodes
    queue: deque = deque()
    queue.append(root)

    while queue:
        next_node = queue.popleft()
        if next_node not in visited_nodes:
            fun(next_node)

            visited_nodes.add(next_node)

            for child in next_node.children:
                queue.append(child)
