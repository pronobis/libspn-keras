# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

import tensorflow as tf
from libspn.graph.sum import Sum
from libspn.graph.parsums import ParSums
from libspn.graph.algorithms import compute_graph_up


class WeightsGenerator:
    """Generates matching weights nodes for each sum node in the SPN graph
    specified by ``root``.

    Weights should be generated once all inputs are added to this node,
    otherwise the number of weight values will be incorrect.

    Attributes:
        init_value: Initial value of the weights. For possible values, see
                    :meth:`~libspn.utils.broadcast_value`.
        trainable: See :class:`~libspn.Weights`.
    """

    def __init__(self, init_value=1, trainable=True):
        self._weights = {}
        self.init_value = init_value
        self.trainable = trainable

    @property
    def weights(self):
        """dict: A dictionary of :class:`~libspn.Weights` indexed by the SPN
        :class:`~libspn.Sum` node to which the weights are attached."""
        return self._weights

    def generate(self, root):
        """Generate the weight nodes.

        Args:
            root: The root node of the SPN graph.
        """
        def gen(node, *input_out_sizes):
            if isinstance(node, (Sum, ParSums)):
                self._weights[node] = node.generate_weights(
                    init_value=self.init_value, trainable=self.trainable,
                    input_sizes=node._gather_input_sizes(*input_out_sizes))
            return node._compute_out_size(*input_out_sizes)

        with tf.name_scope("Weights"):
            self._weights = {}
            # Traverse the graph and compute the out_size for each node
            return compute_graph_up(root, val_fun=gen)


def generate_weights(root, init_value=1, trainable=True):
    """A helper function for quick generation of sum weights in the SPN graph.

    Args:
        root (Node): The root node of the SPN graph.
        init_value: Initial value of the weights. For possible values, see
                    :meth:`~libspn.utils.broadcast_value`.
        trainable: See :class:`~libspn.Weights`.
    """
    WeightsGenerator(init_value=init_value, trainable=trainable).generate(root)
