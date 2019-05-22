import tensorflow as tf
from libspn.graph.op.block_sum import BlockSum
from libspn.graph.op.base_sum import BaseSum
from libspn.graph.algorithms import compute_graph_up


class WeightsGenerator:
    """Generates matching weights nodes for each sum node in the SPN graph
    specified by ``root``.

    Weights should be generated once all inputs are added to this node,
    otherwise the number of weight values will be incorrect.

    Attributes:
        initializer: Initial value of the weights.
        trainable: See :class:`~libspn.Weights`.
        log (bool): If "True", the weights are represented in log space.
    """

    def __init__(self, initializer=tf.initializers.constant(1.0), trainable=True, log=False):
        self._weights = {}
        self.initializer = initializer
        self.trainable = trainable
        self._log = log

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
            if isinstance(node, (BaseSum, BlockSum)):
                self._weights[node] = node.generate_weights(
                    initializer=self.initializer, trainable=self.trainable,
                    input_sizes=node._gather_input_sizes(*input_out_sizes),
                    log=self._log)
            return node._compute_out_size(*input_out_sizes)

        with tf.name_scope("Weights"):
            self._weights = {}
            # Traverse the graph and compute the out_size for each node
            return compute_graph_up(root, val_fun=gen)


def generate_weights(root, initializer=tf.initializers.constant(1.0), trainable=True, log=False):
    """A helper function for quick generation of sum weights in the SPN graph.

    Args:
        root (Node): The root node of the SPN graph.
        init_value: Initial value of the weights.
        trainable: See :class:`~libspn.Weights`.
        log (bool): If "True", the weights are represented in log space.
    """
    WeightsGenerator(initializer=initializer, trainable=trainable, log=log).generate(root)
