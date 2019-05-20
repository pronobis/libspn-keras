from libspn.graph.node import OpNode, Input, BlockNode
from libspn.inference.type import InferenceType
from libspn.exceptions import StructureError
import libspn.utils as utils
from libspn.log import get_logger
import tensorflow as tf


@utils.register_serializable
class BlockMergeDecomps(BlockNode):

    logger = get_logger()
    info = logger.info

    """An abstract node representing sums in an SPN.

    Args:
        *values (input_like): Inputs providing input values to this node.
            See :meth:`~libspn.Input.as_input` for possible values.
        num_sums (int): Number of Sum ops modelled by this node.
        sum_sizes (list): A list of ints corresponding to the sizes of each sum. If both num_sums
                          and sum_sizes are given, we should have len(sum_sizes) == num_sums.
        batch_axis (int): The index of the batch axis.
        op_axis (int): The index of the op axis that contains the individual sums being modeled.
        reduce_axis (int): The axis over which to perform summing (or max for MPE)
        name (str): Name of the node.

    Attributes:
        inference_type(InferenceType): Flag indicating the preferred inference
                                       type for this node that will be used
                                       during value calculation and learning.
                                       Can be changed at any time and will be
                                       used during the next inference/learning
                                       op generation.
    """
    def __init__(self, *values, factor, inference_type=InferenceType.MARGINAL,
                 name="TensorMergeDecomps", input_format="SDBN", output_format="SDBN"):
        super().__init__(
            inference_type=inference_type, name=name, input_format=input_format,
            output_format=output_format, num_scopes=1)
        self.set_values(*values)
        self._factor = factor

    def set_values(self, *values):
        self._values = self._parse_inputs(*values)

    def _get_num_input_scopes(self):
        if not self._values:
            raise StructureError("{}: cannot get num input scopes since this "
                                 "node has no children.".format(self))
        return self._values[0].node.num_vars

    def _assert_generated(self):
        if self._perms is None:
            raise StructureError("{}: First need to generate decompositions.".format(self))

    @property
    def dim_nodes(self):
        if not self._values:
            raise StructureError("{}: cannot get num outputs per decomp and scope since this "
                                 "node has no children.".format(self))
        child = self._values[0].node
        return child.dim_nodes * self._factor

    @property
    def dim_decomps(self):
        if not self._values:
            raise StructureError("{}: cannot get num outputs per decomp and scope since this "
                                 "node has no children.".format(self))
        child = self._values[0].node
        return child.dim_decomps // self._factor

    def _compute_out_size(self, *input_out_sizes):
        pass

    @property
    def dim_scope(self):
        return 1

    @utils.docinherit(OpNode)
    def serialize(self):
        raise NotImplementedError()

    @utils.docinherit(OpNode)
    def deserialize(self, data):
        raise NotImplementedError()

    @utils.docinherit(OpNode)
    def deserialize_inputs(self, data, nodes_by_name):
        raise NotImplementedError()

    @property
    @utils.docinherit(OpNode)
    def inputs(self):
        return self._values

    @property
    def values(self):
        """list of Input: List of value inputs."""
        return self._values

    def add_values(self, *values):
        """Add more inputs providing input values to this node.

        Args:
            *values (input_like): Inputs providing input values to this node.
                See :meth:`~libspn.Input.as_input` for possible values.
        """
        self._values += self._parse_inputs(*values)
        self._reset_sum_sizes()

    @utils.docinherit(OpNode)
    @utils.lru_cache
    def _compute_log_value(self, *value_tensors):
        child = value_tensors[0]
        child_node = self.values[0].node
        if child_node.dim_scope != 1:
            raise StructureError("Can only merge decompositions if scope axis is 1")
        if child_node.dim_decomps % self._factor != 0:
            raise StructureError("Number of decomps in child is not multiple of factor {} vs {} "
                                 .format(child_node.dim_decomps, self._factor))

        dim_decomps = child_node.dim_decomps // self._factor
        child = tf.reshape(child, (child_node.dim_scope, dim_decomps, self._factor, -1, child_node.dim_nodes))

        return tf.reshape(tf.transpose(child, [0, 1, 3, 4, 2]), [1, dim_decomps, -1, self.dim_nodes])

    @utils.docinherit(OpNode)
    @utils.lru_cache
    def _compute_log_mpe_value(self, w_tensor, latent_indicators_tensor, *value_tensors):
        raise self._compute_log_value(*value_tensors)

    @utils.docinherit(OpNode)
    @utils.lru_cache
    def _compute_log_mpe_path(self, counts, *input_tensors):
        child = self.values[0].node
        counts = tf.reshape(counts, (self.dim_decomps, -1, child.dim_nodes, self._factor))
        return tf.reshape(tf.transpose(counts, [0, 3, 1, 2]),
                          (child.dim_scope, child.dim_decomps, -1, child.dim_nodes)),

    @utils.docinherit(OpNode)
    def _compute_scope(self, *value_scopes):
        raise NotImplementedError()

    @utils.docinherit(OpNode)
    def _compute_valid(self, *value_scopes):
        # If already invalid, return None
        raise NotImplementedError()

    @property
    @utils.docinherit(OpNode)
    def _const_out_size(self):
        return True

