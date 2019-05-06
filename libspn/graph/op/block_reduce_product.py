from libspn.graph.node import OpNode
from libspn.graph.op.block_permute_product import BlockPermuteProduct
import libspn.utils as utils
from libspn.log import get_logger
import tensorflow as tf


@utils.register_serializable
class BlockReduceProduct(BlockPermuteProduct):

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

    @property
    def dim_nodes(self):
        return self.child.dim_nodes

    def _compute_out_size(self, *input_out_sizes):
        pass

    @utils.docinherit(OpNode)
    @utils.lru_cache
    def _compute_log_value(self, child_log_prob):

        # Split in list of tensors which will be added up using outer products
        shape = [self.dim_scope, self._num_factors, self.dim_decomps, -1, self.child.dim_nodes]
        return tf.reduce_sum(tf.reshape(child_log_prob, shape=shape), axis=1)

    @utils.docinherit(OpNode)
    @utils.lru_cache
    def _compute_log_mpe_value(self, child_log_prob):
        return self._compute_log_value(child_log_prob)

    @utils.docinherit(OpNode)
    @utils.lru_cache
    def _compute_log_mpe_path(self, counts, *child_log_prob):
        # In the forward pass, we performed out products in #factors directions. For each direction,
        # we simply need to sum up the counts of the other directions from the parent to obtain the
        # counts for the child. This is a many-to-few operation.
        child = self.child
        counts_reshaped = tf.reshape(
            counts, [self.dim_scope, 1, self.dim_decomps, -1, child.dim_nodes])
        return tf.reshape(tf.tile([1, self.num_factors, self.dim_decomps, -1, child.dim_nodes]))
