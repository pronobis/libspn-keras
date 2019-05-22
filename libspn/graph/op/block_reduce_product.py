from libspn.inference.type import InferenceType
from libspn.graph.node import OpNode
from libspn.graph.op.block_permute_product import BlockPermuteProduct
import libspn.utils as utils
import tensorflow as tf


@utils.register_serializable
class BlockReduceProduct(BlockPermuteProduct):

    """
    This node represents products computed in blocks. Each block corresponds to a set of nodes for
    a specific (i) scope and (ii) decomposition. Apart from the axis containing nodes within the
    block, there's an axis for (i) batch element, (ii) the scope and (iii) the decomposition in
    the internal tensor representation.

    'Reduce' products are computed by simply reducing over adjacent scopes. For example, when
    ``num_subsets`` is set to 2 and there are 4 scopes at the child node with 8 nodes each, then
    the output of this layer will have 4 / 2 == 2 scopes and 8 nodes per block.

    Args:
        child (input_like): Child for this node.
            See :meth:`~libspn.Input.as_input` for possible values.
        num_factors (int): Number of factors per product. Corresponds to how many scopes are joined
            in this layer.
        name (str): Name of the node.

    Attributes:
        inference_type(InferenceType): Flag indicating the preferred inference
                                       type for this node that will be used
                                       during value calculation and learning.
                                       Can be changed at any time and will be
                                       used during the next inference/learning
                                       op generation.
    """
    def __init__(self, child, num_factors, num_decomps=None, inference_type=InferenceType.MARGINAL,
                 name="BlockReduceProduct"):
        super().__init__(child=child, num_factors=num_factors,
                         num_decomps=num_decomps, inference_type=inference_type, name=name)

    @property
    def dim_nodes(self):
        """Number of nodes per block """
        return self.child.dim_nodes

    @property
    def dim_scope(self):
        """Number of scopes """
        return self.child.dim_scope // self.num_factors

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
        return (tf.reshape(
            tf.tile(counts_reshaped, [1, self.num_factors, 1, 1, 1]), tf.shape(child_log_prob[0])),)

    def _compute_out_size(self, *input_out_sizes):
        pass