from libspn.graph.node import OpNode, BlockNode
from libspn.graph.leaf.indicator import IndicatorLeaf
from libspn.inference.type import InferenceType
from libspn.exceptions import StructureError
import libspn.utils as utils
from tensorflow.python.ops.array_grad import _GatherV2Grad
import tensorflow as tf
import numpy as np


@utils.register_serializable
class BlockRandomDecompositions(BlockNode):

    """A node that creates random decompositions of its leaf inputs by generating an arbitrary 
    number of random orderings the inputs randomly and having singleton scopes in the output.

    Args:
        child (input_like): Inputs providing input values to this node.
            See :meth:`~libspn.Input.as_input` for possible values.
        num_decomps (int): Number of random decompositions to generate
        name (str): Name of the node.

    Attributes:
        inference_type(InferenceType): Flag indicating the preferred inference
                                       type for this node that will be used
                                       during value calculation and learning.
                                       Can be changed at any time and will be
                                       used during the next inference/learning
                                       op generation.
    """

    def __init__(self, child, num_decomps, inference_type=InferenceType.MARGINAL,
                 name="BlockRandomDecompositions"):
        super().__init__(inference_type=inference_type, name=name, num_decomps=num_decomps)
        self.set_values(child)
        self._num_decomps = num_decomps
        self._permutations = None

    def generate_permutations(self, factors):
        if not factors:
            raise StructureError("{}: factors needs to be a non-empty sequence.")
        num_input_scopes = self._get_num_input_scopes()
        factor_cumprod = np.cumprod(factors)
        factor_prod = factor_cumprod[-1]
        if factor_prod < num_input_scopes:
            raise StructureError("{}: not enough factors to cover all variables ({} vs. {})."
                                 .format(self, factor_prod, num_input_scopes))
        for i, fc in enumerate(factor_cumprod[:-1]):
            if fc >= num_input_scopes:
                raise StructureError(
                    "{}: too many factors, taking out the bottom {} products still "
                    "results in {} factors while {} are needed.".format(
                        self, len(factors) - i - 1, fc, num_input_scopes))

        # Now we generate the random index permutations
        perms = [np.random.permutation(num_input_scopes).astype(int).tolist()
                 for _ in range(self._num_decomps)]

        num_m1 = factor_prod - num_input_scopes
        if num_m1 > 0:
            # e.g. num_m1 == 2 and factor_prod = 32. Then rate_m1 is 16, so once every 16 values
            # we should leave a variable slot empty
            rate_m1 = int(np.floor(factor_prod / num_m1))

            for p in perms:
                for i in range(num_m1):
                    p.insert(i * rate_m1, -1)
        self._permutations = perms = np.asarray(perms)
        return perms

    def set_values(self, *values):
        self._values = self._parse_inputs(*values)

    def _get_num_input_scopes(self):
        return self.child.num_vars

    @property
    def dim_nodes(self):
        if not self._values:
            raise StructureError("{}: cannot get num outputs per decomp and scope since this "
                                 "node has no children.".format(self))
        child = self._values[0].node
        try:
            return child.num_components
        except AttributeError as e:
            return child.num_vals

    def _compute_out_size(self, *input_out_sizes):
        pass

    @property
    def dim_scope(self):
        return self._permutations.shape[1]

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
        if self._permutations is None:
            raise StructureError("First need to determine permutations")
        # [batch, scope, node]
        child, = value_tensors
        dim_scope_in = self.child.num_vars
        dim_nodes_in = self.child.num_vals if isinstance(
            self.child, IndicatorLeaf) else self.child.num_components
        zero_padded = tf.concat(
            [tf.zeros([1, tf.shape(child)[0], dim_nodes_in]),
             tf.transpose(
                 tf.reshape(child, [-1, dim_scope_in, dim_nodes_in]), (1, 0, 2))], axis=0)
        gather_indices = self._permutations + 1
        permuted = self._gather_op = tf.gather(zero_padded, np.transpose(gather_indices))
        self._zero_padded_shape = tf.shape(zero_padded)

        return permuted

    @utils.docinherit(OpNode)
    @utils.lru_cache
    def _compute_log_mpe_value(self, *value_tensors):
        raise self._compute_log_value(*value_tensors)

    @utils.docinherit(OpNode)
    @utils.lru_cache
    def _compute_log_mpe_path(
            self, counts, *input_tensors, use_unweighted=False, sum_weight_grads=False,
            sample=False, sample_prob=None):
        # counts is shape [scope, decomps, batch, nodes]
        # will have to be transformed to [batch, scope * nodes]
        grad = tf.reshape(_GatherV2Grad(self._gather_op._op, counts)[0], self._zero_padded_shape)
        dim_scope_in = self.child.num_vars
        _, counts_in = tf.split(grad, num_or_size_splits=[1, dim_scope_in], axis=0)
        return tf.reshape(tf.transpose(counts_in, (1, 0, 2)),
                          (-1, self.child._compute_out_size())),

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
