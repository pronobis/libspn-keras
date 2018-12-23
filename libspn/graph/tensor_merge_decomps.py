import abc
from libspn.graph.node import OpNode, Input, TensorNode
from libspn.graph.ivs import IVs
from libspn.graph.weights import Weights, TensorWeights
from libspn.inference.type import InferenceType
from libspn.exceptions import StructureError
import libspn.utils as utils
from libspn.log import get_logger
from itertools import chain
import functools
import tensorflow as tf
import numpy as np


@utils.register_serializable
class TensorMergeDecomps(TensorNode):

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
        weights (input_like): Input providing weights node to this sum node.
            See :meth:`~libspn.Input.as_input` for possible values. If set
            to ``None``, the input is disconnected.
        ivs (input_like): Input providing IVs of an explicit latent variable
            associated with this sum node. See :meth:`~libspn.Input.as_input`
            for possible values. If set to ``None``, the input is disconnected.
        name (str): Name of the node.

    Attributes:
        inference_type(InferenceType): Flag indicating the preferred inference
                                       type for this node that will be used
                                       during value calculation and learning.
                                       Can be changed at any time and will be
                                       used during the next inference/learning
                                       op generation.
    """
    def __init__(self, *values, inference_type=InferenceType.MARGINAL,
                 name="TensorMergeDecomps", input_format="SDBN", output_format="SDBN"):
        super().__init__(
            inference_type=inference_type, name=name, input_format=input_format,
            output_format=output_format, num_decomps=1, num_scopes=1)
        self.set_values(*values)

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
        return child.dim_nodes * child.dim_decomps

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
    def _compute_value(self, w_tensor, ivs_tensor, *input_tensors, dropconnect_keep_prob=None):
        # Reduce over last axis
        raise NotImplementedError()

    @utils.docinherit(OpNode)
    @utils.lru_cache
    def _compute_log_value(self, *value_tensors, with_ivs=True):
        child = value_tensors[0]
        return tf.reshape(tf.transpose(child, [0, 2, 3, 1]), [1, 1, -1, self.dim_nodes])

    @utils.docinherit(OpNode)
    def _compute_mpe_value(self, w_tensor, ivs_tensor, *input_tensors, dropconnect_keep_prob=None):
        raise NotImplementedError()

    @utils.docinherit(OpNode)
    @utils.lru_cache
    def _compute_log_mpe_value(self, w_tensor, ivs_tensor, *value_tensors, with_ivs=True,
                               dropconnect_keep_prob=None):
        raise self._compute_log_value(*value_tensors, with_ivs=with_ivs)

    @utils.lru_cache
    def _compute_mpe_path_common(
            self, reducible_tensor, counts, w_tensor, ivs_tensor, *input_tensors,
            log=True, sample=False, sample_prob=None, sum_weight_grads=False):
        """Common operations for computing the MPE path.

        Args:
            reducible_tensor (Tensor): A (weighted) ``Tensor`` of (log-)values of this node.
            counts (Tensor): A ``Tensor`` that contains the accumulated counts of the parents
                             of this node.
            w_tensor (Tensor):  A ``Tensor`` containing the (log-)value of the weights.
            ivs_tensor (Tensor): A ``Tensor`` containing the (log-)value of the IVs.
            input_tensors (list): A list of ``Tensor``s with outputs of the child nodes.
            log (bool): Whether the computation is in log-space or not
            sample (bool): Whether to sample the 'winner' of the max or not
            sample_prob (Tensor): A scalar ``Tensor`` indicating the probability of drawing
                a sample. If a sample is drawn, the probability for each index is given by the
                (log-)normalized probability as given by ``reducible_tensor``.
        Returns:
            A ``list`` of ``tuple``s [(MPE counts, input tensor), ...] where the first corresponds
            to the Weights of this node, the second corresponds to the IVs and the remaining
            tuples correspond to the nodes in ``self._values``.
        """
        raise NotImplementedError()

    @utils.docinherit(OpNode)
    @utils.lru_cache
    def _compute_mpe_path(self, counts, w_tensor, ivs_tensor, *value_tensors,
                          use_unweighted=False, with_ivs=True, add_random=None,
                          sample=False, sample_prob=None, dropconnect_keep_prob=None):
        child_node = self.values[0]
        counts = tf.reshape(counts, (-1, child_node.dim_nodes, child_node.dim_decomps))
        return tf.expand_dims(tf.transpose(counts, [2, 0, 1]), 0)

    @utils.docinherit(OpNode)
    @utils.lru_cache
    def _compute_log_mpe_path(self, counts, w_tensor, ivs_tensor, *input_tensors,
                              use_unweighted=False, with_ivs=True, add_random=None,
                              sum_weight_grads=False, sample=False, sample_prob=None,
                              dropconnect_keep_prob=None):
        raise NotImplementedError()

    @utils.docinherit(OpNode)
    def _compute_scope(self, weight_scopes, ivs_scopes, *value_scopes):
        raise NotImplementedError()

    @utils.docinherit(OpNode)
    def _compute_valid(self, weight_scopes, ivs_scopes, *value_scopes):
        # If already invalid, return None
        raise NotImplementedError()

    @property
    @utils.docinherit(OpNode)
    def _const_out_size(self):
        return True

