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


@utils.register_serializable
class TensorProduct(TensorNode):

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

    @property
    def dim_nodes(self):
        return self.values[0].node.dim_nodes ** self._num_factors

    @property
    def dim_decomps(self):
        return self.values[0].node.dim_decomps

    def _compute_out_size(self, *input_out_sizes):
        pass

    def __init__(self, *values, num_factors, num_decomps=None, num_scopes=None,
                 inference_type=InferenceType.MARGINAL,
                 name="TensorProduct", input_format="SDBN", output_format="SDBN"):
        super().__init__(
            inference_type=inference_type, name=name, input_format=input_format,
            output_format=output_format, num_decomps=num_decomps, num_scopes=num_scopes)
        self.set_values(*values)
        self._num_factors = num_factors
        self._batch_axis = 2
        self._decomp_axis = 1
        self._scope_axis = 0
        self._node_axis = 3

    @property
    def dim_scope(self):
        return self._values[0].node.dim_scope // self._num_factors

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

    def set_values(self, *values):
        """Set the inputs providing input values to this node. If no arguments
        are given, all existing value inputs get disconnected.

        Args:
            *values (input_like): Inputs providing input values to this node.
                See :meth:`~libspn.Input.as_input` for possible values.
        """
        if len(values) > 1:
            raise NotImplementedError("Can only deal with single inputs")
        if not isinstance(values[0], TensorNode):
            raise NotImplementedError("Inputs must be TensorNode")
        self._values = self._parse_inputs(*values)

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
    def _compute_log_value(self, *value_tensors, dropconnect_keep_prob=None,
                           with_ivs=True):
        child = value_tensors[0]
        shape = [self.dim_scope, self._num_factors, self.dim_decomps, -1,
                 self.values[0].node.dim_nodes]
        log_prob_per_out_scope = tf.split(
            tf.reshape(child, shape=shape), axis=1, num_or_size_splits=self._num_factors)

        def log_outer_product(a, b):
            a_last_dim = a.shape[-1].value
            b_last_dim = b.shape[-1].value
            a_shape = [self.dim_scope, self.dim_decomps, -1, a_last_dim, 1]
            b_shape = [self.dim_scope, self.dim_decomps, -1, 1, b_last_dim]
            out_shape = [self.dim_scope, self.dim_decomps, -1, a_last_dim * b_last_dim]
            return tf.reshape(tf.reshape(a, a_shape) + tf.reshape(b, b_shape), out_shape)

        return functools.reduce(log_outer_product, log_prob_per_out_scope)

    @utils.docinherit(OpNode)
    def _compute_mpe_value(self, w_tensor, ivs_tensor, *input_tensors, dropconnect_keep_prob=None):
        raise NotImplementedError()

    @utils.docinherit(OpNode)
    @utils.lru_cache
    def _compute_log_mpe_value(self, w_tensor, ivs_tensor, *value_tensors, with_ivs=True,
                               dropconnect_keep_prob=None):
        raise NotImplementedError()

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
        raise NotImplementedError()

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

