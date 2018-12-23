import abc
from libspn.graph.node import OpNode, Input, TensorNode, Node
from libspn.graph.ivs import IVs
from libspn.graph.weights import Weights, TensorWeights
from libspn.inference.type import InferenceType
from libspn.exceptions import StructureError
import libspn.utils as utils
from libspn.log import get_logger
from itertools import chain
import tensorflow as tf


@utils.register_serializable
class TensorSum(TensorNode):

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

    def _compute_out_size(self, *input_out_sizes):
        pass

    def __init__(self, *values, num_sums, weights=None, ivs=None, sum_sizes=None,
                 inference_type=InferenceType.MARGINAL, masked=False, sample_prob=None,
                 dropconnect_keep_prob=None, name="TensorSum", input_format="SDBN",
                 output_format="SDBN"):
        super().__init__(inference_type=inference_type, name=name,
                         input_format=input_format, output_format=output_format)
        self.set_values(*values)
        self.set_weights(weights)
        self.set_ivs(ivs)

        # Set whether this instance is masked (e.g. SumsLayer)
        self._masked = masked

        # Set the sampling probability and sampling type
        self._sample_prob = sample_prob

        # Set dropconnect and dropout probabilities
        self._dropconnect_keep_prob = dropconnect_keep_prob

        self._num_sums = num_sums

    @property
    def dim_nodes(self):
        return self._num_sums

    @property
    def dim_decomps(self):
        return self.values[0].node.dim_decomps

    @property
    def dim_scope(self):
        return self.values[0].node.dim_scope

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
        return (self._weights, self._ivs) + self._values

    @property
    def dropconnect_keep_prob(self):
        return self._dropconnect_keep_prob

    def set_dropconnect_keep_prob(self, p):
        self._dropconnect_keep_prob = p

    @property
    def weights(self):
        """Input: Weights input."""
        return self._weights

    def set_weights(self, weights=None):
        """Set the weights input.

        Args:
            weights (input_like): Input providing weights node to this sum node.
                See :meth:`~libspn.Input.as_input` for possible values. If set
                to ``None``, the input is disconnected.
        """
        weights, = self._parse_inputs(weights)
        if weights and not isinstance(weights.node, (Weights, TensorWeights)):
            raise StructureError("%s is not Weights" % weights.node)
        self._weights = weights

    @property
    def ivs(self):
        """Input: IVs input."""
        return self._ivs

    def set_ivs(self, ivs=None):
        """Set the IVs input.

        ivs (input_like): Input providing IVs of an explicit latent variable
            associated with this sum node. See :meth:`~libspn.Input.as_input`
            for possible values. If set to ``None``, the input is disconnected.
        """
        self._ivs, = self._parse_inputs(ivs)

    @property
    def values(self):
        """list of Input: List of value inputs."""
        return self._values

    @property
    def num_sums(self):
        """int: The number of sums modeled by this node. """
        return self._num_sums

    @property
    def sum_sizes(self):
        """list of int: A list of the sum sizes. """
        return self._sum_sizes

    def set_values(self, *values):
        """Set the inputs providing input values to this node. If no arguments
        are given, all existing value inputs get disconnected.

        Args:
            *values (input_like): Inputs providing input values to this node.
                See :meth:`~libspn.Input.as_input` for possible values.
        """
        if len(values) > 1:
            raise NotImplementedError("Can only deal with single inputs")
        print(values[0])
        if isinstance(values[0], Input) and not isinstance(values[0].node, TensorNode):
            raise NotImplementedError("Inputs must be TensorNode")
        elif isinstance(values[0], Node) and not isinstance(values[0], TensorNode):
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

    def generate_weights(self, init_value=1, trainable=True, input_sizes=None,
                         log=False, name=None):
        """Generate a weights node matching this sum node and connect it to
        this sum.

        The function calculates the number of weights based on the number
        of input values of this sum. Therefore, weights should be generated
        once all inputs are added to this node.

        Args:
            init_value: Initial value of the weights. For possible values, see
                :meth:`~libspn.utils.broadcast_value`.
            trainable (bool): See :class:`~libspn.Weights`.
            input_sizes (list of int): Pre-computed sizes of each input of
                this node.  If given, this function will not traverse the graph
                to discover the sizes.
            log (bool): If "True", the weights are represented in log space.
            name (str): Name of the weighs node. If ``None`` use the name of the
                        sum + ``_Weights``.

        Return:
            Weights: Generated weights node.
        """
        if not self._values:
            raise StructureError("%s is missing input values" % self)
        if name is None:
            name = self._name + "_Weights"

        # Count all input values
        num_inputs = self._values[0].node.dim_nodes
        # Generate weights
        weights = TensorWeights(
            num_inputs=num_inputs, num_outputs=self._num_sums, num_decomps=self.dim_decomps,
            num_scopes=self.dim_scope, name=name, trainable=trainable, in_logspace=log)
        self.set_weights(weights)
        return weights

    def generate_ivs(self, feed=None, name=None):
        """Generate an IVs node matching this sum node and connect it to
        this sum.

        IVs should be generated once all inputs are added to this node,
        otherwise the number of IVs will be incorrect.

        Args:
            feed (Tensor): See :class:`~libspn.IVs`.
            name (str): Name of the IVs node. If ``None`` use the name of the
                        sum + ``_IVs``.

        Return:
            IVs: Generated IVs node.
        """
        if self.dim_scope != 1 or self.dim_decomps != 1:
            raise NotImplementedError("{}: scope dim and decomposition dim must be 1 to "
                                      "apply IVs".format(self))
        ivs_node = IVs(feed=feed, num_vars=1, num_vals=self.values[0].node.dim_nodes, name=name)
        self.set_ivs(ivs_node)
        return ivs_node

    @utils.docinherit(OpNode)
    @utils.lru_cache
    def _compute_value(self, w_tensor, ivs_tensor, *input_tensors, dropconnect_keep_prob=None):
        # Reduce over last axis
        raise NotImplementedError()

    @utils.docinherit(OpNode)
    @utils.lru_cache
    def _compute_log_value(self, w_tensor, ivs_tensor, *value_tensors, dropconnect_keep_prob=None,
                           with_ivs=True):
        child = value_tensors[0]
        return utils.logmatmul(self._compute_apply_ivs(child, ivs_tensor), w_tensor)

    @utils.docinherit(OpNode)
    def _compute_mpe_value(self, w_tensor, ivs_tensor, *input_tensors, dropconnect_keep_prob=None):
        raise NotImplementedError()

    @utils.docinherit(OpNode)
    @utils.lru_cache
    def _compute_log_mpe_value(self, w_tensor, ivs_tensor, *value_tensors, with_ivs=True,
                               dropconnect_keep_prob=None):
        child = value_tensors[0]
        if len(value_tensors) > 1:
            raise NotImplementedError("Can only deal with a single input")
        # child: [scope, decomp, batch, nodes_in]
        # weights: [scope, decomp, nodes_in, nodes_out]
        return tf.reduce_max(self._compute_weighted(child, w_tensor, ivs_tensor), axis=3)

    @utils.lru_cache
    def _compute_apply_ivs(self, child, ivs_tensor):
        if ivs_tensor is not None:
            if self.dim_scope != 1 or self.dim_decomps != 1:
                raise NotImplementedError("{}: scope dim and decomposition dim must be 1 to "
                                          "apply IVs".format(self))
            # [batch, nodes_input]
            return child + tf.reshape(ivs_tensor, (1, 1, -1, self.values[0].node.dim_nodes))
        return child

    @utils.lru_cache
    def _compute_weighted(self, child, w_tensor, ivs_tensor):
        child = self._compute_apply_ivs(child, ivs_tensor)
        return tf.expand_dims(child, 4) + tf.expand_dims(w_tensor, 2)

    def _compute_mpe_path(self, counts, w_tensor, ivs_tensor, *value_tensors,
                          use_unweighted=False, with_ivs=True, add_random=None,
                          sample=False, sample_prob=None, dropconnect_keep_prob=None):
        # counts: [scope, decomp, batch, nodes_out]
        # ret: [scope, decomp, batch, nodes_in]
        # winning_indices: [scope, decomp, batch, nodes_out]
        child = value_tensors[0]
        child_node = self.values[0].node
        if use_unweighted:
            winning_indices = \
                tf.tile(tf.expand_dims(
                    tf.argmax(self._compute_apply_ivs(child, ivs_tensor), axis=-1), -1
                ), (1, 1, 1, child_node.dim_nodes))
        else:
            weighted = self._compute_weighted(child, w_tensor, ivs_tensor)
            winning_indices = tf.argmax(weighted, axis=3)

        winning_indices_one_hot = tf.one_hot(winning_indices, depth=child_node.dim_nodes, axis=-1)
        paths = tf.expand_dims(counts, axis=3) * winning_indices_one_hot
        input_counts = tf.reduce_sum(paths, axis=3)
        weight_counts = tf.reduce_sum(paths, axis=0)
        return weight_counts, input_counts, input_counts

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
    def _compute_log_mpe_path(self, counts, w_tensor, ivs_tensor, *value_tensors,
                              use_unweighted=False, with_ivs=True, add_random=None,
                              sum_weight_grads=False, sample=False, sample_prob=None,
                              dropconnect_keep_prob=None):
        child = value_tensors[0]
        child_node = self.values[0].node
        if use_unweighted:
            winning_indices = \
                tf.tile(tf.expand_dims(
                    tf.argmax(self._compute_apply_ivs(child, ivs_tensor), axis=-1), -1
                ), (1, 1, 1, child_node.dim_nodes))
        else:
            # print(child, w_tensor, ivs_tensor)
            weighted = self._compute_weighted(child, w_tensor, ivs_tensor)
            winning_indices = tf.argmax(weighted, axis=3)

        winning_indices_one_hot = tf.one_hot(winning_indices, depth=child_node.dim_nodes, axis=3)
        paths = tf.expand_dims(counts, axis=3) * winning_indices_one_hot
        input_counts = tf.reduce_sum(paths, axis=4)
        weight_counts = tf.reduce_sum(paths, axis=0)
        ivs_counts = tf.reshape(input_counts, (-1, self._num_sums))
        return weight_counts, ivs_counts, input_counts

    def _get_flat_value_scopes(self, weight_scopes, ivs_scopes, *value_scopes):
        """Get a flat representation of the value scopes per sum.

        Args:
            weight_scopes (list): A list of ``Scope``s corresponding to the weights.
            ivs_scopes (list): A list of ``Scope``s corresponding to the IVs.
            value_scopes (tuple): A ``tuple`` of ``list``s of ``Scope``s corresponding to the
                                  scope lists of the children of this node.

        Returns:
            A tuple of flat value scopes corresponding to this node's output. The IVs scopes and
            the value scopes.
        """
        if not self._values:
            raise StructureError("%s is missing input values" % self)
        _, ivs_scopes, *value_scopes = self._gather_input_scopes(
            weight_scopes, ivs_scopes, *value_scopes)
        return list(chain.from_iterable(value_scopes)), ivs_scopes, value_scopes

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

