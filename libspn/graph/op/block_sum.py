from libspn.graph.node import OpNode, Input, BlockNode, Node
from libspn.graph.leaf.indicator import IndicatorLeaf
from libspn.graph.weights import Weights, BlockWeights
from libspn.inference.type import InferenceType
from libspn.exceptions import StructureError
import libspn.utils as utils
from tensorflow.python.framework import tensor_util
from libspn.log import get_logger
from itertools import chain
import tensorflow as tf


@utils.register_serializable
class BlockSum(BlockNode):

    """
    This node represents sums computed in blocks. Each block corresponds to a set of nodes for
    a specific (i) scope and (ii) decomposition. Apart from the axis containing nodes within the
    block, there's an axis for (i) batch element, (ii) the scope and (iii) the decomposition in
    the internal tensor representation.

    Args:
        child (input_like): Child for this node.
            See :meth:`~libspn.Input.as_input` for possible values.
        num_sums_per_block (int): Number of sums modeled per blobck.
        weights (input_like): Input providing weights node to this sum node.
            See :meth:`~libspn.Input.as_input` for possible values. If set
            to ``None``, the input is disconnected.
        sample_prob (float): Probability for sampling on MPE path computation.
        name (str): Name of the node.

    Attributes:
        inference_type(InferenceType): Flag indicating the preferred inference
                                       type for this node that will be used
                                       during value calculation and learning.
                                       Can be changed at any time and will be
                                       used during the next inference/learning
                                       op generation.
    """

    def __init__(self, child, num_sums_per_block, weights=None, latent_indicators=None,
                 inference_type=InferenceType.MARGINAL, sample_prob=None,
                 name="BlockSum"):
        super().__init__(inference_type=inference_type, name=name)
        self.set_values(child)
        self.set_weights(weights)
        self.set_latent_indicator(latent_indicators)

        # Set the sampling probability and sampling type
        self._sample_prob = sample_prob

        self._num_sums = num_sums_per_block

    @property
    def dim_nodes(self):
        """Dim size nodes"""
        return self._num_sums

    @property
    def dim_decomps(self):
        """Dim size decompositions"""
        return self.child.dim_decomps

    @property
    def dim_scope(self):
        """Dim size scopes"""
        return self.child.dim_scope

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
        return (self._weights, self._latent_indicator) + self._values

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
        if weights and not isinstance(weights.node, (Weights, BlockWeights)):
            raise StructureError("%s is not Weights" % weights.node)
        self._weights = weights

    @property
    def latent_indicators(self):
        """Input: IndicatorLeaf input."""
        return self._latent_indicator

    def set_latent_indicator(self, latent_indicator=None):
        """Set the latent indicator input.

        latent_indicator (input_like): Input providing indicators of an explicit latent variable
            associated with this sum node. See :meth:`~libspn.Input.as_input`
            for possible values. If set to ``None``, the input is disconnected.
        """
        self._latent_indicator, = self._parse_inputs(latent_indicator)

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
        if isinstance(values[0], Input) and not isinstance(values[0].node, BlockNode):
            raise NotImplementedError("Inputs must be TensorNode")
        elif isinstance(values[0], Node) and not isinstance(values[0], BlockNode):
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

    def generate_weights(self, initializer=tf.initializers.constant(1.0),
                         trainable=True, log=False, name=None, input_sizes=None):
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
        num_inputs = self.child.dim_nodes
        # Generate weights
        weights = BlockWeights(
            num_inputs=num_inputs, num_outputs=self._num_sums, num_decomps=self.dim_decomps,
            num_scopes=self.dim_scope, name=name, trainable=trainable, in_logspace=log,
            initializer=initializer)
        self.set_weights(weights)
        return weights

    def generate_latent_indicators(self, feed=None, name=None):
        """Generate an IndicatorLeaf node matching this sum node and connect it to
        this sum.

        IndicatorLeaf should be generated once all inputs are added to this node,
        otherwise the number of IndicatorLeaf variables will be incorrect.

        Args:
            feed (Tensor): See :class:`~libspn.IndicatorLeaf`.
            name (str): Name of the latent indicator node. If ``None`` use the name of the
                sum + ``_LatentIndicator``.

        Return:
            Latent indicators: Generated IndicatorLeaf node.
        """
        if self.dim_scope != 1 or self.dim_decomps != 1:
            raise NotImplementedError(
                "{}: scope dim and decomposition dim must be 1 to apply latent indicators, got "
                "{} and  {} instead".format(self, self.dim_scope, self.dim_decomps))
        latent_indicator = IndicatorLeaf(
            feed=feed, num_vars=1, num_vals=self.child.dim_nodes,
            name=name or self._name + "_LatentIndicator")
        self.set_latent_indicator(latent_indicator)
        return latent_indicator

    @utils.docinherit(OpNode)
    @utils.lru_cache
    def _compute_log_value(
            self, w_log_prob, latent_indicator_log_prob, child_log_prob):
        return utils.logmatmul(
            self._compute_apply_latent_indicators(child_log_prob, latent_indicator_log_prob), w_log_prob)

    @utils.docinherit(OpNode)
    @utils.lru_cache
    def _compute_log_mpe_value(self, w_tensor, latent_indicator_tensor, child_log_prob):
        return tf.reduce_max(
            self._compute_weighted(child_log_prob, w_tensor, latent_indicator_tensor), axis=3)

    @utils.lru_cache
    def _compute_apply_latent_indicators(self, child_log_prob, latent_indicator_log_prob):
        if latent_indicator_log_prob is not None:
            if self.dim_scope != 1 or self.dim_decomps != 1:
                raise NotImplementedError("{}: scope dim and decomposition dim must be 1 to "
                                          "apply ltent indicators".format(self))
            # [batch, nodes_input]
            return child_log_prob + tf.reshape(latent_indicator_log_prob, (1, 1, -1, self.child.dim_nodes))
        return child_log_prob

    @utils.lru_cache
    def _compute_weighted(self, child, w_tensor, latent_indicator_tensor):
        child = self._compute_apply_latent_indicators(child, latent_indicator_tensor)
        return tf.expand_dims(child, axis=4) + tf.expand_dims(w_tensor, axis=2)

    @utils.docinherit(OpNode)
    @utils.lru_cache
    def _compute_log_mpe_path(self, counts, w_tensor, latent_indicator_tensor, child_log_prob,
                              use_unweighted=False, sum_weight_grads=False, sample=False,
                              sample_prob=None):
        if use_unweighted:
            winning_indices = utils.argmax_breaking_ties(
                self._compute_apply_latent_indicators(child_log_prob, latent_indicator_tensor),
                num_samples=self.child.dim_nodes, keepdims=True)
        else:
            weighted = self._compute_weighted(child_log_prob, w_tensor, latent_indicator_tensor)
            winning_indices = utils.argmax_breaking_ties(weighted, axis=-2)

        # Paths has shape [scope, decomp, batch, node_out, node_in]
        paths = utils.scatter_values_nd(counts, winning_indices, depth=self.child.dim_nodes)
        # Get child counts by summing over out nodes
        child_counts = tf.reduce_sum(paths, axis=3)
        # Get weight counts by summing over batch and transposing last two dims
        weight_counts = tf.transpose(tf.reduce_sum(paths, axis=self._batch_axis), (0, 1, 3, 2))
        latent_counts = tf.reshape(child_counts, (-1, self._num_sums))
        return weight_counts, latent_counts, child_counts

    def _get_flat_value_scopes(self, weight_scopes, latent_indicator_scopes, *value_scopes):
        """Get a flat representation of the value scopes per sum.

        Args:
            weight_scopes (list): A list of ``Scope``s corresponding to the weights.
            latent_indicator_scopes (list): A list of ``Scope``s corresponding to the latent indicators.
            value_scopes (tuple): A ``tuple`` of ``list``s of ``Scope``s corresponding to the
                                  scope lists of the children of this node.

        Returns:
            A tuple of flat value scopes corresponding to this node's output. The latent indicator
            scopes and the value scopes.
        """
        if not self._values:
            raise StructureError("%s is missing input values" % self)
        _, latent_indicator_scopes, *value_scopes = self._gather_input_scopes(
            weight_scopes, latent_indicator_scopes, *value_scopes)
        return list(chain.from_iterable(value_scopes)), latent_indicator_scopes, value_scopes

    @utils.docinherit(OpNode)
    def _compute_scope(self, weight_scopes, latent_scopes, *value_scopes):
        raise NotImplementedError()

    @utils.docinherit(OpNode)
    def _compute_valid(self, weight_scopes, latent_scopes, *value_scopes):
        # If already invalid, return None
        raise NotImplementedError()

    @property
    @utils.docinherit(OpNode)
    def _const_out_size(self):
        return True

    def _compute_out_size(self, *input_out_sizes):
        pass
