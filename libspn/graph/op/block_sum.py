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

    def __init__(self, child, num_sums, weights=None, latent_ivs=None,
                 inference_type=InferenceType.MARGINAL, masked=False, sample_prob=None,
                 dropconnect_keep_prob=None, name="TensorSum", input_format="SDBN",
                 output_format="SDBN"):
        super().__init__(inference_type=inference_type, name=name,
                         input_format=input_format, output_format=output_format)
        self.set_values(child)
        self.set_weights(weights)
        self.set_latent_indicator(latent_ivs)

        # Set the sampling probability and sampling type
        self._sample_prob = sample_prob

        # Set dropconnect and dropout probabilities
        self._dropconnect_keep_prob = dropconnect_keep_prob

        self._num_sums = num_sums

    @property
    def child(self):
        return self._values[0].node
    
    @property
    def dim_nodes(self):
        return self._num_sums

    @property
    def dim_decomps(self):
        return self.child.dim_decomps

    @property
    def dim_scope(self):
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
        if weights and not isinstance(weights.node, (Weights, BlockWeights)):
            raise StructureError("%s is not Weights" % weights.node)
        self._weights = weights

    @utils.lru_cache
    def _create_dropconnect_mask(
            self, keep_prob, to_be_masked, enforce_one_axis=-1, name="DropconnectMask"):
        with tf.name_scope(name):
            shape = tf.shape(to_be_masked)
            drop_mask = tf.random_uniform(shape=shape, minval=0.0, maxval=1.0)
            # To ensure numerical stability and the opportunity to always learn something,
            # we enforce at least a single 'True' value along the last axis (sum axis) by comparing
            # the randomly drawn floats with their minimum and setting True in case of equality.
            # return tf.less(mask, keep_prob)
            if self._masked:
                rank = tf.size(shape)
                size_mask = tf.reshape(
                    self._build_mask(),
                    tf.concat([tf.ones(rank - 2, dtype=tf.int32),
                               [self._num_sums, self._max_sum_size]], axis=0))
                size_mask = tf.tile(size_mask, tf.concat([shape[:rank - 2], [1, 1]], axis=0))
                drop_mask = tf.where(
                    size_mask, drop_mask, tf.ones_like(size_mask, dtype=tf.float32) * 1e20)

            if enforce_one_axis is None:
                return tf.less(drop_mask, keep_prob)
            mask_min = tf.reduce_min(drop_mask, axis=enforce_one_axis, keepdims=True)
            out = tf.logical_or(tf.equal(drop_mask, mask_min), tf.less(drop_mask, keep_prob))
            return out

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
            raise NotImplementedError("{}: scope dim and decomposition dim must be 1 to "
                                      "apply latent indicators".format(self))
        latent_indicator = IndicatorLeaf(
            feed=feed, num_vars=1, num_vals=self.child.dim_nodes,
            name=name or self._name + "_LatentIndicator")
        self.set_latent_indicator(latent_indicator)
        return latent_indicator

    @utils.docinherit(OpNode)
    @utils.lru_cache
    def _compute_log_value(self, w_log_prob, latent_indicator_log_prob, child_log_prob,
                           dropconnect_keep_prob=None, with_ivs=True):
        if latent_indicator_log_prob is None and dropconnect_keep_prob is not None and \
                tensor_util.constant_value(dropconnect_keep_prob) != 1.0:
            mask = self._create_dropconnect_mask(dropconnect_keep_prob, to_be_masked=child_log_prob)
            child_log_prob += tf.log(mask)

        return utils.logmatmul(
            self._compute_apply_ivs(child_log_prob, latent_indicator_log_prob), w_log_prob)

    @utils.docinherit(OpNode)
    @utils.lru_cache
    def _compute_log_mpe_value(self, w_tensor, ivs_tensor, child_log_prob, with_ivs=True,
                               dropconnect_keep_prob=None):
        return tf.reduce_max(self._compute_weighted(child_log_prob, w_tensor, ivs_tensor), axis=3)

    @utils.lru_cache
    def _compute_apply_ivs(self, child, ivs_tensor):
        if ivs_tensor is not None:
            if self.dim_scope != 1 or self.dim_decomps != 1:
                raise NotImplementedError("{}: scope dim and decomposition dim must be 1 to "
                                          "apply ltent indicators".format(self))
            # [batch, nodes_input]
            return child + tf.reshape(ivs_tensor, (1, 1, -1, self.child.dim_nodes))
        return child

    @utils.lru_cache
    def _compute_weighted(self, child, w_tensor, ivs_tensor):
        child = self._compute_apply_ivs(child, ivs_tensor)
        return tf.expand_dims(child, axis=4) + tf.expand_dims(w_tensor, axis=2)

    @utils.docinherit(OpNode)
    @utils.lru_cache
    def _compute_log_mpe_path(self, counts, w_tensor, ivs_tensor, child_log_prob,
                              use_unweighted=False, add_random=None, sum_weight_grads=False,
                              sample=False, sample_prob=None, dropconnect_keep_prob=None):
        if use_unweighted:
            winning_indices = utils.argmax_breaking_ties(
                self._compute_apply_ivs(child_log_prob, ivs_tensor),
                num_samples=self.child.dim_nodes, keepdims=True)
        else:
            weighted = self._compute_weighted(child_log_prob, w_tensor, ivs_tensor)
            winning_indices = utils.argmax_breaking_ties(weighted, axis=-2)

        paths = utils.scatter_values_nd(counts, winning_indices, depth=self.child.dim_nodes)
        input_counts = tf.reduce_sum(paths, axis=3)
        weight_counts = tf.transpose(tf.reduce_sum(paths, axis=self._batch_axis), (0, 1, 3, 2))
        ivs_counts = tf.reshape(input_counts, (-1, self._num_sums))
        return weight_counts, ivs_counts, input_counts

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

