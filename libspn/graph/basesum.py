import abc
from libspn.graph.node import OpNode, Input
from libspn.graph.ivs import IVs
from libspn.graph.weights import Weights
from libspn.graph.scope import Scope
from libspn.inference.type import InferenceType
from libspn import conf
from libspn.exceptions import StructureError
import libspn.utils as utils
from libspn.log import get_logger
from itertools import chain
import tensorflow as tf


@utils.register_serializable
class BaseSum(OpNode, abc.ABC):

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
        reduce_axis (int): The axis over which to perform summing (or max for MPE). in (log-)space
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

    def __init__(self, *values, num_sums, weights=None, ivs=None, sum_sizes=None,
                 inference_type=InferenceType.MARGINAL, batch_axis=0, op_axis=1, reduce_axis=2,
                 name="Sum"):
        super().__init__(inference_type, name)
        self.set_values(*values)
        self.set_weights(weights)
        self.set_ivs(ivs)

        self._reset_sum_sizes(num_sums=num_sums, sum_sizes=sum_sizes)

        self._batch_axis = batch_axis
        self._op_axis = op_axis
        self._reduce_axis = reduce_axis

    def _get_sum_sizes(self, num_sums):
        input_sizes = self.get_input_sizes()
        num_values = sum(input_sizes[2:])  # Skip ivs, weights
        return num_sums * [num_values]

    def serialize(self):
        data = super().serialize()
        data['values'] = [(i.node.name, i.indices) for i in self._values]
        if self._weights:
            data['weights'] = (self._weights.node.name, self._weights.indices)
        if self._ivs:
            data['ivs'] = (self._ivs.node.name, self._ivs.indices)
        data['num_sums'] = self._num_sums
        data['sum_sizes'] = self._sum_sizes
        return data

    def deserialize(self, data):
        super().deserialize(data)
        self.set_values()
        self.set_weights()
        self.set_ivs()
        self._reset_sum_sizes(num_sums=data['num_sums'], sum_sizes=data['sum_sizes'])

    def deserialize_inputs(self, data, nodes_by_name):
        super().deserialize_inputs(data, nodes_by_name)
        self._values = tuple(Input(nodes_by_name[nn], i)
                             for nn, i in data['values'])
        weights = data.get('weights', None)
        if weights:
            self._weights = Input(nodes_by_name[weights[0]], weights[1])
        ivs = data.get('ivs', None)
        if ivs:
            self._ivs = Input(nodes_by_name[ivs[0]], ivs[1])

    @property
    @utils.docinherit(OpNode)
    def inputs(self):
        return (self._weights, self._ivs) + self._values

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
        if weights and not isinstance(weights.node, Weights):
            raise StructureError("%s is not Weights" % weights.node)
        self._weights = weights

    def _reset_sum_sizes(self, num_sums=None, sum_sizes=None):
        self._num_sums = num_sums or self._num_sums
        self._sum_sizes = sum_sizes or self._get_sum_sizes(self._num_sums)
        self._max_sum_size = max(self._sum_sizes)

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
        return self._num_sums

    @property
    def sum_sizes(self):
        return self._sum_sizes

    def set_values(self, *values):
        """Set the inputs providing input values to this node. If no arguments
        are given, all existing value inputs get disconnected.

        Args:
            *values (input_like): Inputs providing input values to this node.
                See :meth:`~libspn.Input.as_input` for possible values.
        """
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
        if input_sizes:
            num_values = sum(input_sizes[2:])  # Skip ivs, weights
        else:
            num_values = max(self._sum_sizes)
        # Generate weights
        weights = Weights(
            init_value=init_value, num_weights=num_values, num_sums=self._num_sums,
            log=log, trainable=trainable, name=name)
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
        if not self._values:
            raise StructureError("%s is missing input values" % self)
        if name is None:
            name = self._name + "_IVs"
        ivs = IVs(feed=feed, num_vars=self._num_sums, num_vals=self._max_sum_size, name=name)
        self.set_ivs(ivs)
        return ivs

    @utils.lru_cache
    def _compute_reducible(
            self, w_tensor, ivs_tensor, *input_tensors, log=True, use_ivs=True, weighted=True):
        """Computes a reducible ``Tensor`` so that reducing it over the last axis can be used for
        marginal inference, MPE inference and MPE path computation.

        Args:
            w_tensor (Tensor): A ``Tensor`` with the value of the weights of shape
                               ``[num_sums, max_sum_size]``
            ivs_tensor (Tensor): A ``Tensor`` with the value of the IVs corresponding to this node
                                 of shape ``[batch, num_sums * max_sum_size]``.
            input_tensors (tuple): A ``tuple`` of ``Tensors``s with the values of the children of
                                   this node.
            log (bool): A ``bool`` marking whether the computation is performed in log space or not.
            use_ivs (bool): Whether to apply the IVs to the reducible values if possible.
            weighted (bool): Whether to apply the weights to the reducible values if possible.

        Returns:
            A ``Tensor`` of shape ``[batch, num_sums, max_sum_size]`` that can be used for computing
            marginal inference, MPE inference, gradients or MPE paths.
        """
        if not self._values:
            raise StructureError("%s is missing input values" % self)
        if not self._weights:
            raise StructureError("%s is missing weights" % self)

        # Set up component-wise Op and zero probability value depending on log-space flag
        zero_prob_val = -float('inf') if log else 0.0
        cwise_op = self.cwise_add if log else self.cwise_mul

        # Prepare tensors for component-wise application of weights and IVs
        w_tensor, ivs_tensor, reducible = self._prepare_component_wise_processing(
            w_tensor, ivs_tensor, *input_tensors, zero_prob_val=zero_prob_val)
        if use_ivs and self._ivs:
            reducible = cwise_op(reducible, ivs_tensor)
        if weighted:
            reducible = cwise_op(reducible, w_tensor)
        return reducible

    @utils.docinherit(OpNode)
    @utils.lru_cache
    def _compute_out_size(self, *input_out_sizes):
        return self._num_sums

    @utils.docinherit(OpNode)
    @utils.lru_cache
    def _compute_value(self, w_tensor, ivs_tensor, *input_tensors):
        return self._reduce_marginal_inference(self._compute_reducible(
            w_tensor, ivs_tensor, *input_tensors, log=False, weighted=True, use_ivs=True))

    @utils.docinherit(OpNode)
    @utils.lru_cache
    def _compute_log_value(self, w_tensor, ivs_tensor, *value_tensors):

        # Defines gradient for the log value
        def soft_gradient(grad):
            scattered_grads = self._compute_log_gradient(
                grad, w_tensor, ivs_tensor, *value_tensors, sum_weight_grads=True)
            return [sg for sg in scattered_grads if sg is not None]

        # Wrap the log value with its custom gradient
        @tf.custom_gradient
        def _log_value(*input_tensors):
            ret = self._reduce_marginal_inference_log(self._compute_reducible(
                w_tensor, ivs_tensor, *value_tensors, log=True, weighted=True, use_ivs=True))
            return ret, soft_gradient
        return _log_value(*self._get_differentiable_inputs(w_tensor, ivs_tensor, *value_tensors))

    def _get_differentiable_inputs(self, w_tensor, ivs_tensor, *value_tensors):
        """Selects the tensors to include for a tf.custom_gradient when computing the log-value.

        Args:
            w_tensor (Tensor): A ``Tensor`` of shape [num_sums, max_sum_size] with the value of
                               the weights corresponding to this node.
            ivs_tensor (Tensor): A ``Tensor`` of shape [batch, num_sums, max_sum_size] with the
                                 value of the IVs corresponding to this node.

        """
        return [w_tensor] + ([ivs_tensor] if self._ivs else []) + list(value_tensors)

    @utils.docinherit(OpNode)
    def _compute_mpe_value(self, w_tensor, ivs_tensor, *input_tensors):
        return self._reduce_mpe_inference(self._compute_reducible(
            w_tensor, ivs_tensor, *input_tensors, log=False, weighted=True, use_ivs=True))

    @utils.docinherit(OpNode)
    @utils.lru_cache
    def _compute_log_mpe_value(self, w_tensor, ivs_tensor, *input_tensors):
        return self._reduce_mpe_inference_log(self._compute_reducible(
            w_tensor, ivs_tensor, *input_tensors, log=True, weighted=True, use_ivs=True))

    @utils.lru_cache
    def _compute_mpe_path_common(
            self, reducible_tensor, counts, w_tensor, ivs_tensor, *input_tensors):
        """Common operations for computing the MPE path.

        Args:
            reducible_tensor (Tensor): A (weighted) ``Tensor`` of (log-)values of this node.
            counts (Tensor): A ``Tensor`` that contains the accumulated counts of the parents
                             of this node.
            w_tensor (Tensor):  A ``Tensor`` containing the (log-)value of the weights.
            ivs_tensor (Tensor): A ``Tensor`` containing the (log-)value of the IVs.
            input_tensors (list): A list of ``Tensor``s with outputs of the child nodes.

        Returns:
            A ``list`` of ``tuple``s [(MPE counts, input tensor), ...] where the first corresponds
            to the Weights of this node, the second corresponds to the IVs and the remaining
            tuples correspond to the nodes in ``self._values``.
        """
        max_indices = self._reduce_argmax(reducible_tensor)
        max_counts = utils.scatter_values(
            params=counts, indices=max_indices, num_out_cols=self._max_sum_size)
        max_counts_acc, max_counts_split = self._accumulate_and_split_to_children(
            max_counts, *input_tensors)
        return self._scatter_to_input_tensors(
            (max_counts, w_tensor),  # Weights
            (max_counts_acc, ivs_tensor),  # IVs
            *[(t, v) for t, v in zip(max_counts_split, input_tensors)])  # Values

    @utils.lru_cache
    def _accumulate_and_split_to_children(self, x, *input_tensors):
        """Accumulates the values in x over the op axis. Potentially also accumulates for every
        unique input if appropriate (e.g. in SumsLayer).

        Args:
            x (Tensor): A ``Tensor`` containing the values to accumulate and split among the
                        children.
            input_tensors (tuple): A ``tuple`` of ``Tensors`` holding the value of the children's
                                   outputs. These might be used in e.g. SumsLayer to determine
                                   unique inputs so that values can be accumulated before passing
                                   them downward.
        Returns:
            A ``tuple`` of size 2 with the accumulated values and a list of accumulated values
            corresponding to each input.
        """
        if self._num_sums > 1:
            x_acc = tf.reduce_sum(x, axis=self._op_axis)
        else:
            x_acc = tf.squeeze(x, axis=self._op_axis)

        _, _, *value_sizes = self.get_input_sizes()
        return x_acc, tf.split(x_acc, value_sizes, axis=self._op_axis)

    @utils.docinherit(OpNode)
    @utils.lru_cache
    def _compute_mpe_path(self, counts, w_tensor, ivs_tensor, *input_tensors,
                          use_unweighted=False, with_ivs=True, add_random=None):
        weighted = not use_unweighted or any(v.node.is_var for v in self._values)
        reducible = self._compute_reducible(w_tensor, ivs_tensor, *input_tensors, log=False,
                                            weighted=weighted, use_ivs=with_ivs)
        if add_random is not None:
            self.logger.warn(
                "%s: no support for add_random in non-log MPE path computation." % self)
        return self._compute_mpe_path_common(
            reducible, counts, w_tensor, ivs_tensor, *input_tensors)

    @utils.docinherit(OpNode)
    @utils.lru_cache
    def _compute_log_mpe_path(self, counts, w_tensor, ivs_tensor, *input_tensors,
                              use_unweighted=False, with_ivs=True, add_random=None):
        weighted = not use_unweighted or any(v.node.is_var for v in self._values)
        reducible = self._compute_reducible(w_tensor, ivs_tensor, *input_tensors, log=True,
                                            weighted=weighted, use_ivs=with_ivs)
        if not weighted and self._num_sums > 1 and reducible.shape[self._op_axis].value == 1:
            reducible = tf.tile(reducible, (1, self._num_sums, 1))
        # Add random
        if add_random is not None:
            reducible += tf.random_uniform(
                tf.shape(reducible), minval=0.0, maxval=add_random, dtype=conf.dtype)
        return self._compute_mpe_path_common(
                    reducible, counts, w_tensor, ivs_tensor, *input_tensors)

    @utils.lru_cache
    def _compute_log_gradient(
            self, gradients, w_tensor, ivs_tensor, *value_tensors, with_ivs=True,
            sum_weight_grads=False):
        """Computes gradient for log probabilities.

        Args:
            gradients (Tensor): A ``Tensor`` of shape [batch, num_sums] that contains the
                                accumulated backpropagated gradient coming from this node's parents.
            w_tensor (Tensor): A ``Tensor`` of shape [num_sums, max_sum_size] that contains the
                               weights corresponding to this node.
            ivs_tensor (Tensor): A ``Tensor`` of shape [batch, num_sums, max_sum_size] that
                                 corresponds to the IVs of this node.
            value_tensors (tuple): A ``tuple`` of ``Tensor``s that correspond to the values of the
                                   children of this node.
            sum_weight_grads (bool): A ``bool`` that marks whether the weight gradients should be
                                     summed over the batch axis.
        Returns:
            A ``list`` of ``tuple``s where each tuple consists of a gradient and the forward-pass
            tensor corresponding to the gradient. Starts with weights, then IVs and the remaining
            ``tuple`` correspond to ``input_tensors``.
        """

        reducible = self._compute_reducible(
            w_tensor, ivs_tensor, *value_tensors, log=True, use_ivs=with_ivs, weighted=True)

        # Below exploits the memoization decorator, since
        log_sum = tf.expand_dims(
            self._reduce_marginal_inference_log(reducible), axis=self._reduce_axis)
        w_grad = tf.expand_dims(gradients, axis=self._reduce_axis) * tf.exp(reducible - log_sum)

        inp_grad, inp_grad_split = self._accumulate_and_split_to_children(w_grad)

        if sum_weight_grads:
            w_grad = tf.reduce_sum(w_grad, axis=0, keepdims=False)

        return self._scatter_to_input_tensors(
            (w_grad, w_tensor),
            (inp_grad, ivs_tensor),
            *[(t, v) for t, v in zip(inp_grad_split, value_tensors)])

    @utils.lru_cache
    def _compute_gradient(self, gradients, w_tensor, ivs_tensor, *input_tensors, with_ivs=True):
        """Computes gradient for non-log probabilities.

        Args:
            gradients (Tensor): A ``Tensor`` of shape [batch, num_sums] that contains the
                                accumulated backpropagated gradient coming from this node's parents.
            w_tensor (Tensor): A ``Tensor`` of shape [num_sums, max_sum_size] that contains the
                               weights corresponding to this node.
            ivs_tensor (Tensor): A ``Tensor`` of shape [batch, num_sums, max_sum_size] that
                                 corresponds to the IVs of this node.
            input_tensors (tuple): A ``tuple`` of ``Tensor``s that correspond to the values of the
                                   children of this node.
        Returns:
            A ``list`` of ``tuple``s where each tuple consists of a gradient and the forward-pass
            tensor corresponding to the gradient. Starts with weights, then IVs and the remaining
            ``tuple`` correspond to ``input_tensors``.
        """
        raise NotImplementedError("%s: Currently there is no support for non-log values." % self)

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
        flat_value_scopes, ivs_scopes, *value_scopes = self._get_flat_value_scopes(
            weight_scopes, ivs_scopes, *value_scopes)
        if self._ivs:
            sublist_size = int(len(ivs_scopes) / self._num_sums)
            # Divide gathered ivs scopes into sublists, one per modelled Sum node.
            ivs_scopes_sublists = [ivs_scopes[i:i + sublist_size] for i in
                                   range(0, len(ivs_scopes), sublist_size)]
        return [Scope.merge_scopes(flat_value_scopes + ivs_scopes_sublists[i]
                                   if self._ivs else flat_value_scopes)
                for i in range(self._num_sums)]

    @utils.docinherit(OpNode)
    def _compute_valid(self, weight_scopes, ivs_scopes, *value_scopes):
        flat_value_scopes, ivs_scopes_, *value_scopes_ = self._get_flat_value_scopes(
            weight_scopes, ivs_scopes, *value_scopes)
        # If already invalid, return None
        if (any(s is None for s in value_scopes)
                or (self._ivs and ivs_scopes is None)):
            return None
        # IVs
        if self._ivs:
            # Verify number of IVs
            if len(ivs_scopes_) != len(flat_value_scopes) * self._num_sums:
                raise StructureError("Number of IVs (%s) and values (%s) does "
                                     "not match for %s"
                                     % (len(ivs_scopes_),
                                        len(flat_value_scopes) * self._num_sums,
                                        self))
            # Check if scope of all IVs is just one and the same variable
            if len(Scope.merge_scopes(ivs_scopes_)) > self._num_sums:
                return None
        # Check sum for completeness wrt values
        first_scope = flat_value_scopes[0]
        if any(s != first_scope for s in flat_value_scopes[1:]):
            self.info("%s is not complete with input value scopes %s", self, flat_value_scopes)
            return None

        return self._compute_scope(weight_scopes, ivs_scopes, *value_scopes)

    @property
    @utils.docinherit(OpNode)
    def _const_out_size(self):
        return True

    @utils.lru_cache
    def _prepare_component_wise_processing(
            self, w_tensor, ivs_tensor, *input_tensors, zero_prob_val=0.0):
        """Gathers inputs and combines them so that the resulting tensor can be reduced over the
        last axis to compute the (weighted) sums.

        Args:
            w_tensor (Tensor): A ``Tensor`` with the (log-)value of the weights of this node of
                               shape [num_sums, max_sum_size]
            ivs_tensor (Tensor): A ``Tensor`` with the (log-)value of the 'latent' ``IVs``.
            input_tensors (tuple): A tuple of ``Tensor``s  holding the value of the children of this
                                   node.
            zero_prob_val (float): The value of zero probability. This is important to know if some
                                   parts of the computation should be left out for masking.
        Returns:
            A tuple of size 3 containing: a weight ``Tensor`` that can be broadcast across sums, an
            IVs ``Tensor`` that can be applied component-wise to the sums and a ``Tensor`` that
            holds the unweighted values of the sum inputs of shape [batch, num_sums, max_sum_size].
        """
        w_tensor, ivs_tensor, *input_tensors = self._gather_input_tensors(
            w_tensor, ivs_tensor, *input_tensors)
        input_tensors = [tf.expand_dims(t, axis=self._op_axis) if len(t.shape) == 2 else t for
                         t in input_tensors]
        w_tensor = tf.expand_dims(w_tensor, axis=self._batch_axis)
        reducible_inputs = utils.concat_maybe(input_tensors, axis=self._reduce_axis)
        if ivs_tensor is not None:
            ivs_tensor = tf.reshape(ivs_tensor, shape=(-1, self._num_sums, self._max_sum_size))
        return w_tensor, ivs_tensor, reducible_inputs

    @utils.lru_cache
    def _reduce_marginal_inference(self, x):
        """Reduces a tensor for marginal non-log inference by sum(x, axis=reduce_axis).

        Args:
            x (Tensor): A ``Tensor`` of shape [batch, num_sums, max_sum_size] to reduce over the
                        last axis.

        Returns:
            A ``Tensor`` reduced over the last axis.
        """
        return tf.reduce_sum(x, axis=self._reduce_axis, keepdims=False)

    @utils.lru_cache
    def _reduce_marginal_inference_log(self, x):
        """Reduces a tensor for marginal log inference by log(sum(exp(x), axis=reduce_axis)).

        Args:
            x (Tensor): A ``Tensor`` of shape [batch, num_sums, max_sum_size] to reduce over the
                        last axis.

        Returns:
            A ``Tensor`` reduced over the last axis.
        """
        return tf.reduce_logsumexp(x, axis=self._reduce_axis, keepdims=False)

    @utils.lru_cache
    def _reduce_mpe_inference(self, x):
        """Reduces a tensor for MPE non-log inference by max(x, axis=reduce_axis)).

        Args:
            x (Tensor): A ``Tensor`` of shape [batch, num_sums, max_sum_size] to reduce over the
                        last axis.

        Returns:
            A ``Tensor`` reduced over the last axis.
        """
        return tf.reduce_max(x, axis=self._reduce_axis, keepdims=False)

    @utils.lru_cache
    def _reduce_mpe_inference_log(self, x):
        """Reduces a tensor for MPE log inference by max(x, axis=reduce_axis).

        Args:
            x (Tensor): A ``Tensor`` of shape [batch, num_sums, max_sum_size] to reduce over the
                        last axis.

        Returns:
            A ``Tensor`` reduced over the last axis.
        """
        return self._reduce_mpe_inference(x)

    @utils.lru_cache
    def _reduce_argmax(self, x):
        """Reduces a tensor by argmax(x, axis=reduce_axis)).

        Args:
            x (Tensor): A ``Tensor`` of shape [batch, num_sums, max_sum_size] to reduce over the
                        last axis.

        Returns:
            A ``Tensor`` reduced over the last axis.
        """
        return tf.argmax(x, axis=self._reduce_axis)  # Does not have a keepdims parameter

    @staticmethod
    @utils.lru_cache
    def cwise_add(a, b):
        """Component-wise addition of two tensors. Added explicitly for readability elsewhere and
        for straightforward memoization.

        Args:
            a (Tensor): Left-hand side.
            b (Tensor): Right-hand side.

        Returns:
            A component wise addition of ``a`` and ``b``.
        """
        return a + b

    @staticmethod
    @utils.lru_cache
    def cwise_mul(a, b):
        """Component-wise multiplication of two tensors. Added explicitly for readability elsewhere
        and for straightforward memoization.

        Args:
            a (Tensor): Left-hand side.
            b (Tensor): Right-hand side.

        Returns:
            A component wise multiplication of ``a`` and ``b``.
        """
        return a * b
