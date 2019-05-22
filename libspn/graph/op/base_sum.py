import abc
from libspn.graph.node import OpNode, Input
from libspn.graph.leaf.indicator import IndicatorLeaf
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
        reduce_axis (int): The axis over which to perform summing (or max for MPE)
        weights (input_like): Input providing weights node to this sum node.
            See :meth:`~libspn.Input.as_input` for possible values. If set
            to ``None``, the input is disconnected.
        latent_indicators (input_like): Input providing IndicatorLeafs of an explicit latent variable
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

    def __init__(self, *values, num_sums, weights=None, latent_indicators=None, sum_sizes=None,
                 inference_type=InferenceType.MARGINAL, batch_axis=0, op_axis=1,
                 reduce_axis=2, masked=False, sample_prob=None, name="Sum"):
        super().__init__(inference_type=inference_type, name=name)

        self.set_values(*values)
        self.set_weights(weights)
        self.set_latent_indicators(latent_indicators)

        # Initialize the number of sums and the sum sizes
        self._reset_sum_sizes(num_sums=num_sums, sum_sizes=sum_sizes)

        # Set the axes
        self._batch_axis = batch_axis
        self._op_axis = op_axis
        self._reduce_axis = reduce_axis

        # Set whether this instance is masked (e.g. SumsLayer)
        self._masked = masked

        # Set the sampling probability and sampling type
        self._sample_prob = sample_prob

    def _get_sum_sizes(self, num_sums):
        """Computes a list of sum sizes given the number of sums and the currently attached input
        nodes.

        Args:
            num_sums (int): The number of sums modeled by this node.
        Returns:
            A list of sum sizes, where the i-th element corresponds to the size of the i-th sum.
        """
        input_sizes = self.get_input_sizes()
        num_values = sum(input_sizes[2:])  # Skip latent_indicators, weights
        return num_sums * [num_values]

    def _build_mask(self):
        """Constructs mask that could be used to cancel out 'columns' that are padded as a result of
        varying reduction sizes. Returns a Boolean mask.

        Returns:
            By default the sums are not masked, returns ``None``
        """
        return None

    @utils.docinherit(OpNode)
    def serialize(self):
        data = super().serialize()
        data['values'] = [(i.node.name, i.indices) for i in self._values]
        if self._weights:
            data['weights'] = (self._weights.node.name, self._weights.indices)
        if self._latent_indicators:
            data['latent_indicators'] = (
            self._latent_indicators.node.name, self._latent_indicators.indices)
        data['num_sums'] = self._num_sums
        data['sum_sizes'] = self._sum_sizes
        data['op_axis'] = self._op_axis
        data['reduce_axis'] = self._reduce_axis
        data['batch_axis'] = self._batch_axis
        data['sample_prob'] = self._sample_prob
        return data

    @utils.docinherit(OpNode)
    def deserialize(self, data):
        super().deserialize(data)
        self.set_values()
        self.set_weights()
        self.set_latent_indicators()
        self._sample_prob = data['sample_prob']
        self._num_sums = data['num_sums']
        self._sum_sizes = data['sum_sizes']
        self._max_sum_size = max(self._sum_sizes) if self._sum_sizes else 0
        self._batch_axis = data['batch_axis']
        self._op_axis = data['op_axis']
        self._reduce_axis = data['reduce_axis']

    def disconnect_inputs(self):
        self._latent_indicators = self._weights = self._values = None

    @utils.docinherit(OpNode)
    def deserialize_inputs(self, data, nodes_by_name):
        super().deserialize_inputs(data, nodes_by_name)
        self._values = tuple(Input(nodes_by_name[nn], i)
                             for nn, i in data['values'])
        weights = data.get('weights', None)
        if weights:
            self._weights = Input(nodes_by_name[weights[0]], weights[1])
        latent_indicators = data.get('latent_indicators', None)
        if latent_indicators:
            self._latent_indicators = Input(nodes_by_name[latent_indicators[0]],
                                            latent_indicators[1])

    @property
    @utils.docinherit(OpNode)
    def inputs(self):
        return (self._weights, self._latent_indicators) + self._values

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
        """Resets the sizes and number of sums. If number of sums is specified, it will take that
        value, otherwise it will take the value that is already set. If sum_sizes is specified
        it will take that value, otherwise it will infer that using
        :meth:`~libspn.BaseSum._get_sum_sizes`. Finally, it also sets the maximum sum size.

        Args:
            num_sums (int): Number of sums modeled by this ``Node``.
            sum_sizes (int): A list of sum sizes with as many ``int``s as there are sums modeled.
        """
        self._num_sums = num_sums or self._num_sums
        self._sum_sizes = sum_sizes or self._get_sum_sizes(self._num_sums)
        self._max_sum_size = max(self._sum_sizes) if self._sum_sizes else 0

    @property
    def latent_indicators(self):
        """Input: IndicatorLeafs input."""
        return self._latent_indicators

    def set_latent_indicators(self, latent_indicators=None):
        """Set the IndicatorLeafs input.

        latent_indicators (input_like): Input providing IndicatorLeaf of an explicit latent variable
            associated with this sum node. See :meth:`~libspn.Input.as_input`
            for possible values. If set to ``None``, the input is disconnected.
        """
        self._latent_indicators, = self._parse_inputs(latent_indicators)

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
        self._values = self._parse_inputs(*values)

    def add_values(self, *values):
        """Add more inputs providing input values to this node.

        Args:
            *values (input_like): Inputs providing input values to this node.
                See :meth:`~libspn.Input.as_input` for possible values.
        """
        self._values += self._parse_inputs(*values)
        self._reset_sum_sizes()

    def generate_weights(self, initializer=tf.initializers.constant(1.0), trainable=True,
                         input_sizes=None, log=False, name=None):
        """Generate a weights node matching this sum node and connect it to
        this sum.

        The function calculates the number of weights based on the number
        of input values of this sum. Therefore, weights should be generated
        once all inputs are added to this node.

        Args:
            initializer: Initial value of the weights.
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
            num_values = sum(input_sizes[2:])  # Skip latent_indicators, weights
        else:
            num_values = max(self._sum_sizes)
        # Generate weights
        weights = Weights(
            initializer=initializer, num_weights=num_values, num_sums=self._num_sums,
            log=log, trainable=trainable, name=name)
        self.set_weights(weights)
        return weights

    def generate_latent_indicators(self, feed=None, name=None):
        """Generate an IndicatorLeaf node matching this sum node and connect it to
        this sum.

        IndicatorLeafs should be generated once all inputs are added to this node,
        otherwise the number of IndicatorLeafs will be incorrect.

        Args:
            feed (Tensor): See :class:`~libspn.IndicatorLeaf`.
            name (str): Name of the IndicatorLeaf node. If ``None`` use the name of the
                        sum + ``_IndicatorLeaf``.

        Return:
            IndicatorLeaf: Generated IndicatorLeaf node.
        """
        if not self._values:
            raise StructureError("%s is missing input values" % self)
        if name is None:
            name = self._name + "_IndicatorLeaf"
        latent_indicators = IndicatorLeaf(feed=feed, num_vars=self._num_sums,
                                          num_vals=self._max_sum_size, name=name)
        self.set_latent_indicators(latent_indicators)
        return latent_indicators

    @utils.lru_cache
    def _compute_reducible(self, w_tensor, latent_indicators_tensor, *input_tensors, weighted=True):
        """Computes a reducible ``Tensor`` so that reducing it over the last axis can be used for
        marginal inference, MPE inference and MPE path computation.

        Args:
            w_tensor (Tensor): A ``Tensor`` with the value of the weights of shape
                ``[num_sums, max_sum_size]``
            latent_indicators_tensor (Tensor): A ``Tensor`` with the value of the IndicatorLeaf corresponding to this node
                of shape ``[batch, num_sums * max_sum_size]``.
            input_tensors (tuple): A ``tuple`` of ``Tensors``s with the values of the children of
                this node.
            weighted (bool): Whether to apply the weights to the reducible values if possible.

        Returns:
            A ``Tensor`` of shape ``[batch, num_sums, max_sum_size]`` that can be used for computing
            marginal inference, MPE inference, gradients or MPE paths.
        """
        if not self._values:
            raise StructureError("%s is missing input values" % self)
        if not self._weights:
            raise StructureError("%s is missing weights" % self)

        # Prepare tensors for component-wise application of weights and IndicatorLeaf
        w_tensor, latent_indicators_tensor, reducible = self._prepare_component_wise_processing(
            w_tensor, latent_indicators_tensor, *input_tensors, zero_prob_val=-float('inf'))

        # Apply latent IndicatorLeaf
        if self._latent_indicators:
            reducible = utils.cwise_add(reducible, latent_indicators_tensor)

        # Apply weights
        if weighted:
            reducible = utils.cwise_add(reducible, w_tensor)

        return reducible

    @utils.docinherit(OpNode)
    @utils.lru_cache
    def _compute_out_size(self, *input_out_sizes):
        return self._num_sums

    @utils.docinherit(OpNode)
    @utils.lru_cache
    def _compute_log_value(self, w_tensor, latent_indicators_tensor, *value_tensors):
        return self._reduce_marginal_inference_log(self._compute_reducible(
            w_tensor, latent_indicators_tensor, *value_tensors, weighted=True))

    def _get_differentiable_inputs(self, w_tensor, latent_indicators_tensor, *value_tensors):
        """Selects the tensors to include for a tf.custom_gradient when computing the log-value.

        Args:
            w_tensor (Tensor): A ``Tensor`` of shape [num_sums, max_sum_size] with the value of
                               the weights corresponding to this node.
            latent_indicators_tensor (Tensor): A ``Tensor`` of shape [batch, num_sums, max_sum_size] with the
                                 value of the IndicatorLeaf corresponding to this node.
`
        """
        return [w_tensor] + ([latent_indicators_tensor] if self._latent_indicators else []) + list(
            value_tensors)

    @utils.docinherit(OpNode)
    @utils.lru_cache
    def _compute_log_mpe_value(self, w_tensor, latent_indicators_tensor, *value_tensors):
        return self._reduce_mpe_inference_log(self._compute_reducible(
            w_tensor, latent_indicators_tensor, *value_tensors, weighted=True))

    @utils.lru_cache
    def _compute_mpe_path_common(
        self, reducible_tensor, counts, w_tensor, latent_indicators_tensor, *input_tensors,
        sample=False, sample_prob=None, accumulate_weights_batch=False):
        """Common operations for computing the MPE path.

        Args:
            reducible_tensor (Tensor): A (weighted) ``Tensor`` of (log-)values of this node.
            counts (Tensor): A ``Tensor`` that contains the accumulated counts of the parents
                             of this node.
            w_tensor (Tensor):  A ``Tensor`` containing the (log-)value of the weights.
            latent_indicators_tensor (Tensor): A ``Tensor`` containing the (log-)value of the IndicatorLeaf.
            input_tensors (list): A list of ``Tensor``s with outputs of the child nodes.
            log (bool): Whether the computation is in log-space or not
            sample (bool): Whether to sample the 'winner' of the max or not
            sample_prob (Tensor): A scalar ``Tensor`` indicating the probability of drawing
                a sample. If a sample is drawn, the probability for each index is given by the
                (log-)normalized probability as given by ``reducible_tensor``.
        Returns:
            A ``list`` of ``tuple``s [(MPE counts, input tensor), ...] where the first corresponds
            to the Weights of this node, the second corresponds to the IndicatorLeaf and the remaining
            tuples correspond to the nodes in ``self._values``.
        """
        sample_prob = utils.maybe_first(sample_prob, self._sample_prob)
        num_samples = 1 if reducible_tensor.shape[self._reduce_axis] != 1 else self._num_sums
        if sample:
            max_indices = self._reduce_sample_log(
                reducible_tensor, sample_prob=sample_prob, num_samples=num_samples)
        else:
            max_indices = self._reduce_argmax(reducible_tensor, num_samples=num_samples)
        max_counts = utils.scatter_values(
            params=counts, indices=max_indices, num_out_cols=self._max_sum_size)
        max_counts_acc, max_counts_split = self._accumulate_and_split_to_children(
            max_counts, *input_tensors)
        if accumulate_weights_batch:
            max_counts = tf.reduce_sum(max_counts, axis=0, keepdims=False)
        return self._scatter_to_input_tensors(
            (max_counts, w_tensor),  # Weights
            (max_counts_acc, latent_indicators_tensor),  # IndicatorLeaf
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
        return x_acc, tf.split(x_acc, value_sizes, axis=self._batch_axis + 1)

    @utils.docinherit(OpNode)
    @utils.lru_cache
    def _compute_log_mpe_path(self, counts, w_tensor, latent_indicators_tensor, *input_tensors,
                              use_unweighted=False,
                              accumulate_weights_batch=False, sample=False, sample_prob=None):
        weighted = not use_unweighted or any(v.node.is_var for v in self._values)
        reducible = self._compute_reducible(
            w_tensor, latent_indicators_tensor, *input_tensors, weighted=weighted)

        return self._compute_mpe_path_common(
            reducible, counts, w_tensor, latent_indicators_tensor, *input_tensors,
            accumulate_weights_batch=accumulate_weights_batch, sample=sample,
            sample_prob=sample_prob)

    @property
    def _tile_unweighted_size(self):
        return self._num_sums

    def _get_flat_value_scopes(self, weight_scopes, latent_indicators_scopes, *value_scopes):
        """Get a flat representation of the value scopes per sum.

        Args:
            weight_scopes (list): A list of ``Scope``s corresponding to the weights.
            latent_indicators_scopes (list): A list of ``Scope``s corresponding to the IndicatorLeaf.
            value_scopes (tuple): A ``tuple`` of ``list``s of ``Scope``s corresponding to the
                                  scope lists of the children of this node.

        Returns:
            A tuple of flat value scopes corresponding to this node's output. The IndicatorLeaf scopes and
            the value scopes.
        """
        if not self._values:
            raise StructureError("%s is missing input values" % self)
        _, latent_indicators_scopes, *value_scopes = self._gather_input_scopes(
            weight_scopes, latent_indicators_scopes, *value_scopes)
        return list(chain.from_iterable(value_scopes)), latent_indicators_scopes, value_scopes

    @utils.docinherit(OpNode)
    def _compute_scope(self, weight_scopes, latent_indicators_scopes, *value_scopes):
        flat_value_scopes, latent_indicators_scopes, *value_scopes = self._get_flat_value_scopes(
            weight_scopes, latent_indicators_scopes, *value_scopes)
        if self._latent_indicators:
            sublist_size = int(len(latent_indicators_scopes) / self._num_sums)
            # Divide gathered latent_indicators scopes into sublists, one per modelled Sum node.
            latent_indicators_scopes_sublists = [latent_indicators_scopes[i:i + sublist_size] for i
                                                 in
                                                 range(0, len(latent_indicators_scopes),
                                                       sublist_size)]
        return [Scope.merge_scopes(flat_value_scopes + latent_indicators_scopes_sublists[i]
                                   if self._latent_indicators else flat_value_scopes)
                for i in range(self._num_sums)]

    @utils.docinherit(OpNode)
    def _compute_valid(self, weight_scopes, latent_indicators_scopes, *value_scopes):
        # If already invalid, return None
        if (any(s is None for s in value_scopes)
                or (self._latent_indicators and latent_indicators_scopes is None)):
            return None
        flat_value_scopes, latent_indicators_scopes_, *value_scopes_ = self._get_flat_value_scopes(
            weight_scopes, latent_indicators_scopes, *value_scopes)
        # IndicatorLeaf
        if self._latent_indicators:
            # Verify number of IndicatorLeaf
            if len(latent_indicators_scopes_) != len(flat_value_scopes) * self._num_sums:
                raise StructureError("Number of IndicatorLeaf (%s) and values (%s) does "
                                     "not match for %s"
                                     % (len(latent_indicators_scopes_),
                                        len(flat_value_scopes) * self._num_sums,
                                        self))
            # Check if scope of all IndicatorLeaf is just one and the same variable
            if len(Scope.merge_scopes(latent_indicators_scopes_)) > self._num_sums:
                return None
        # Check sum for completeness wrt values
        first_scope = flat_value_scopes[0]
        if any(s != first_scope for s in flat_value_scopes[1:]):
            self.info("%s is not complete with input value scopes %s", self, flat_value_scopes)
            return None

        return self._compute_scope(weight_scopes, latent_indicators_scopes, *value_scopes)

    @property
    @utils.docinherit(OpNode)
    def _const_out_size(self):
        return True

    @utils.lru_cache
    def _prepare_component_wise_processing(
            self, w_tensor, latent_indicators_tensor, *input_tensors, zero_prob_val=0.0):
        """
        Gathers inputs and combines them so that the resulting tensor can be reduced over the
        last axis to compute the (weighted) sums.

        Args:
            w_tensor (Tensor): A ``Tensor`` with the (log-)value of the weights of this node of
                               shape [num_sums, max_sum_size]
            latent_indicators_tensor (Tensor): A ``Tensor`` with the (log-)value of the 'latent' ``IndicatorLeaf``.
            input_tensors (tuple): A tuple of ``Tensor``s  holding the value of the children of this
                                   node.
            zero_prob_val (float): The value of zero probability. This is important to know if some
                                   parts of the computation should be left out for masking.
        Returns:
            A tuple of size 3 containing: a weight ``Tensor`` that can be broadcast across sums, an
            IndicatorLeaf ``Tensor`` that can be applied component-wise to the sums and a ``Tensor`` that
            holds the unweighted values of the sum inputs of shape [batch, num_sums, max_sum_size].
        """

        w_tensor, latent_indicators_tensor, *input_tensors = self._gather_input_tensors(
            w_tensor, latent_indicators_tensor, *input_tensors)

        reducible_inputs = tf.expand_dims(
            tf.concat(input_tensors, axis=self._reduce_axis - 1), axis=self._op_axis)

        w_tensor = tf.expand_dims(w_tensor, axis=self._batch_axis)
        if latent_indicators_tensor is not None:
            latent_indicators_tensor = tf.reshape(
                latent_indicators_tensor, shape=(-1, self._num_sums, self._max_sum_size))

        return w_tensor, latent_indicators_tensor, reducible_inputs

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
    def _reduce_argmax(self, x, num_samples=1):
        """Reduces a tensor by argmax(x, axis=reduce_axis)).

        Args:
            x (Tensor): A ``Tensor`` of shape [batch, num_sums, max_sum_size] to reduce over the
                        last axis.

        Returns:
            A ``Tensor`` reduced over the last axis.
        """
        if conf.argmax_zero:
            # If true, uses TensorFlow's argmax directly, yielding a bias towards the zeroth index
            argmax = tf.argmax(x, axis=self._reduce_axis)
            if num_samples == 1:
                return argmax
            return tf.tile(tf.expand_dims(
                argmax, axis=-1), [1] * (len(argmax.shape) - 1) + [self._tile_unweighted_size])

        # Return random index in case multiple values equal max
        x_max = tf.expand_dims(self._reduce_mpe_inference(x), self._reduce_axis)
        x_eq_max = tf.cast(tf.equal(x, x_max), tf.float32)
        if self._masked:
            x_eq_max *= tf.expand_dims(tf.cast(self._build_mask(), tf.float32),
                                       axis=self._batch_axis)
        sample = self.multinomial_sample(tf.log(x_eq_max), num_samples)
        return sample

    @staticmethod
    def sample_and_transpose(d, sample_shape):
        sample = d.sample(sample_shape=sample_shape)
        if sample_shape == ():
            return sample
        else:
            return tf.transpose(sample, list(range(1, len(sample.shape))) + [0])

    @utils.lru_cache
    def _reduce_sample_log(self, logits, sample_prob=None, num_samples=1):
        """Samples a tensor with log likelihoods, i.e. sample(x, axis=reduce_axis)).

        Args:
            logits (Tensor): A ``Tensor`` of shape [batch, num_sums, max_sum_size] to reduce over
                the last axis.
            sample_prob (Tensor or float): A ``Tensor`` or float indicating the probability of
                taking a sample.

        Returns:
            A ``Tensor`` reduced over the last axis.
        """

        # Categorical eventually uses non-log probabilities, so here we reuse as much as we can to
        # predetermine it
        def _sample():
            sample = self.multinomial_sample(logits, num_samples=num_samples)
            return sample

        def _select_sample_or_argmax(x):
            mask = tf.less(tf.random_uniform(tf.shape(x)), sample_prob)
            return tf.where(mask, x, self._reduce_argmax(logits, num_samples=num_samples))

        if sample_prob is not None:
            if isinstance(sample_prob, (float, int)):
                if sample_prob < 0 or sample_prob > 1:
                    raise ValueError("{}: Sample probability should be between 0 and 1. Got {} "
                                     "instead.".format(self, sample_prob))
                if sample_prob != 0:
                    sample_op = _sample()
                    if sample_prob == 1.0:
                        return sample_op
                    return _select_sample_or_argmax(sample_op)

            return _select_sample_or_argmax(_sample())
        else:
            return _sample()

    @utils.lru_cache
    def multinomial_sample(self, logits, num_samples):
        shape = tf.shape(logits)
        last_dim = shape[-1]
        logits = tf.reshape(logits, (-1, last_dim))
        sample = tf.multinomial(logits, num_samples)

        if self._tile_unweighted_size == num_samples and self._max_sum_size > 1:
            shape = tf.concat((shape[:-1], [num_samples]), axis=0)
            return tf.squeeze(tf.reshape(sample, shape), axis=self._reduce_axis - 1)

        return tf.reshape(sample, shape[:-1])
