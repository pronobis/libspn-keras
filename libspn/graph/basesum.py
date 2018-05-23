import abc
from libspn.graph.node import OpNode, Input
from libspn.graph.ivs import IVs
from libspn.graph.weights import Weights
from libspn.inference.type import InferenceType
from libspn import conf
from libspn.exceptions import StructureError
import libspn.utils as utils
from libspn.log import get_logger
import tensorflow as tf


class BaseSum(OpNode, abc.ABC):

    logger = get_logger()
    info = logger.info

    __info = info

    def __init__(self, *values, num_sums, weights=None, ivs=None, sum_sizes=None,
                 inference_type=InferenceType.MARGINAL, batch_axis=0, op_axis=1, reduce_axis=2,
                 name="Sum"):
        super().__init__(inference_type, name)
        self.set_values(*values)
        self.set_weights(weights)
        self.set_ivs(ivs)

        # if num_sums < 1:
        #     raise StructureError("%s: Number of sums must be positive. Got %s." % self, num_sums)

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
        return data

    def deserialize(self, data):
        super().deserialize(data)
        self.set_values()
        self.set_weights()
        self.set_ivs()

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

    def _compute_reducible(
            self, w_tensor, ivs_tensor, *input_tensors, log=True,
            apply_ivs=False, apply_weights=True):
        if not self._values:
            raise StructureError("%s is missing input values" % self)
        if not self._weights:
            raise StructureError("%s is missing weights" % self)

        zero_prob_val = -float('inf') if log else 0.0
        cwise_op = tf.add if log else tf.multiply

        w_tensor, ivs_tensor, reducible = self._gather_and_combine_to_reducible(
            w_tensor, ivs_tensor, input_tensors, zero_prob_val=zero_prob_val)
        if apply_ivs and self._ivs:
            reducible = cwise_op(reducible, ivs_tensor)
        if apply_weights:
            reducible = cwise_op(reducible, w_tensor)
        return reducible

    def _compute_out_size(self, *input_out_sizes):
        return self._num_sums

    def _compute_value(self, w_tensor, ivs_tensor, *input_tensors):
        return tf.reduce_sum(
            self._compute_reducible(
                w_tensor, ivs_tensor, *input_tensors, log=False, apply_ivs=self._ivs,
                apply_weights=True),
            axis=self._reduce_axis)

    def _compute_log_value(self, w_tensor, ivs_tensor, *input_tensors):
        return tf.reduce_logsumexp(
            self._compute_reducible(
                w_tensor, ivs_tensor, *input_tensors, log=True, apply_ivs=self._ivs,
                apply_weights=True),
            axis=self._reduce_axis)

    def _compute_mpe_value(self, w_tensor, ivs_tensor, *input_tensors):
        return tf.reduce_max(
            self._compute_reducible(
                w_tensor, ivs_tensor, *input_tensors, log=False, apply_ivs=self._ivs,
                apply_weights=True),
            axis=self._reduce_axis)

    def _compute_log_mpe_value(self, w_tensor, ivs_tensor, *input_tensors):
        return tf.reduce_max(
            self._compute_reducible(
                w_tensor, ivs_tensor, *input_tensors, log=True, apply_ivs=self._ivs,
                apply_weights=True),
            axis=self._reduce_axis)

    def _compute_mpe_path_common(
            self, reducible_tensor, counts, w_tensor, ivs_tensor, *input_tensors):
        max_indices = tf.argmax(reducible_tensor, axis=self._reduce_axis)
        max_counts = utils.scatter_values(
            params=counts, indices=max_indices, num_out_cols=self._max_sum_size)
        max_counts_acc, max_counts_split = self._accumulate_and_split_to_children(
            max_counts, *input_tensors)
        return self._scatter_to_input_tensors(
            (max_counts, w_tensor),  # Weights
            (max_counts_acc, ivs_tensor),  # IVs
            *[(t, v) for t, v in zip(max_counts_split, input_tensors)])  # Values

    def _accumulate_and_split_to_children(self, x, *input_tensors):
        x_acc = self._accumulate_unique_inputs(x)
        x_acc_split = self._distribute_to_children(x_acc)
        return x_acc, x_acc_split

    def _accumulate_unique_inputs(self, x):
        return tf.reduce_sum(x, axis=self._op_axis)

    def _compute_mpe_path(self, counts, w_tensor, ivs_tensor, *input_tensors,
                          use_unweighted=False, with_ivs=True, add_random=None):
        apply_weights = not use_unweighted or any(v.node.is_var for v in self._values)
        reducible = self._compute_reducible(w_tensor, ivs_tensor, *input_tensors, log=False,
                                            apply_weights=apply_weights, apply_ivs=with_ivs)
        return self._compute_mpe_path_common(
            reducible, counts, w_tensor, ivs_tensor, *input_tensors)

    def _compute_log_mpe_path(self, counts, w_tensor, ivs_tensor, *input_tensors,
                              use_unweighted=False, with_ivs=True, add_random=None):
        apply_weights = not use_unweighted or any(v.node.is_var for v in self._values)
        reducible = self._compute_reducible(w_tensor, ivs_tensor, *input_tensors, log=True,
                                            apply_weights=apply_weights, apply_ivs=with_ivs)
        if add_random is not None:
            reducible += tf.random_uniform(
                tf.shape(reducible), minval=0., maxval=add_random, dtype=conf.dtype)
        if not apply_weights and self._num_sums > 1 and reducible.shape[self._op_axis].value == 1:
            reducible = tf.tile(reducible, (1, self._num_sums, 1))
        return self._compute_mpe_path_common(
                    reducible, counts, w_tensor, ivs_tensor, *input_tensors)

    def _compute_log_gradient(
            self, gradients, w_tensor, ivs_tensor, *input_tensors, with_ivs=True):
        reducible = self._compute_reducible(
            w_tensor, ivs_tensor, *input_tensors, log=True, apply_ivs=with_ivs, apply_weights=True)
        log_sum = tf.reduce_logsumexp(reducible, axis=self._reduce_axis, keepdims=True)
        w_grad = tf.expand_dims(gradients, axis=self._reduce_axis) * tf.exp(reducible - log_sum)

        inp_grad, inp_grad_split = self._accumulate_and_split_to_children(w_grad)
        return self._scatter_to_input_tensors(
            (w_grad, w_tensor),
            (inp_grad, ivs_tensor),
            *[(t, v) for t, v in zip(inp_grad_split, input_tensors)])

    def _compute_gradient(self, gradients, w_tensor, ivs_tensor, *input_tensors, with_ivs=True):
        reducible = self._compute_reducible(
            w_tensor, ivs_tensor, *input_tensors, log=False, apply_ivs=False, apply_weights=False)
        gradients = tf.expand_dims(gradients, axis=self._reduce_axis)

        w_grad = tf.reduce_sum(gradients * reducible, axis=self._op_axis)
        inp_grad, inp_grad_split = self._accumulate_and_split_to_children(gradients * w_tensor)

        return self._scatter_to_input_tensors(
            (w_grad, w_tensor),  # Weights
            (w_grad, ivs_tensor),  # IVs
            *[(t, v) for t, v in zip(inp_grad_split, input_tensors)])  # Values

    @abc.abstractmethod
    def _compute_scope(self, weight_scopes, ivs_scopes, *value_scopes):
        pass

    @abc.abstractmethod
    def _compute_valid(self, weight_scopes, ivs_scopes, *value_scopes):
        pass

    @property
    def _const_out_size(self):
        return True

    def _gather_and_combine_to_reducible(
            self, w_tensor, ivs_tensor, input_tensors, zero_prob_val=None):
        w_tensor, ivs_tensor, *input_tensors = self._gather_input_tensors(
            w_tensor, ivs_tensor, *input_tensors)
        input_tensors = [tf.expand_dims(t, axis=self._op_axis) if len(t.shape) == 2 else t for
                         t in input_tensors]
        w_tensor = tf.expand_dims(w_tensor, axis=self._batch_axis)
        reducible_inputs = utils.concat_maybe(input_tensors, axis=self._reduce_axis)
        if ivs_tensor is not None:
            ivs_tensor = tf.reshape(ivs_tensor, shape=(-1, self._num_sums, self._max_sum_size))
        return w_tensor, ivs_tensor, reducible_inputs

    def _distribute_to_children(self, x):
        _, _, *value_sizes = self.get_input_sizes()
        return tf.split(x, value_sizes, axis=self._op_axis)
