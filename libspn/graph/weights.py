import tensorflow as tf
from libspn.graph.node import ParamNode
from libspn.graph.algorithms import traverse_graph
from libspn import conf
from libspn.graph.leaf.location_scale import LocationScaleLeaf
from libspn.utils.serialization import register_serializable
from libspn import utils
from libspn.exceptions import StructureError
from libspn.utils.graphkeys import SPNGraphKeys


@register_serializable
class Weights(ParamNode):
    """A node containing a vector of weights of a sum node.

    Args:
        initializer: Initial value of the weights.
        num_weights (int): Number of weights in the vector.
        num_sums (int): Number of sum nodes the weight vector/matrix represents.
        log (bool): If "True", the weights are represented in log space.
        mask (list): List of booleans with num_weights * num_sums elements, used for masking weights
        name (str): Name of the node.

    Attributes:
        trainable(bool): Should these weights be updated during training?
    """

    def __init__(self, initializer=tf.initializers.constant(1.0), num_weights=1, num_sums=1,
                 log=False, trainable=True, mask=None, name="Weights"):
        if not isinstance(num_weights, int) or num_weights < 1:
            raise ValueError("num_weights must be a positive integer")

        if not isinstance(num_sums, int) or num_sums < 1:
            raise ValueError("num_sums must be a positive integer, got {} ({})".format(
                num_sums, type(num_sums)))

        self._initializer = initializer
        self._num_weights = num_weights
        self._num_sums = num_sums
        self._log = log
        self._trainable = trainable
        self._mask = mask
        super().__init__(name)

    def serialize(self):
        data = super().serialize()
        data['num_weights'] = self._num_weights
        data['num_sums'] = self._num_sums
        data['log'] = self._log
        data['trainable'] = self._trainable
        data['initializer'] = self._initializer
        data['value'] = self._variable
        data['mask'] = self._mask
        return data

    def deserialize(self, data):
        self._initializer = data['initializer']
        self._num_weights = data['num_weights']
        self._num_sums = data['num_sums']
        self._log = data['log']
        self._trainable = data['trainable']
        self._mask = data['mask']
        super().deserialize(data)
        # Create an op for deserializing value
        v = data['value']
        if v is not None:
            with tf.name_scope(self._name + "/"):
                return tf.assign(self._variable, v)
        else:
            return None

    @property
    def log(self):
        """bool: Boolean variable indicating the space in which weights are stored."""
        return self._log

    @property
    def mask(self):
        """list(int): Boolean mask for weights."""
        return self._mask

    @property
    def num_weights(self):
        """int: Number of weights in the vector."""
        return self._num_weights

    @property
    def num_sums(self):
        """int: Number of sum nodes the weights vector/matrix represents."""
        return self._num_sums

    @property
    def variable(self):
        """Variable: The TF variable of shape ``[num_sums, num_weights]``
        holding the weights."""
        return self._variable

    def initialize(self):
        """Return a TF operation assigning the initial value to the weights.

        Returns:
            Tensor: The initialization assignment operation.
        """
        return self._variable.initializer

    def assign(self, value):
        """Return a TF operation assigning values to the weights.

        Args:
            value: The value to assign to the weights.
        Returns:
            Tensor: The assignment operation.
        """
        if self._log:
            raise StructureError("Trying to assign non-log values to log-weights.")

        value = tf.where(tf.is_nan(value), tf.ones_like(value) * 0.01, value)
        if self._mask and not all(self._mask):
            # Only perform masking if mask is given and mask contains any 'False'
            value *= tf.cast(tf.reshape(self._mask, value.shape), dtype=conf.dtype)
        value = value / tf.reduce_sum(value, axis=-1, keepdims=True)
        return tf.assign(self._variable, value)

    def assign_log(self, value):
        """Return a TF operation assigning log values to the weights.

        Args:
            value: The value to assign to the weights.
        Returns:
            Tensor: The assignment operation.
        """
        if not self._log:
            raise StructureError("Trying to assign log values to non-log weights.")

        value = tf.where(tf.is_nan(value), tf.log(tf.ones_like(value) * 0.01), value)
        if self._mask and not all(self._mask):
            # Only perform masking if mask is given and mask contains any 'False'
            value += tf.log(tf.cast(tf.reshape(self._mask, value.shape), dtype=conf.dtype))
        normalized_value = value - tf.reduce_logsumexp(value, axis=-1, keepdims=True)
        return tf.assign(self._variable, normalized_value)

    def normalize(self, value=None, name="Normalize", linear_w_minimum=1e-2, log_w_minimum=-1e10):
        """Renormalizes the weights. If no value is given, the method will use the current
        weight values.

        Args:
            value (Tensor): A tensor to normalize and assign to this weight node.

        Returns:
            An Op that assigns a normalized value to this node.
        """
        with tf.name_scope(name):
            value = value or self._variable
            if self._log:
                value = tf.maximum(value, log_w_minimum)
                if self._mask and not all(self._mask):
                    # Only perform masking if mask is given and mask contains any 'False'
                    value += tf.log(tf.cast(tf.reshape(self._mask, value.shape), dtype=conf.dtype))
                return tf.assign(self._variable, tf.nn.log_softmax(value, axis=-1))
            else:
                value = tf.maximum(value, linear_w_minimum)
                if self._mask and not all(self._mask):
                    # Only perform masking if mask is given and mask contains any 'False'
                    value *= tf.cast(tf.reshape(self._mask, value.shape), dtype=conf.dtype)
                return tf.assign(self._variable, value / tf.reduce_sum(
                    value, axis=-1, keepdims=True))

    def update(self, value):
        """Return a TF operation adding the log-values to the log-weights.

        Args:
            value: The log-value to be added to the log-weights.

        Returns:
            Tensor: The assignment operation.
        """
        if self._mask and not all(self._mask):
            # Only perform masking if mask is given and mask contains any 'False'
            value += tf.log(tf.cast(tf.reshape(self._mask, value.shape), dtype=conf.dtype))
        # w_ij: w_ij + Δw_ij
        update_value = (self._variable if self._log else tf.log(self._variable)) + value
        normalized_value = tf.nn.log_softmax(update_value, axis=-1)
        return tf.assign(self._variable, (normalized_value if self._log else
                                          tf.exp(normalized_value)))

    def update_log(self, value):
        """Return a TF operation adding the log-values to the log-weights.

        Args:
            value: The log-value to be added to the log-weights.

        Returns:
            Tensor: The assignment operation.
        """
        if not self._log:
            raise StructureError("Trying to update non-log weights with log values.")
        if self._mask and not all(self._mask):
            # Only perform masking if mask is given and mask contains any 'False'
            value += tf.log(tf.cast(tf.reshape(self._mask, value.shape), dtype=conf.dtype))
        # w_ij: w_ij + Δw_ij
        update_value = self._variable + value
        normalized_value = tf.nn.log_softmax(update_value, axis=-1)
        return tf.assign(self._variable, normalized_value)

    def _create(self):
        """Creates a TF variable holding the vector of the SPN weights.

        Returns:
            Variable: A TF variable of shape ``[num_weights]``.
        """
        shape = (self._num_sums, self._num_weights)
        init_val = self._initializer(shape=shape, dtype=conf.dtype)
        if self._mask and not all(self._mask):
            # Only perform masking if mask is given and mask contains any 'False'
            init_val *= tf.cast(tf.reshape(self._mask, init_val.shape), dtype=conf.dtype)
        init_val = init_val / tf.reduce_sum(init_val, axis=-1, keepdims=True)
        if self._log:
            init_val = tf.log(init_val)

        self._variable = tf.Variable(init_val, dtype=conf.dtype)


    @utils.lru_cache
    def _compute_out_size(self):
        return self._num_weights * self._num_sums

    @utils.lru_cache
    def _compute_log_value(self):
        if self._log:
            return self._variable
        else:
            return tf.log(self._variable)

    def _compute_hard_gd_update(self, grads):
        if len(grads.shape) == 3:
            return tf.reduce_sum(grads, axis=0)
        return grads
    
    def _compute_hard_em_update(self, counts):
        if len(counts.shape) == 3:
            return tf.reduce_sum(counts, axis=0)
        return counts


@register_serializable
class BlockWeights(ParamNode):
    """A node containing a vector of weights of a sum node.

    Args:
        init_value: Initial value of the weights. For possible values, see
                    :meth:`~libspn.utils.broadcast_value`.
        num_weights (int): Number of weights in the vector.
        num_sums (int): Number of sum nodes the weight vector/matrix represents.
        log (bool): If "True", the weights are represented in log space.
        mask (list): List of booleans with num_weights * num_sums elements, used for masking weights
        name (str): Name of the node.

    Attributes:
        trainable(bool): Should these weights be updated during training?
    """

    def __init__(self, num_inputs, num_outputs, num_decomps, num_scopes, in_logspace=True,
                 trainable=True, mask=None, initializer=tf.random_uniform_initializer(0.0, 1.0),
                 name="TensorWeights"):
        if not isinstance(num_inputs, int) or num_inputs < 1:
            raise ValueError("num_inputs must be a positive integer")

        if not isinstance(num_outputs, int) or num_outputs < 1:
            raise ValueError("num_outputs must be a positive integer")

        self._num_inputs = num_inputs
        self._num_outputs = num_outputs
        self._num_decomps = num_decomps
        self._num_scopes = num_scopes
        self._normalize_axis = 2
        self._in_logspace = in_logspace
        self._trainable = trainable
        self._initializer = initializer
        self._mask = mask
        super().__init__(name)

    def serialize(self):
        raise NotImplementedError()

    def deserialize(self, data):
        raise NotImplementedError()

    @property
    def in_logspace(self):
        """bool: Boolean variable indicating the space in which weights are stored."""
        return self._in_logspace

    @property
    def mask(self):
        """list(int): Boolean mask for weights."""
        return self._mask

    @property
    def num_inputs(self):
        """int: Number of weights in the vector."""
        return self._num_inputs

    @property
    def num_outputs(self):
        """int: Number of sum nodes the weights vector/matrix represents."""
        return self._num_outputs

    @property
    def num_decomps(self):
        return self._num_decomps

    @property
    def num_scopes(self):
        return self._scopes

    @property
    def variable(self):
        """Variable: The TF variable of shape ``[num_sums, num_weights]``
        holding the weights."""
        return self._variable

    def initialize(self):
        """Return a TF operation assigning the initial value to the weights.

        Returns:
            Tensor: The initialization assignment operation.
        """
        return self._variable.initializer

    @property
    def log(self):
        return self._in_logspace

    def _normalized(self, x=None, logspace=None, linear_w_minimum=1e-2, log_w_minimum=-1e10):
        x = x if x is not None else self._variable
        logspace = logspace or self._in_logspace
        if logspace:
            x = tf.maximum(x, log_w_minimum)
            return x - tf.reduce_logsumexp(x, axis=self._normalize_axis, keepdims=True)
        x = tf.maximum(x, linear_w_minimum)
        return x / tf.reduce_sum(x, axis=self._normalize_axis, keepdims=True)

    def normalize(self, value=None, name="Normalize", linear_w_minimum=1e-2, log_w_minimum=-1e10):
        with tf.name_scope(name):
            normalized = self._normalized(
                x=value, linear_w_minimum=linear_w_minimum, log_w_minimum=log_w_minimum)
            return tf.assign(self._variable, normalized)

    def assign_log(self, value):
        return self.assign(value)

    def assign(self, value):
        """Return a TF operation assigning values to the weights.

        Args:
            value: The value to assign to the weights. For possible values, see
                   :meth:`~libspn.utils.broadcast_value`.

        Returns:
            Tensor: The assignment operation.
        """
        return tf.assign(self._variable, self._normalized(value))

    def renormalize(self):
        return self.assign(self._variable)

    @property
    def shape(self):
        return [self._num_scopes, self._num_decomps, self._num_inputs, self._num_outputs]

    def _create(self):
        """Creates a TF variable holding the vector of the SPN weights.

        Returns:
            Variable: A TF variable of shape ``[num_weights]``.
        """

        def log_wrapper(shape, dtype=None, partition_info=None):
            linspace = self._initializer(
                shape=shape, dtype=dtype, partition_info=partition_info)
            return self._normalized(tf.log(linspace) if self._in_logspace else linspace)

        self._variable = tf.get_variable(
            self._name, shape=self.shape, trainable=self._trainable, initializer=log_wrapper)

    def _compute_out_size(self):
        return self._num_inputs * self._num_outputs

    @utils.lru_cache
    def _compute_value(self):
        if self._in_logspace:
            return tf.exp(self._variable)
        else:
            return self._variable

    @utils.lru_cache
    def _compute_log_value(self):
        if self._in_logspace:
            return self._variable
        else:
            return tf.log(self._variable)

    def _compute_hard_gd_update(self, grads):
        if len(grads.shape) == 3:
            return tf.reduce_sum(grads, axis=0)
        return grads

    def _compute_hard_em_update(self, counts):
        if len(counts.shape) == 3:
            return tf.reduce_sum(counts, axis=0)
        return counts


def assign_weights(root, value, name=None):
    """Generate an assign operation assigning a value to all the weights in
    the SPN graph rooted in ``root``.

    Args:
        root (Node): The root node of the SPN graph.
        value: The value to assign to the weights.
    """
    assign_ops = []

    def assign(node):
        if isinstance(node, Weights):
            assign_ops.append(node.assign(value))

    with tf.name_scope(name, "AssignWeights", [root, value]):
        # Get all assignment operations
        traverse_graph(root, fun=assign, skip_params=False)

        # Return a collective operation
        return tf.group(*assign_ops)


def initialize_weights(root, name="InitializeWeights"):
    """Generate an assign operation initializing all the sum weights in the SPN
    graph rooted in ``root``.

    Args:
        root (Node): The root node of the SPN graph.
        name: Name of scope to group the weight initializers in
    """
    initialize_ops = []

    def initialize(node):
        if isinstance(node, (Weights, BlockWeights, LocationScaleLeaf)):
            initialize_ops.append(node.initialize())

    with tf.name_scope(name):
        # Get all assignment operations
        traverse_graph(root, fun=initialize, skip_params=False)

        # Return collective operation
        return tf.group(*initialize_ops)
