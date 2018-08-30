# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

import tensorflow as tf
from libspn.graph.node import ParamNode
from libspn.graph.algorithms import traverse_graph
from libspn import conf
from libspn.graph.distribution import GaussianLeaf
from libspn.utils.serialization import register_serializable
from libspn import utils
from libspn.exceptions import StructureError

import numbers


@register_serializable
class Weights(ParamNode):
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

    def __init__(self, init_value=1, num_weights=1, num_sums=1, log=False,
                 trainable=True, mask=None, name="Weights"):
        if not isinstance(num_weights, int) or num_weights < 1:
            raise ValueError("num_weights must be a positive integer")

        if not isinstance(num_sums, int) or num_sums < 1:
            raise ValueError("num_sums must be a positive integer")

        self._init_value = init_value
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
        data['init_value'] = self._init_value
        data['value'] = self._variable
        data['mask'] = self._mask
        return data

    def deserialize(self, data):
        self._init_value = data['init_value']
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
            value: The value to assign to the weights. For possible values, see
                   :meth:`~libspn.utils.broadcast_value`.

        Returns:
            Tensor: The assignment operation.
        """
        if self._log:
            raise StructureError("Trying to assign non-log values to log-weights.")

        if isinstance(value, utils.ValueType.RANDOM_UNIFORM) \
           or isinstance(value, numbers.Real):
            shape = self._num_sums * self._num_weights
        else:
            shape = self._num_weights
        value = utils.broadcast_value(value, (shape,), dtype=conf.dtype)
        value = tf.where(tf.is_nan(value), tf.ones_like(value) * 0.01, value)
        if self._mask and not all(self._mask):
            # Only perform masking if mask is given and mask contains any 'False'
            value *= tf.cast(tf.reshape(self._mask, value.shape), dtype=conf.dtype)
        value = utils.normalize_tensor_2D(value, self._num_weights, self._num_sums)
        return tf.assign(self._variable, value)

    def assign_log(self, value):
        """Return a TF operation assigning log values to the weights.

        Args:
            value: The value to assign to the weights. For possible values, see
                   :meth:`~libspn.utils.broadcast_value`.

        Returns:
            Tensor: The assignment operation.
        """
        if not self._log:
            raise StructureError("Trying to assign log values to non-log weights.")

        if isinstance(value, utils.ValueType.RANDOM_UNIFORM) \
           or isinstance(value, numbers.Real):
            shape = self._num_sums * self._num_weights
        else:
            shape = self._num_weights
        value = utils.broadcast_value(value, (shape,), dtype=conf.dtype)
        value = tf.where(tf.is_nan(value), tf.log(tf.ones_like(value) * 0.01), value)
        if self._mask and not all(self._mask):
            # Only perform masking if mask is given and mask contains any 'False'
            value += tf.log(tf.cast(tf.reshape(self._mask, value.shape), dtype=conf.dtype))
        normalized_value = \
            utils.normalize_log_tensor_2D(value, self._num_weights, self._num_sums)
        return tf.assign(self._variable, normalized_value)

    def normalize(self, value=None, name="Normalize"):
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
                if self._mask and not all(self._mask):
                    # Only perform masking if mask is given and mask contains any 'False'
                    value += tf.log(tf.cast(tf.reshape(self._mask, value.shape), dtype=conf.dtype))
                return tf.assign(self._variable, value - tf.reduce_logsumexp(
                    value, axis=-1, keepdims=True))
            else:
                value = tf.maximum(value, 1e-6)
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
        normalized_value = \
            utils.normalize_log_tensor_2D(update_value, self._num_weights, self._num_sums)
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
        normalized_value = \
            utils.normalize_log_tensor_2D(update_value, self._num_weights, self._num_sums)
        return tf.assign(self._variable, normalized_value)

    def _create(self):
        """Creates a TF variable holding the vector of the SPN weights.

        Returns:
            Variable: A TF variable of shape ``[num_weights]``.
        """
        if isinstance(self._init_value, utils.ValueType.RANDOM_UNIFORM) \
           or isinstance(self._init_value, numbers.Real):
            shape = self._num_sums * self._num_weights
        else:
            shape = self._num_weights
        init_val = utils.broadcast_value(self._init_value,
                                         shape=(shape,),
                                         dtype=conf.dtype)
        if self._mask and not all(self._mask):
            # Only perform masking if mask is given and mask contains any 'False'
            init_val *= tf.cast(tf.reshape(self._mask, init_val.shape), dtype=conf.dtype)
        init_val = utils.normalize_tensor_2D(init_val, self._num_weights, self._num_sums)
        if self._log:
            init_val = tf.log(init_val)
        self._variable = tf.Variable(init_val, dtype=conf.dtype,
                                     collections=['spn_weights'])

    def _compute_out_size(self):
        return self._num_weights * self._num_sums

    @utils.lru_cache
    def _compute_value(self):
        if self._log:
            return tf.exp(self._variable)
        else:
            return self._variable

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


def assign_weights(root, value, name=None):
    """Generate an assign operation assigning a value to all the weights in
    the SPN graph rooted in ``root``.

    Args:
        root (Node): The root node of the SPN graph.
        value: The value to assign to the weights. For possible values, see
               :meth:`~libspn.utils.broadcast_value`.
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
        if isinstance(node, Weights):
            initialize_ops.append(node.initialize())
        if isinstance(node, GaussianLeaf):
            initialize_ops.extend(node.initialize())

    with tf.name_scope(name):
        # Get all assignment operations
        traverse_graph(root, fun=initialize, skip_params=False)

        # Return collective operation
        return tf.group(*initialize_ops)
