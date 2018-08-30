# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

from types import MappingProxyType
import tensorflow as tf
from libspn.inference.value import Value, LogValue
from libspn.graph.algorithms import compute_graph_up_down
from libspn.graph.basesum import BaseSum


class Gradient:
    """Assembles TF operations computing gradients of the SPN through
    backpropagation, in a downward pass through the network.

    Args:
        value (Value or LogValue): Pre-computed SPN values.
        value_inference_type (InferenceType): The inference type used during the
            upwards pass through the SPN. Ignored if ``value`` is given.
        log (bool): If ``True``, calculate the value in the log space. Ignored
                    if ``value`` is given.
    """

    def __init__(self, value=None, value_inference_type=None, log=True, dropconnect_keep_prob=None):
        self._true_gradients = {}
        self._actual_gradients = {}
        self._log = log
        self._dropconnect_keep_prob = dropconnect_keep_prob
        # Create internal value generator
        if value is None:
            if log:
                self._value = LogValue(
                    value_inference_type, dropconnect_keep_prob=dropconnect_keep_prob)
            else:
                self._value = Value(
                    value_inference_type, dropconnect_keep_prob=dropconnect_keep_prob)
        else:
            self._value = value
            self._log = value.log()

    @property
    def value(self):
        """Value or LogValue: Computed SPN values."""
        return self._value

    @property
    def gradients(self):
        """dict: Dictionary indexed by node, where each value is a list of tensors
        computing the gradients."""
        return MappingProxyType(self._true_gradients)

    @property
    def actual_gradients(self):
        """dict: Dictionary indexed by node, where each value is a list of tensors
        computing the actual gradients."""
        return MappingProxyType(self._actual_gradients)

    @property
    def log(self):
        return self._log

    def get_gradients(self, root):
        """Assemble TF operations computing the gradients of the SPN rooted in
        ``root``.

        Args:
            root (Node): The root node of the SPN graph.
        """
        def down_fun(node, parent_vals):
            # Sum up all parent vals
            parent_vals = [pv for pv in parent_vals if pv is not None]
            if len(parent_vals) > 1:
                summed = tf.add_n(parent_vals, name=node.name + "_add")
            else:
                summed = parent_vals[0]
            self._true_gradients[node] = summed
            if node.is_op:
                # Compute for inputs
                if isinstance(node, BaseSum):
                    kwargs = dict(dropconnect_keep_prob=self._dropconnect_keep_prob)
                else:
                    kwargs = dict()
                with tf.name_scope(node.name):
                    if self._log:
                        return node._compute_log_gradient(
                            summed, *[self._value.values[i.node]
                                      if i else None
                                      for i in node.inputs], **kwargs)
                    else:
                        return node._compute_log_gradient(
                            summed, *[self._value.values[i.node]
                                      if i else None
                                      for i in node.inputs], **kwargs)

        # Generate values if not yet generated
        if not self._value.values:
            self._value.get_value(root)

        with tf.name_scope("Gradient"):
            # Compute the tensor to feed to the root node
            graph_input = tf.ones_like(self._value.values[root])

            # Traverse the graph computing gradients
            self._true_gradients = {}
            compute_graph_up_down(root, down_fun=down_fun, graph_input=graph_input)

    def get_actual_gradients(self, root):
        """Assemble TF operations computing the actual gradients of the SPN
           rooted in ``root``.

        Args:
            root (Node): The root node of the SPN graph.
        """
        def down_fun(node, parent_vals):
            # Sum up all parent vals
            parent_vals = [pv for pv in parent_vals if pv is not None]
            if len(parent_vals) > 1:
                summed = tf.add_n(parent_vals, name=node.name + "_add")
            else:
                summed = parent_vals[0]
            self._actual_gradients[node] = summed
            if node.is_op:
                if isinstance(node, BaseSum):
                    kwargs = dict(dropconnect_keep_prob=self._dropconnect_keep_prob)
                else:
                    kwargs = dict()
                # Compute for inputs
                with tf.name_scope(node.name):
                    if self._log:
                        return node._compute_log_gradient(
                            summed, *[self._value.values[i.node]
                                      if i else None
                                      for i in node.inputs], **kwargs)
                    else:
                        return node._compute_log_gradient(
                            summed, *[self._value.values[i.node]
                                      if i else None
                                      for i in node.inputs], **kwargs)

        # Generate values if not yet generated
        if not self._value.values:
            self._value.get_value(root)

        with tf.name_scope("Gradient"):
            # Compute the tensor to feed to the root node
            graph_input = tf.ones_like(self._value.values[root])

            # Traverse the graph computing gradients
            self._actual_gradients = {}
            compute_graph_up_down(root, down_fun=down_fun, graph_input=graph_input)
