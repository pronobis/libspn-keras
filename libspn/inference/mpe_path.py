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


class MPEPath:
    """Assembles TF operations computing the branch counts for the MPE downward
    path through the SPN. It computes the number of times each branch was
    traveled by a complete subcircuit determined by the MPE value of the latent
    variables in the model.

    Args:
        value (Value or LogValue): Pre-computed SPN values.
        value_inference_type (InferenceType): The inference type used during the
            upwards pass through the SPN. Ignored if ``value`` is given.
        log (bool): If ``True``, calculate the value in the log space. Ignored
                    if ``value`` is given.
    """

    def __init__(self, value=None, value_inference_type=None, log=True, add_random=None,
                 use_unweighted=False, sample=False, sample_prob=None,
                 dropconnect_keep_prob=None):
        self._true_counts = {}
        self._actual_counts = {}
        self._log = log
        self._add_random = add_random
        self._use_unweighted = use_unweighted
        self._sample = sample
        self._sample_prob = sample_prob
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
    def counts(self):
        """dict: Dictionary indexed by node, where each value is a list of tensors
        computing the branch counts, based on the true value of the SPN's latent
        variable, for the inputs of the node."""
        return MappingProxyType(self._true_counts)

    @property
    def actual_counts(self):
        """dict: Dictionary indexed by node, where each value is a list of tensors
        computing the branch counts, based on the actual value calculated by the
        SPN, for the inputs of the node."""
        return MappingProxyType(self._actual_counts)

    @property
    def log(self):
        return self._log

    def get_mpe_path(self, root):
        """Assemble TF operations computing the true branch counts for the MPE
        downward path through the SPN rooted in ``root``.

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
            self._true_counts[node] = summed
            basesum_kwargs = dict(
                add_random=self._add_random, use_unweighted=self._use_unweighted,
                sample=self._sample, sample_prob=self._sample_prob)
            if node.is_op:
                kwargs = basesum_kwargs if isinstance(node, BaseSum) else dict()
                # Compute for inputs
                with tf.name_scope(node.name):
                    if self._log:
                        return node._compute_log_mpe_path(
                            summed, *[self._value.values[i.node] if i else None
                                      for i in node.inputs], **kwargs)
                    else:
                        return node._compute_mpe_path(
                            summed, *[self._value.values[i.node] if i else None
                                      for i in node.inputs], **kwargs)

        # Generate values if not yet generated
        if not self._value.values:
            self._value.get_value(root)

        with tf.name_scope("TrueMPEPath"):
            # Compute the tensor to feed to the root node
            graph_input = tf.ones_like(self._value.values[root])

            # Traverse the graph computing counts
            self._true_counts = {}
            compute_graph_up_down(root, down_fun=down_fun, graph_input=graph_input)

    def get_mpe_path_actual(self, root):
        """Assemble TF operations computing the actual branch counts for the MPE
        downward path through the SPN rooted in ``root``.

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
            self._actual_counts[node] = summed
            basesum_kwargs = dict(
                add_random=self._add_random, use_unweighted=self._use_unweighted,
                sample=self._sample, sample_prob=self._sample_prob)
            if node.is_op:
                # Compute for inputs
                kwargs = basesum_kwargs if isinstance(node, BaseSum) else dict()
                with tf.name_scope(node.name):
                    if self._log:
                        return node._compute_log_mpe_path(
                            summed, *[self._value.values[i.node] if i else None
                                      for i in node.inputs], **kwargs)
                    else:
                        return node._compute_mpe_path(
                            summed, *[self._value.values[i.node] if i else None
                                      for i in node.inputs], **kwargs)

        # Generate values if not yet generated
        if not self._value.values:
            self._value.get_value(root)

        with tf.name_scope("ActualMPEPath"):
            graph_input = tf.ones_like(self._value.values[root])

            # Traverse the graph computing counts
            self._actual_counts = {}
            compute_graph_up_down(root, down_fun=down_fun, graph_input=graph_input)
