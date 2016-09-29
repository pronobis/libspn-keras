# ------------------------------------------------------------------------
# Copyright (C) 2016 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

from types import MappingProxyType
import tensorflow as tf
from libspn.inference.value import Value, LogValue
from libspn.graph.algorithms import compute_graph_up_down


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
                 use_unweighted=False):
        self._counts = {}
        self._log = log
        self._add_random = add_random
        self._use_unweighted = use_unweighted
        # Create internal value generator
        if value is None:
            if log:
                self._value = LogValue(value_inference_type)
            else:
                self._value = Value(value_inference_type)
        else:
            self._value = value

    @property
    def value(self):
        """Value or LogValue: Computed SPN values."""
        return self._value

    @property
    def counts(self):
        """dict: Dictionary indexed by node, where each value is a lists of
        tensors computing the branch counts for the inputs of the node."""
        return MappingProxyType(self._counts)

    def get_mpe_path(self, root):
        """Assemble TF operations computing the branch counts for the MPE
        downward path through the SPN rooted in ``root``.

        Args:
            root (Node): The root node of the SPN graph.
        """
        def down_fun(node, parent_vals):
            # Sum up all parent vals
            if len(parent_vals) > 1:
                summed = tf.add_n(parent_vals, name=node.name + "_add")
            else:
                summed = parent_vals[0]
            self._counts[node] = summed
            if node.is_op:
                # Compute for inputs
                with tf.name_scope(node.name):
                    if self._log:
                        return node._compute_log_mpe_path(
                            summed, *[self._value.values[i.node]
                                      if i else None
                                      for i in node.inputs],
                            add_random=self._add_random,
                            use_unweighted=self._use_unweighted)
                    else:
                        return node._compute_mpe_path(
                            summed, *[self._value.values[i.node]
                                      if i else None
                                      for i in node.inputs],
                            add_random=self._add_random,
                            use_unweighted=self._use_unweighted)

        # Generate values if not yet generated
        if not self._value.values:
            self._value.get_value(root)

        with tf.name_scope("MPEPath"):
            # Compute the tensor to feed to the root node
            graph_input = tf.ones_like(self._value.values[root])

            # Traverse the graph computing counts
            self._counts = {}
            compute_graph_up_down(root, down_fun=down_fun, graph_input=graph_input)
