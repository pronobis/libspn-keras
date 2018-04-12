# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

from collections import namedtuple
import tensorflow as tf
from libspn.inference.mpe_path import MPEPath
from libspn.graph.algorithms import traverse_graph
from libspn import conf


class GDLearning():
    """Assembles TF operations performing Gradient Descent learning of an SPN.

    Args:
        log (bool): If ``True``, calculate the value in the log space. Ignored
                    if ``mpe_path`` is given.
        value_inference_type (InferenceType): The inference type used during the
            upwards pass through the SPN. Ignored if ``mpe_path`` is given.
        learning_rate (float): Learning rate parameter used for updating SPN weights.
    """
    ParamNode = namedtuple("ParamNode", ["node", "name_scope", "accum"])

    def __init__(self, root, mpe_path=None, log=True, value_inference_type=None,
                 learning_rate=0.1, additive_smoothing=None, add_random=None,
                 initial_accum_value=None, use_unweighted=False):
        self._root = root
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be a positive number")
        else:
            self._learning_rate = learning_rate
        self._additive_smoothing = additive_smoothing
        self._initial_accum_value = initial_accum_value
        # Create internal MPE path generator
        if mpe_path is None:
            self._mpe_path = MPEPath(log=log,
                                     value_inference_type=value_inference_type,
                                     add_random=add_random,
                                     use_unweighted=use_unweighted)
        else:
            self._mpe_path = mpe_path
        # Create a name scope
        with tf.name_scope("GDLearning") as self._name_scope:
            pass
        # Create accumulators
        self._create_accumulators()

    @property
    def mpe_path(self):
        """MPEPath: Computed MPE path."""
        return self._mpe_path

    @property
    def value(self):
        """Value or LogValue: Computed SPN values."""
        return self._mpe_path.value

    # TODO: For testing only
    def root_accum(self):
        for pn in self._param_nodes:
            if pn.node == self._root.weights.node:
                return pn.accum
        return None

    def reset_accumulators(self):
        with tf.name_scope(self._name_scope):
            return tf.group(*[pn.accum.initializer for pn in self._param_nodes],
                            name="reset_accumulators")

    def accumulate_updates(self):
        # Generate path if not yet generated
        if not self._mpe_path.counts:
            self._mpe_path.get_mpe_path(self._root)

        if not self._mpe_path.actual_counts:
            self._mpe_path.get_mpe_path_actual(self._root)

        # Generate all accumulate operations
        with tf.name_scope(self._name_scope):
            assign_ops = []
            for pn in self._param_nodes:
                with tf.name_scope(pn.name_scope):
                    counts = self._mpe_path.counts[pn.node]
                    actual_counts = self._mpe_path.actual_counts[pn.node]
                    update_value = pn.node._compute_hard_gd_update(counts,
                                                                   actual_counts)
                    # Apply learning-rate
                    update_value *= self._learning_rate
                    op = tf.assign_add(pn.accum, update_value)
                    assign_ops.append(op)

            return tf.group(*assign_ops, name="accumulate_updates")

    def update_spn(self):
        # Generate all update operations
        with tf.name_scope(self._name_scope):
            assign_ops = []
            for pn in self._param_nodes:
                with tf.name_scope(pn.name_scope):
                    # Readjust accumulator values to be >= 0
                    # accum: accum - min(accum)
                    accum = tf.subtract(pn.accum, tf.reduce_min(pn.accum, axis=-1,
                                                                keep_dims=True))
                    # Apply addtivie-smooting
                    if self._additive_smoothing is not None:
                        accum = tf.add(accum, self._additive_smoothing)
                    # Assign accumulators to respective weights
                    assign_ops.append(pn.node.assign(accum))
            return tf.group(*assign_ops, name="update_spn")

    def _create_accumulators(self):
        def fun(node):
            if node.is_param:
                with tf.name_scope(node.name) as scope:
                    if self._initial_accum_value is not None:
                        if node.mask and not all(node.mask):
                            accum = tf.Variable(tf.cast(tf.reshape(node.mask,
                                                node.variable.shape),
                                                dtype=conf.dtype) *
                                                self._initial_accum_value,
                                                dtype=conf.dtype,
                                                collections=['gd_accumulators'])
                        else:
                            accum = tf.Variable(tf.ones_like(node.variable,
                                                             dtype=conf.dtype) *
                                                self._initial_accum_value,
                                                dtype=conf.dtype,
                                                collections=['gd_accumulators'])
                    else:
                        accum = tf.Variable(tf.zeros_like(node.variable,
                                                          dtype=conf.dtype),
                                            dtype=conf.dtype,
                                            collections=['gd_accumulators'])
                    param_node = GDLearning.ParamNode(node=node, accum=accum,
                                                      name_scope=scope)
                    self._param_nodes.append(param_node)

        self._param_nodes = []
        with tf.name_scope(self._name_scope):
            traverse_graph(self._root, fun=fun)

    def learn(self):
        """Assemble TF operations performing gradient descent learning of the SPN."""
        return None
