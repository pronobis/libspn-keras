# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

from collections import namedtuple
import tensorflow as tf

from libspn.graph.distribution import LocationScaleLeaf
from libspn.inference.mpe_path import MPEPath
from libspn.graph.algorithms import traverse_graph
from libspn import conf


class EMLearning():
    """Assembles TF operations performing EM learning of an SPN.

    Args:
        mpe_path (MPEPath): Pre-computed MPE_path.
        value_inference_type (InferenceType): The inference type used during the
            upwards pass through the SPN. Ignored if ``mpe_path`` is given.
        log (bool): If ``True``, calculate the value in the log space. Ignored
                    if ``mpe_path`` is given.
    """

    ParamNode = namedtuple("ParamNode", ["node", "name_scope", "accum"])
    GaussianLeafNode = namedtuple(
        "GaussianLeafNode", ["node", "name_scope", "accum", "sum_data", "sum_data_squared"])

    def __init__(self, root, mpe_path=None, log=True, value_inference_type=None,
                 additive_smoothing=None, add_random=None, initial_accum_value=None,
                 use_unweighted=False, sample=False, sample_prob=None,
                 dropconnect_keep_prob=None, matmul_or_conv=True, accum_decay_factor=None):
        self._root = root
        self._log = log
        self._additive_smoothing = additive_smoothing
        self._initial_accum_value = initial_accum_value
        self._sample = sample
        self._accum_decay_factor = accum_decay_factor
        # Create internal MPE path generator
        if mpe_path is None:
            self._mpe_path = MPEPath(
                log=log, value_inference_type=value_inference_type, add_random=add_random,
                use_unweighted=use_unweighted, sample=sample, sample_prob=sample_prob,
                dropconnect_keep_prob=dropconnect_keep_prob, matmul_or_conv=matmul_or_conv)
        else:
            self._mpe_path = mpe_path
        # Create a name scope
        with tf.name_scope("EMLearning") as self._name_scope:
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
            return tf.group(*(
                    [pn.accum.initializer for pn in self._param_nodes] +
                    [dn.accum.initializer for dn in self._gaussian_leaf_nodes] +
                    [dn.sum_data.initializer for dn in self._gaussian_leaf_nodes] +
                    [dn.sum_data_squared.initializer for dn in self._gaussian_leaf_nodes] +
                    [dn.node._total_count_variable.initializer
                     for dn in self._gaussian_leaf_nodes]),
                            name="reset_accumulators")

    def accumulate_updates(self):
        # Generate path if not yet generated
        if not self._mpe_path.counts:
            self._mpe_path.get_mpe_path(self._root)

        # Generate all accumulate operations
        with tf.name_scope(self._name_scope):
            assign_ops = []
            for pn in self._param_nodes:
                with tf.name_scope(pn.name_scope):
                    # counts = self._mpe_path.counts[pn.node]
                    # update_value = pn.node._compute_hard_em_update(counts)
                    # with tf.control_dependencies([update_value]):
                    # op = tf.assign_add(pn.accum, self._mpe_path.counts[pn.node])
                    counts_summed_batch = pn.node._compute_hard_em_update(
                        self._mpe_path.counts[pn.node])
                    if self._accum_decay_factor is not None and self._accum_decay_factor != 0.0:
                        decayed = tf.maximum(
                            pn.accum + counts_summed_batch - self._accum_decay_factor,
                            self._initial_accum_value)
                        assign_ops.append(tf.assign(pn.accum, decayed))
                    else:
                        assign_ops.append(tf.assign_add(pn.accum, counts_summed_batch))

            for dn in self._gaussian_leaf_nodes:
                with tf.name_scope(dn.name_scope):
                    counts = self._mpe_path.counts[dn.node]
                    update_value = dn.node._compute_hard_em_update(counts)
                    with tf.control_dependencies(update_value.values()):
                        if dn.node.dimensionality > 1:
                            accum = tf.squeeze(update_value['accum'], axis=-1)
                        else:
                            accum = update_value['accum']
                        assign_ops.extend(
                            [tf.assign_add(dn.accum, accum),
                             tf.assign_add(dn.sum_data, update_value['sum_data']),
                             tf.assign_add(
                                 dn.sum_data_squared, update_value['sum_data_squared'])])

            return tf.group(*assign_ops, name="accumulate_updates")

    def update_spn(self):
        # Generate all update operations
        with tf.name_scope(self._name_scope):
            assign_ops = []
            for pn in self._param_nodes:
                with tf.name_scope(pn.name_scope):
                    accum = pn.accum
                    if self._additive_smoothing is not None:
                        accum = tf.add(accum, self._additive_smoothing)
                    if pn.node.log:
                        assign_ops.append(pn.node.assign_log(tf.log(accum)))
                    else:
                        assign_ops.append(pn.node.assign(accum))

            for dn in self._gaussian_leaf_nodes:
                with tf.name_scope(dn.name_scope):
                    assign_ops.extend(dn.node.assign(dn.accum, dn.sum_data, dn.sum_data_squared))

            return tf.group(*assign_ops, name="update_spn")

    def learn(self):
        """Assemble TF operations performing EM learning of the SPN."""
        return None

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
                                                collections=['em_accumulators'])
                        else:
                            accum = tf.Variable(tf.ones_like(node.variable,
                                                             dtype=conf.dtype) *
                                                self._initial_accum_value,
                                                dtype=conf.dtype,
                                                collections=['em_accumulators'])
                    else:
                        accum = tf.Variable(tf.zeros_like(node.variable,
                                                          dtype=conf.dtype),
                                            dtype=conf.dtype,
                                            collections=['em_accumulators'])
                    param_node = EMLearning.ParamNode(node=node, accum=accum,
                                                      name_scope=scope)
                    self._param_nodes.append(param_node)
            if isinstance(node, (LocationScaleLeaf)) \
                    and node.trainable:
                shape = (node.num_vars, node.num_components)
                with tf.name_scope(node.name) as scope:
                    if self._initial_accum_value is not None:
                        accum = tf.Variable(tf.ones(shape, dtype=conf.dtype) *
                                            self._initial_accum_value,
                                            dtype=conf.dtype,
                                            collections=['em_accumulators'])
                        sum_x = tf.Variable(node.loc_variable * self._initial_accum_value,
                                            dtype=conf.dtype, collections=['em_accumulators'])
                        sum_x2 = tf.Variable(tf.square(node.loc_variable) *
                                             self._initial_accum_value,
                                             dtype=conf.dtype, collections=['em_accumulators'])
                    else:
                        accum = tf.Variable(tf.zeros(shape, dtype=conf.dtype),
                                            dtype=conf.dtype,
                                            collections=['em_accumulators'])
                        sum_x = tf.Variable(tf.zeros_like(node.loc_variable), dtype=conf.dtype,
                                            collections=['em_accumulators'])
                        sum_x2 = tf.Variable(tf.zeros_like(node.loc_variable), dtype=conf.dtype,
                                             collections=['em_accumulators'])
                    gaussian_node = EMLearning.GaussianLeafNode(
                        node=node, accum=accum, sum_data=sum_x, sum_data_squared=sum_x2,
                        name_scope=scope)
                    self._gaussian_leaf_nodes.append(gaussian_node)

        self._gaussian_leaf_nodes = []
        self._param_nodes = []
        with tf.name_scope(self._name_scope):
            traverse_graph(self._root, fun=fun)