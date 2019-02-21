# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

from collections import namedtuple, OrderedDict
import tensorflow as tf
from libspn.inference.mpe_path import MPEPath
from libspn.inference.value import LogValue
from libspn.graph.algorithms import traverse_graph
from libspn.graph.basesum import BaseSum
from libspn.graph.weights import Weights
from libspn.graph.distribution import NormalLeaf


class SoftEMLearning():
    """Assembles TF operations performing EM learning of an SPN.

    Args:
        mpe_path (MPEPath): Pre-computed MPE_path.
        value_inference_type (InferenceType): The inference type used during the
            upwards pass through the SPN. Ignored if ``mpe_path`` is given.
        log (bool): If ``True``, calculate the value in the log space. Ignored
                    if ``mpe_path`` is given.
    """

    def __init__(self, root, additive_smoothing=1e-8, dropconnect_keep_prob=None, val_gen=None):
        self._root = root
        self._additive_smoothing = additive_smoothing
        self._val_gen = val_gen or LogValue(dropconnect_keep_prob=dropconnect_keep_prob)
        self._root_val = self._val_gen.get_value(self._root)
        # Create a name scope
        with tf.name_scope("SoftEMLearning") as self._name_scope:
            pass

    @property
    def value(self):
        """Value or LogValue: Computed SPN values."""
        return self._val_gen

    def update_spn(self):
        # Generate all update operations
        with tf.name_scope(self._name_scope):
            # w_vars = OrderedDict()
            #
            # def _gather_weights(node):
            #     if node.is_param:
            #         w_vars[node] = node.variable
            #
            # traverse_graph(self._root, _gather_weights)
            # w_grads = tf.gradients(self._root_val, list(w_vars.values()))

            # sum_nodes = [n for n in self._val_gen.values.keys() if isinstance(n, BaseSum)]
            weight_nodes, weight_vars, log_weights = zip(*[
                (n, n.variable, log_w) for n, log_w in self._val_gen.values.items()
                if isinstance(n, Weights)
            ])
            cont_vars = [n for n in self._val_gen.values.keys()
                         if isinstance(n, NormalLeaf)]
            num_w = len(weight_vars)

            # sum_tensors = [self._val_gen.values[n] for n in sum_nodes]
            # child_tensors = [self._val_gen.values[n.values[0].node] for n in sum_nodes]
            # weight_tensors = [self._val_gen.values[n] for n in weight_nodes]
            w_and_d_grads = tf.gradients(
                self._root_val, list(log_weights) + [self._val_gen.values[n] for n in cont_vars])
            w_grads = w_and_d_grads[:num_w]
            d_grads = w_and_d_grads[num_w:]

            accumulators_w = [tf.Variable(tf.ones_like(w) * 1e-2) for w in weight_vars]

            accumulators_x = [tf.Variable(
                initial_value=n.loc_variable.initial_value) for n in cont_vars]
            accumulators_v = [tf.Variable(
                initial_value=tf.square(n.scale_variable.initial_value)) for n in cont_vars]
            accumulators_p = [tf.Variable(tf.ones_like(n.loc_variable) * 1e-2) for n in cont_vars]

            # dR/dS = dR/dlogS * dlogS/dS ==> dR/dlogS = S * dR/dS
            # dlogR/dlogS = 1/R dR/dlogS ==> S * dR/dS == R * dlogR/dlogS
            # dR/dS == R/S dlogR/dlogS
            # sum_tensors = []

            acc_update = [tf.assign_add(a, wg) for a, wg in zip(accumulators_w[:num_w], w_grads)]

            for n, dg, a_x, a_p, a_v in zip(cont_vars, d_grads, accumulators_x, accumulators_p,
                                            accumulators_v):
                current_mode = tf.reshape(tf.tile(
                    tf.expand_dims(n.loc_variable, 0), [tf.shape(n.feed)[0], 1, 1]),
                    [-1, n.num_vars * n.num_components])
                x = tf.reshape(
                    n._evidence_mask(n._preprocessed_feed(), lambda *_: current_mode),
                    [-1, n.num_vars, n.num_components])
                dg = tf.reshape(dg, [-1, n.num_vars, n.num_components])
                dg /= tf.reduce_sum(dg, axis=-1, keepdims=True)

                acc_update.append(tf.assign_add(a_x, tf.reduce_sum(dg * x, axis=0)))
                acc_update.append(tf.assign_add(a_p, tf.reduce_sum(dg, axis=0)))
                acc_update.append(tf.assign_add(a_v, tf.reduce_sum(
                    dg * tf.square(x - tf.reshape(current_mode, [-1, n.num_vars, n.num_components])), axis=0)))

            with tf.control_dependencies(acc_update):
                accumulators_decayed = [tf.maximum(acc - 1e-2, 1e-8) for acc in accumulators_w]
                return tf.group(
                    *(
                        [tf.assign(w, acc / tf.reduce_sum(acc, axis=-1, keepdims=True))
                         for w, acc in zip(weight_vars, accumulators_decayed)] +
                        [tf.assign(n.loc_variable, a_x / a_p)
                         for n, a_x, a_p in zip(cont_vars, accumulators_x, accumulators_p)] +
                        [tf.assign(n.scale_variable, tf.sqrt(a_v / a_p))
                         for n, a_v, a_p in zip(cont_vars, accumulators_v, accumulators_p)]
                    )
                )

    def learn(self):
        """Assemble TF operations performing EM learning of the SPN."""
        return None