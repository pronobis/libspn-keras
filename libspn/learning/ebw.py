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


class ExtendedBaumWelch:
    """Assembles TF operations performing Extended Baum Welch learning.

    Args:
        mpe_path (MPEPath): Pre-computed MPE_path.
        value_inference_type (InferenceType): The inference type used during the
            upwards pass through the SPN. Ignored if ``mpe_path`` is given.
        log (bool): If ``True``, calculate the value in the log space. Ignored
                    if ``mpe_path`` is given.
    """

    def __init__(self, root, root_cond, additive_smoothing=1e-8,
                 dropconnect_keep_prob=None, val_gen=None, decay=1e-1):
        self._root = root
        self._root_cond = root_cond
        self._additive_smoothing = additive_smoothing
        self._val_gen = val_gen or LogValue(dropconnect_keep_prob=dropconnect_keep_prob)
        self._root_val = self._val_gen.get_value(self._root)
        self._root_cond_val = self._val_gen.get_value(self._root_cond)
        self._decay = decay
        # Create a name scope
        with tf.name_scope("ExtendedBaumWelch") as self._name_scope:
            pass

    @property
    def value(self):
        """Value or LogValue: Computed SPN values."""
        return self._val_gen

    def update_spn(self):
        # Generate all update operations
        with tf.name_scope(self._name_scope):

            weight_nodes, weight_vars, log_weights = zip(*[
                (n, n.variable, log_w) for n, log_w in self._val_gen.values.items()
                if isinstance(n, Weights)
            ])
            cont_vars = [n for n in self._val_gen.values.keys()
                         if isinstance(n, NormalLeaf)]
            num_w = len(weight_vars)

            w_and_d_grads = tf.gradients(
                self._root_cond_val - self._root_val,
                list(weight_vars) + [self._val_gen.values[n] for n in cont_vars])
            w_grads = w_and_d_grads[:num_w]
            d_grads = w_and_d_grads[num_w:]

            new_vals = [tf.maximum((w_g * w + self._decay * w)
                        / (tf.reduce_sum(w * w_g, axis=-1, keepdims=True) + self._decay), 1e-10)
                        for w, w_g in zip(weight_vars, w_grads)]
            updates = [tf.assign(w, nv) for w, nv in zip(weight_vars, new_vals)]
            # for n, dg in zip(cont_vars, d_grads):
            #     current_mode = tf.reshape(tf.tile(
            #         tf.expand_dims(n.loc_variable, 0), [tf.shape(n.feed)[0], 1, 1]),
            #         [-1, n.num_vars * n.num_components])
            #     x = tf.reshape(
            #         n._evidence_mask(n._preprocessed_feed(), lambda *_: current_mode),
            #         [-1, n.num_vars, n.num_components])
            #     dg = tf.reshape(dg, [-1, n.num_vars, n.num_components])
            #
            #     denom = tf.reduce_sum(dg, axis=0)
            #
            #     mu_new = tf.reduce_sum(dg * x, axis=0) + self._decay * n.loc_variable
            #     mu_new /= denom + self._decay
            #
            #     sigma2 = tf.reduce_sum(dg * tf.square(x), axis=0) + (
            #         tf.square(n.loc_variable) + tf.square(n.scale_variable))
            #     sigma2 /= denom + self._decay
            #     sigma2 -= tf.square(mu_new)
            #
            #     sigma = tf.maximum(tf.sqrt(sigma2), 0.01)
            #     updates.append(tf.assign(n.loc_variable, mu_new))
            #     updates.append(tf.assign(n.scale_variable, sigma))

            return tf.group(*updates)

    def learn(self):
        """Assemble TF operations performing EM learning of the SPN."""
        return None