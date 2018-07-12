# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

from collections import namedtuple
import tensorflow as tf
from libspn.inference.mpe_path import MPEPath
from libspn.inference.gradient import Gradient
from libspn.graph.algorithms import traverse_graph
from libspn.learning.type import LearningType
from libspn.learning.type import LearningInferenceType
from libspn import conf
from libspn.graph.distribution import GaussianLeaf


class GDLearning:
    """Assembles TF operations performing Gradient Descent learning of an SPN.

    Args:
        log (bool): If ``True``, calculate the value in the log space. Ignored
                    if ``mpe_path`` is given.
        value_inference_type (InferenceType): The inference type used during the
            upwards pass through the SPN. Ignored if ``mpe_path`` is given.
        learning_rate (float): Learning rate parameter used for updating SPN weights.
        learning_type (LearningType): Learning type used while learning.
        learning_inference_type (LearningInferenceType): Learning inference type
            used while learning.
    """
    ParamNode = namedtuple("ParamNode", ["node", "name_scope", "accum"])
    GaussianLeafNode = namedtuple(
        "GaussianLeafNode", ["node", "name_scope", "mean_grad", "var_grad"])

    def __init__(self, root, mpe_path=None, gradient=None, learning_rate=0.001,
                 log=True, value_inference_type=None,
                 learning_type=LearningType.DISCRIMINATIVE,
                 learning_inference_type=LearningInferenceType.HARD,
                 add_random=None, use_unweighted=False, dropconnect_keep_prob=None,
                 sample_path=False, sample_prob=None, dropout_keep_prob=None):
        self._root = root
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be a positive number")
        else:
            self._learning_rate = learning_rate
        self._log = log
        self._learning_type = learning_type
        self._learning_inference_type = learning_inference_type
        if self._learning_inference_type == LearningInferenceType.HARD:
            self._gradient = None
            # Create internal MPE path generator
            if mpe_path is None:
                self._mpe_path = MPEPath(
                    log=log, value_inference_type=value_inference_type, add_random=add_random,
                    use_unweighted=use_unweighted, sample_prob=sample_prob, sample=sample_path,
                    dropout_keep_prob=dropout_keep_prob,
                    dropconnect_keep_prob=dropconnect_keep_prob)
            else:
                self._mpe_path = mpe_path
                self._log = mpe_path.log
        else:
            self._mpe_path = None
            # Create internal gradient generator
            if gradient is None:
                self._gradient = \
                    Gradient(log=log, value_inference_type=value_inference_type,
                             dropout_keep_prob=dropout_keep_prob,
                             dropconnect_keep_prob=dropconnect_keep_prob)
            else:
                self._gradient = gradient
                self._log = gradient.log
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
    def gradient(self):
        """Gradient: Computed gradients."""
        return self._gradient

    @property
    def value(self):
        """Value or LogValue: Computed SPN values."""
        if self._learning_inference_type == LearningInferenceType.HARD:
            return self._mpe_path.value
        else:
            return self._gradient.value

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
                [gn.mean_grad.initializer for gn in self._gaussian_leaf_nodes] +
                [gn.var_grad.initializer for gn in self._gaussian_leaf_nodes]
            ), name="reset_accumulators")

    def accumulate_updates(self):

        if self._learning_inference_type == LearningInferenceType.HARD:
            # Generate path if not yet generated
            if not self._mpe_path.counts:
                self._mpe_path.get_mpe_path(self._root)

            if self._learning_type == LearningType.DISCRIMINATIVE and \
               not self._mpe_path.actual_counts:
                self._mpe_path.get_mpe_path_actual(self._root)

            # Left hand side of gradient
            positive_grad_table = self._mpe_path.counts
            # Right hand side of gradient (when using discriminative learning)
            negative_grad_table = self._mpe_path.actual_counts
        else:
            # Generate gradients if not yet generated
            if not self._gradient.gradients:
                self._gradient.get_gradients(self._root)

            if self._learning_type == LearningType.DISCRIMINATIVE and \
               not self._gradient.actual_gradients:
                self._gradient.get_actual_gradients(self._root)

            # Left hand side of the gradient
            positive_grad_table = self._gradient.gradients
            # Right hand side of the gradient (when using discriminative learning)
            negative_grad_table = self._gradient.actual_gradients

        # Generate all accumulate operations
        with tf.name_scope(self._name_scope):
            assign_ops = []
            for pn in self._param_nodes:
                with tf.name_scope(pn.name_scope):
                    incoming_grad = positive_grad_table[pn.node]
                    if self._learning_type == LearningType.DISCRIMINATIVE:
                        incoming_grad -= negative_grad_table[pn.node]
                    grad_batch_summed = pn.node._compute_hard_gd_update(incoming_grad)
                    assign_ops.append(tf.assign_sub(pn.accum, grad_batch_summed))

            for gn in self._gaussian_leaf_nodes:
                with tf.name_scope(gn.name_scope):
                    incoming_grad = positive_grad_table[gn.node]
                    if self._learning_type == LearningType.DISCRIMINATIVE:
                        incoming_grad -= negative_grad_table[gn.node]
                    mean_grad, var_grad = gn.node._compute_gradient(incoming_grad)
                    assign_ops.append(tf.assign_add(gn.mean_grad, mean_grad))
                    assign_ops.append(tf.assign_add(gn.var_grad, var_grad))

            return tf.group(*assign_ops, name="accumulate_updates")

    def update_spn(self, optimizer=None):
        if not optimizer:
            raise ValueError("No Optimizer provide for updating SPN")
        # Generate all update operations
        with tf.name_scope(self._name_scope):
            apply_grad_op = \
                optimizer(self._learning_rate).apply_gradients(self._grads_and_vars)

            # After applying gradients to weights, normalize weights
            with tf.control_dependencies([apply_grad_op]):
                weight_norm_ops = []

                def fun(node):
                    if node.is_param:
                        if node.log:
                            weight_norm_ops.append(node.assign_log(node.variable))
                        else:
                            weight_norm_ops.append(node.assign(node.variable))

                    if isinstance(node, GaussianLeaf) and node.learn_distribution_parameters:
                        weight_norm_ops.append(tf.assign(node.scale_variable, tf.maximum(
                            node.scale_variable, node._min_stddev)))
                with tf.name_scope("Weight_Normalization"):
                    traverse_graph(self._root, fun=fun)
            return tf.group(*weight_norm_ops, name="weight_norm")

    def _create_accumulators(self):
        def fun(node):
            if node.is_param:
                with tf.name_scope(node.name) as scope:
                    accum = tf.Variable(tf.zeros_like(node.variable, dtype=conf.dtype),
                                        dtype=conf.dtype, collections=['gd_accumulators'])
                    param_node = GDLearning.ParamNode(node=node, accum=accum,
                                                      name_scope=scope)
                    self._param_nodes.append(param_node)
                    self._grads_and_vars.append((accum, node.variable))

            if isinstance(node, GaussianLeaf) and node.learn_distribution_parameters:
                with tf.name_scope(node.name) as scope:
                    mean_grad_accum = tf.Variable(
                        tf.zeros_like(node.loc_variable, dtype=conf.dtype),
                        dtype=conf.dtype, collections=['gd_accumulators'])
                    variance_grad_accum = tf.Variable(
                        tf.zeros_like(node.scale_variable, dtype=conf.dtype),
                        dtype=conf.dtype, collections=['gd_accumulators'])
                    gauss_leaf_node = GDLearning.GaussianLeafNode(
                        node=node, mean_grad=mean_grad_accum, var_grad=variance_grad_accum,
                        name_scope=scope)
                    self._gaussian_leaf_nodes.append(gauss_leaf_node)

        self._param_nodes = []
        self._grads_and_vars = []
        self._gaussian_leaf_nodes = []
        with tf.name_scope(self._name_scope):
            traverse_graph(self._root, fun=fun)

    def learn(self, loss=None, optimizer=tf.train.AdamOptimizer):
        with tf.name_scope(self._name_scope):
            reset_accum = self.reset_accumulators()

            with tf.control_dependencies([reset_accum]):
                accum_op = self.accumulate_updates()

            with tf.control_dependencies([accum_op]):
                return self.update_spn(optimizer)
