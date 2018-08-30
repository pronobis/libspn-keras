# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

import tensorflow as tf
from libspn.inference.value import Value, LogValue
from libspn.graph.algorithms import traverse_graph
from libspn.exceptions import StructureError
from libspn.learning.type import LearningTaskType
from libspn.learning.type import LearningMethodType
from libspn.learning.type import GradientType
from libspn.graph.distribution import GaussianLeaf
from libspn.graph.sum import Sum
from libspn.log import get_logger


class GDLearning:
    """Assembles TF operations performing Gradient Descent learning of an SPN.

    Args:
        value_inference_type (InferenceType): The inference type used during the
            upwards pass through the SPN. Ignored if ``mpe_path`` is given.
        learning_rate (float): Learning rate parameter used for updating SPN weights.
        learning_task_type (LearningTaskType): Learning type used while learning.
        learning_method (LearningMethodType): Learning method type, can be either generative
            (LearningMethodType.GENERATIVE) or discriminative (LearningMethodType.DISCRIMINATIVE).
        gradient_type (GradientType): The type of gradients to use for backpropagation, can be
            either soft (effectively viewing sum nodes as weighted sums) or hard (effectively
            viewing sum nodes as weighted maxes). Soft and hard correspond to GradientType.SOFT
            and GradientType.HARD, respectively.
        marginalizing_root (Sum, ParSums, SumsLayer): A sum node without IVs attached to it (or
            IVs with a fixed no-evidence feed). If it is omitted here, the node will constructed
            internally once needed.
        name (str): The name given to this instance of GDLearning.
        l1_regularize_coeff (float or Tensor): The L1 regularization coefficient.
        l2_regularize_coeff (float or Tensor): The L2 regularization coefficient.
    """

    __logger = get_logger()

    def __init__(self, root, value=None, value_inference_type=None, dropconnect_keep_prob=None,
                 learning_task_type=LearningTaskType.SUPERVISED,
                 learning_method=LearningMethodType.DISCRIMINATIVE,
                 gradient_type=GradientType.SOFT, learning_rate=1e-4, marginalizing_root=None,
                 name="GDLearning", l1_regularize_coeff=None, l2_regularize_coeff=None):

        if learning_task_type == LearningTaskType.UNSUPERVISED and \
                learning_method == LearningMethodType.DISCRIMINATIVE:
            raise ValueError("It is not possible to do unsupervised learning discriminatively.")

        self._root = root
        self._marginalizing_root = marginalizing_root
        if self._turn_off_dropconnect(dropconnect_keep_prob, learning_task_type):
            self._root.set_dropconnect_keep_prob(1.0)
            if self._marginalizing_root is not None:
                self._marginalizing_root.set_dropconnect_keep_prob(1.0)

        if value is not None and isinstance(value, LogValue):
            self._log_value = value
        else:
            if value is not None:
                GDLearning.__logger.warn(
                    "{}: Value instance is ignored since the current implementation does "
                    "not support gradients with non-log inference. Using a LogValue instance "
                    "instead.".format(name))
            self._log_value = LogValue(
                value_inference_type, dropconnect_keep_prob=dropconnect_keep_prob)
        self._learning_rate = learning_rate
        self._learning_task_type = learning_task_type
        self._learning_method = learning_method
        self._l1_regularize_coeff = l1_regularize_coeff
        self._l2_regularize_coeff = l2_regularize_coeff
        self._dropconnect_keep_prob = dropconnect_keep_prob
        self._gradient_type = gradient_type
        self._name = name

    def learn(self, loss=None, gradient_type=None, optimizer=tf.train.GradientDescentOptimizer):
        """Assemble TF operations performing GD learning of the SPN. This includes setting up
        the loss function (with regularization), setting up the optimizer and setting up
        post gradient-update ops.

        loss (Tensor): The operation corresponding to the loss to minimize.
        optimizer (tf.train.Optimizer): A TensorFlow optimizer to use for minimizing the loss.
        gradient_type (GradientType): The type of gradients to use for backpropagation, can be
            either soft (effectively viewing sum nodes as weighted sums) or hard (effectively
            viewing sum nodes as weighted maxes). Soft and hard correspond to GradientType.SOFT
            and GradientType.HARD, respectively.

        Returns:
            A grouped operation that (i) updates the parameters using gradient descent, (ii)
            projects new weights onto the probability simplex and (iii) clips new variances of
            GaussianLeaf nodes.
        """
        if self._learning_task_type == LearningTaskType.SUPERVISED and self._root.ivs is None:
            raise StructureError(
                "{}: the SPN rooted at {} does not have a latent IVs node, so cannot setup "
                "conditional class probabilities.".format(self._name, self._root))

        # Traverse the graph and set gradient-type for all OpNodes
        self._root.set_gradient_types(gradient_type or self._gradient_type)

        # If a loss function is not provided, define the loss function based
        # on learning-type and learning-method
        with tf.name_scope("Loss"):
            loss = loss or (self.mle_loss() if
                            self._learning_method == LearningMethodType.GENERATIVE else
                            self.cross_entropy_loss())
            if self._l1_regularize_coeff is not None or self._l2_regularize_coeff is not None:
                loss += self.regularization_loss()

        # Assemble TF ops for optimizing and weights normalization
        with tf.name_scope("ParameterUpdate"):
            minimize = optimizer(self._learning_rate).minimize(loss=loss)
            return self.post_gradient_update(minimize), loss

    def post_gradient_update(self, update_op):
        """Constructs post-parameter update ops such as normalization of weights and clipping of
        scale parameters of GaussianLeaf nodes.

        Args:
            update_op (Tensor): A Tensor corresponding to the parameter update.

        Returns:
            An updated operation where the post-processing has been ensured by TensorFlow's control
            flow mechanisms.
        """
        with tf.name_scope("PostGradientUpdate"):

            # After applying gradients to weights, normalize weights
            with tf.control_dependencies([update_op]):
                weight_norm_ops = []

                def fun(node):
                    if node.is_param:
                        weight_norm_ops.append(node.normalize())

                    if isinstance(node, GaussianLeaf) and node.learn_distribution_parameters:
                        weight_norm_ops.append(tf.assign(node.scale_variable, tf.maximum(
                            node.scale_variable, node._min_stddev)))

                with tf.name_scope("WeightNormalization"):
                    traverse_graph(self._root, fun=fun)
            return tf.group(*weight_norm_ops, name="weight_norm")

    def cross_entropy_loss(self, name="CrossEntropyLoss", reduce_fn=tf.reduce_mean,
                           dropconnect_keep_prob=None):
        """Sets up the cross entropy loss, which is equivalent to -log(p(Y|X)).

        Args:
            name (str): Name of the name scope for the Ops defined here
            reduce_fn (Op): An operation that reduces the losses for all samples to a scalar.
            dropconnect_keep_prob (float or Tensor): Keep probability for dropconnect, will
                override the value of GDLearning._dropconnect_keep_prob.

        Returns:
            A Tensor corresponding to the cross-entropy loss.
        """
        with tf.name_scope(name):
            log_prob_data_and_labels = self._log_value.get_value(self._root)
            log_prob_data = self._log_likelihood(dropconnect_keep_prob=dropconnect_keep_prob)
            return -reduce_fn(log_prob_data_and_labels - log_prob_data)

    def mle_loss(self, name="MaximumLikelihoodLoss", reduce_fn=tf.reduce_mean,
                 dropconnect_keep_prob=None):
        """Returns the maximum (log) likelihood estimate loss function which corresponds to
        -log(p(X)) in the case of unsupervised learning or -log(p(X,Y)) in the case of supservised
        learning.

        Args:
            name (str): The name for the name scope to use
            reduce_fn (Op): An operation that reduces the losses for all samples to a scalar.
            dropconnect_keep_prob (float or Tensor): Keep probability for dropconnect, will
                override the value of GDLearning._dropconnect_keep_prob.
        Returns:
            A Tensor corresponding to the MLE loss
        """
        with tf.name_scope(name):
            if self._learning_task_type == LearningTaskType.UNSUPERVISED:
                if self._root.ivs is not None:
                    likelihood = self._log_likelihood(dropconnect_keep_prob=dropconnect_keep_prob)
                else:
                    likelihood = self._log_value.get_value(self._root)
            elif self._root.ivs is None:
                raise StructureError("Root should have IVs node when doing supervised learning.")
            else:
                likelihood = self._log_value.get_value(self._root)
            return -reduce_fn(likelihood)

    def _log_likelihood(self, learning_task_type=None, dropconnect_keep_prob=None):
        """Computes log(p(X)) by creating a copy of the root node without IVs. Also turns off
        dropconnect at the root if necessary.

        Returns:
            A Tensor of shape [batch, 1] corresponding to the log likelihood of the data.
        """
        marginalizing_root = self._marginalizing_root or Sum(
            *self._root.values, weights=self._root.weights)
        learning_task_type = learning_task_type or self._learning_task_type
        dropconnect_keep_prob = dropconnect_keep_prob or self._dropconnect_keep_prob
        if self._turn_off_dropconnect(dropconnect_keep_prob, learning_task_type):
            marginalizing_root.set_dropconnect_keep_prob(1.0)
        return self._log_value.get_value(marginalizing_root)

    def regularization_loss(self, name="Regularization"):
        """Adds regularization to the weight nodes. This can be either L1 or L2 or both, depending
        on what is specified at instantiation of GDLearning.

        Returns:
            A Tensor with the total regularization loss.
        """

        with tf.name_scope(name):
            losses = []

            def regularize_node(node):
                if node.is_param:
                    if self._l1_regularize_coeff is not None:
                        losses.append(
                            self._l1_regularize_coeff * tf.reduce_sum(tf.abs(node.variable)))
                    if self._l2_regularize_coeff is not None:
                        losses.append(
                            self._l2_regularize_coeff * tf.reduce_sum(tf.square(node.variable)))

            traverse_graph(self._root, fun=regularize_node)
            return tf.add_n(losses)

    @staticmethod
    def _turn_off_dropconnect(dropconnect_keep_prob, learning_task_type):
        """Determines whether to turn off dropconnect for the root node. """
        return dropconnect_keep_prob is not None and \
            (not isinstance(dropconnect_keep_prob, (int, float)) or dropconnect_keep_prob == 1.0) \
            and learning_task_type == LearningTaskType.SUPERVISED

    @property
    def value(self):
        """Value or LogValue: Computed SPN values."""
        return self._log_value