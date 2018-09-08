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
from libspn.graph.weights import Weights
from libspn.graph.sum import Sum
from libspn.graph.concat import Concat
from libspn.log import get_logger
from libspn.utils import maybe_first


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

    def __init__(self, root, value_inference_type=None, dropconnect_keep_prob=None,
                 dropprod_keep_prob=None, noise=None,
                 learning_task_type=LearningTaskType.SUPERVISED,
                 learning_method=LearningMethodType.DISCRIMINATIVE,
                 gradient_type=GradientType.SOFT, learning_rate=1e-4, marginalizing_root=None,
                 name="GDLearning", l1_regularize_coeff=None, l2_regularize_coeff=None,
                 confidence_penalty_coeff=None,
                 entropy_regularize_coeff=None,
                 gauss_regularize_coeff=None,
                 batch_noise=None,
                 linear_w_minimum=1e-2):

        if learning_task_type == LearningTaskType.UNSUPERVISED and \
                learning_method == LearningMethodType.DISCRIMINATIVE:
            raise ValueError("It is not possible to do unsupervised learning discriminatively.")

        self._root = root
        self._marginalizing_root = marginalizing_root
        if self._turn_off_dropconnect_root(dropconnect_keep_prob, learning_task_type):
            self._root.set_dropconnect_keep_prob(1.0)
            if self._marginalizing_root is not None:
                self._marginalizing_root.set_dropconnect_keep_prob(1.0)

        self.__logger.debug1("Dropconnect malfunctioning {}".format(dropconnect_keep_prob))
        self._learning_rate = learning_rate
        self._learning_task_type = learning_task_type
        self._learning_method = learning_method
        self._l1_regularize_coeff = l1_regularize_coeff
        self._l2_regularize_coeff = l2_regularize_coeff
        self._entropy_regularize_coeff = entropy_regularize_coeff
        self._gauss_regularize_coeff = gauss_regularize_coeff
        self._confidence_penalty_coeff = confidence_penalty_coeff
        self._dropconnect_keep_prob = dropconnect_keep_prob
        self._dropprod_keep_prob = dropprod_keep_prob
        self._batch_noise = batch_noise
        self._gradient_type = gradient_type
        self._value_inference_type = value_inference_type
        self._name = name
        self._noise = noise
        self._linear_w_minimum = linear_w_minimum

    def loss(self, learning_method=None, dropconnect_keep_prob=None, dropprod_keep_prob=None,
             noise=None, batch_noise=None):
        """Assembles main objective operations. In case of generative learning it will select 
        the MLE objective, whereas in discriminative learning it selects the cross entropy.
        
        Args:
            learning_method (LearningMethodType): The learning method (can be either generative 
                or discriminative).
            dropconnect_keep_prob (float or Tensor): The dropconnect keep probability to use. 
                Overrides the value of GDLearning._dropconnect_keep_prob
            
        Returns:
            An operation to compute the main loss function.
        """
        learning_method = learning_method or self._learning_method
        if learning_method == LearningMethodType.GENERATIVE:
            return self.mle_loss(
                dropconnect_keep_prob=dropconnect_keep_prob, dropprod_keep_prob=dropprod_keep_prob,
                noise=noise, batch_noise=batch_noise)
        return self.cross_entropy_loss(
            dropconnect_keep_prob=dropconnect_keep_prob, dropprod_keep_prob=dropprod_keep_prob,
            noise=noise, batch_noise=batch_noise)

    def learn(self, loss=None, gradient_type=None, optimizer=tf.train.GradientDescentOptimizer):
        """Assemble TF operations performing GD learning of the SPN. This includes setting up
        the loss function (with regularization), setting up the optimizer and setting up
        post gradient-update ops.

        Args:
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
            loss = loss or self.loss()
            reg_coeffs = [self._l1_regularize_coeff, self._l2_regularize_coeff,
                          self._entropy_regularize_coeff]
            if any(c is not None for c in reg_coeffs) and any(c != 0.0 for c in reg_coeffs):
                loss += self.regularization_loss()
            if self._confidence_penalty_coeff is not None and self._confidence_penalty_coeff != 0.0:
                loss += self.confidence_penalty_loss()

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
                        weight_norm_ops.append(
                            node.normalize(linear_w_minimum=self._linear_w_minimum))

                    if isinstance(node, GaussianLeaf) and node.learn_distribution_parameters:
                        weight_norm_ops.append(tf.assign(node.scale_variable, tf.maximum(
                            node.scale_variable, node._min_stddev)))

                with tf.name_scope("WeightNormalization"):
                    traverse_graph(self._root, fun=fun)
            return tf.group(*weight_norm_ops, name="weight_norm")

    def cross_entropy_loss(self, name="CrossEntropyLoss", reduce_fn=tf.reduce_mean,
                           dropconnect_keep_prob=None, dropprod_keep_prob=None, noise=None,
                           batch_noise=None):
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
            dropconnect_keep_prob = maybe_first(dropconnect_keep_prob, self._dropconnect_keep_prob)
            dropprod_keep_prob = maybe_first(dropprod_keep_prob, self._dropprod_keep_prob)
            noise = maybe_first(noise, self._noise)
            batch_noise = maybe_first(batch_noise, self._batch_noise)
            value_gen = LogValue(
                dropconnect_keep_prob=dropconnect_keep_prob, dropprod_keep_prob=dropprod_keep_prob,
                noise=noise, inference_type=self._value_inference_type, batch_noise=batch_noise,
                matmul_or_conv=not self._turn_off_dropconnect_root(
                    dropconnect_keep_prob, self._learning_task_type))
            log_prob_data_and_labels = value_gen.get_value(self._root)
            log_prob_data = self._log_likelihood(
                dropconnect_keep_prob=dropconnect_keep_prob,
                dropprod_keep_prob=dropprod_keep_prob,
                noise=noise, batch_noise=batch_noise)
            return -reduce_fn(log_prob_data_and_labels - log_prob_data)

    def mle_loss(self, name="MaximumLikelihoodLoss", reduce_fn=tf.reduce_mean,
                 dropconnect_keep_prob=None, dropprod_keep_prob=None, noise=None, batch_noise=None):
        """Returns the maximum (log) likelihood estimator loss function which corresponds to
        -log(p(X)) in the case of unsupervised learning or -log(p(X,Y)) in the case of supervised
        learning.

        Args:
            name (str): The name for the name scope to use
            reduce_fn (function): An function that returns an operation that reduces the losses for 
                all samples to a scalar.
            dropconnect_keep_prob (float or Tensor): Keep probability for dropconnect, will
                override the value of GDLearning._dropconnect_keep_prob.
        Returns:
            A Tensor corresponding to the MLE loss
        """
        with tf.name_scope(name):
            dropconnect_keep_prob = maybe_first(dropconnect_keep_prob, self._dropconnect_keep_prob)
            dropprod_keep_prob = maybe_first(dropprod_keep_prob, self._dropprod_keep_prob)
            noise = maybe_first(noise, self._noise)
            batch_noise = maybe_first(batch_noise, self._batch_noise)
            value_gen = LogValue(
                dropconnect_keep_prob=dropconnect_keep_prob,
                dropprod_keep_prob=dropprod_keep_prob,
                noise=noise, batch_noise=batch_noise,
                inference_type=self._value_inference_type,
                matmul_or_conv=not self._turn_off_dropconnect_root(
                    dropconnect_keep_prob, learning_task_type=self._learning_task_type))
            if self._learning_task_type == LearningTaskType.UNSUPERVISED:
                if self._root.ivs is not None:
                    likelihood = self._log_likelihood(
                        dropconnect_keep_prob=dropconnect_keep_prob,
                        dropprod_keep_prob=dropprod_keep_prob,
                        noise=noise, batch_noise=batch_noise)
                else:
                    likelihood = value_gen.get_value(self._root)
            elif self._root.ivs is None:
                raise StructureError("Root should have IVs node when doing supervised learning.")
            else:
                likelihood = value_gen.get_value(self._root)
            return -reduce_fn(likelihood)

    def confidence_penalty_loss(
            self, confidence_penalty_coeff=None, dropconnect_keep_prob=None, 
            dropprod_keep_prob=None, noise=None, name="ConfidencePenalty", batch_noise=None):
        self.__logger.debug1("Assembling confidence penalty loss")
        with tf.name_scope(name):
            confidence_penalty_coeff = maybe_first(
                confidence_penalty_coeff, self._confidence_penalty_coeff)
            dropconnect_keep_prob = maybe_first(dropconnect_keep_prob, self._dropconnect_keep_prob)
            dropprod_keep_prob = maybe_first(dropprod_keep_prob, self._dropprod_keep_prob)
            noise = maybe_first(noise, self._noise)
            batch_noise = maybe_first(batch_noise, self._batch_noise)

            matmul_or_conv = not self._turn_off_dropconnect_root(
                dropconnect_keep_prob, learning_task_type=self._learning_task_type)

            value_gen = LogValue(
                dropprod_keep_prob=dropprod_keep_prob, dropconnect_keep_prob=dropconnect_keep_prob,
                noise=noise, matmul_or_conv=matmul_or_conv, 
                inference_type=self._value_inference_type, batch_noise=batch_noise)
            if len(self._root.values) > 1:
                sub_spns = Concat(*self._root.values)
            else:
                sub_spns = self._root.values[0].node
            weight_value = value_gen.get_value(self._root.weights.node)
            sub_spn_value = value_gen.get_value(sub_spns)
            log_p_joint_xy = sub_spn_value + weight_value
            log_p_x = self._log_likelihood(
                dropconnect_keep_prob=dropconnect_keep_prob, dropprod_keep_prob=dropprod_keep_prob,
                noise=noise, batch_noise=batch_noise)
            # Confidences
            log_p_y_given_x = log_p_joint_xy - log_p_x
            negative_entropy = tf.reduce_mean(tf.reduce_sum(
                log_p_y_given_x * tf.exp(log_p_y_given_x), axis=-1))
            return confidence_penalty_coeff * negative_entropy

    def _log_likelihood(self, learning_task_type=None, dropconnect_keep_prob=None,
                        dropprod_keep_prob=None, noise=None, batch_noise=None):
        """Computes log(p(X)) by creating a copy of the root node without IVs. Also turns off
        dropconnect at the root if necessary.

        Returns:
            A Tensor of shape [batch, 1] corresponding to the log likelihood of the data.
        """
        marginalizing_root = self._marginalizing_root or Sum(
            *self._root.values, weights=self._root.weights)
        learning_task_type = learning_task_type or self._learning_task_type
        dropconnect_keep_prob = dropconnect_keep_prob or self._dropconnect_keep_prob
        dropprod_keep_prob = dropprod_keep_prob or self._dropprod_keep_prob
        batch_noise = maybe_first(batch_noise, self._batch_noise)
        noise = maybe_first(noise, self._noise)
        if self._turn_off_dropconnect_root(dropconnect_keep_prob, learning_task_type):
            marginalizing_root.set_dropconnect_keep_prob(1.0)
        return LogValue(
            dropconnect_keep_prob=dropconnect_keep_prob,
            dropprod_keep_prob=dropprod_keep_prob,
            noise=noise, batch_noise=batch_noise,
            inference_type=self._value_inference_type,
            matmul_or_conv=not self._turn_off_dropconnect_root(
                dropconnect_keep_prob, learning_task_type)).get_value(marginalizing_root)

    def regularization_loss(self, name="Regularization"):
        """Adds regularization to the weight nodes. This can be either L1 or L2 or both, depending
        on what is specified at instantiation of GDLearning.

        Returns:
            A Tensor computing the total regularization loss.
        """

        def _enable(c):
            return c is not None and c != 0.0

        with tf.name_scope(name):
            losses = []

            def regularize_node(node):
                if isinstance(node, GaussianLeaf):
                    if _enable(self._gauss_regularize_coeff):
                        losses.append(self._gauss_regularize_coeff * tf.negative(node.entropy()))

                if isinstance(node, Weights):
                    linear_w = tf.exp(node.variable) if node.log else node.variable
                    if _enable(self._l1_regularize_coeff):
                        losses.append(
                            self._l1_regularize_coeff * tf.reduce_sum(tf.abs(linear_w)))
                    if _enable(self._l2_regularize_coeff):
                        losses.append(
                            self._l2_regularize_coeff * tf.reduce_sum(tf.square(linear_w)))
                    if _enable(self._entropy_regularize_coeff):
                        if node.log:
                            losses.append(
                                self._entropy_regularize_coeff *
                                -tf.reduce_sum(node.variable * linear_w))
                        else:
                            losses.append(self._entropy_regularize_coeff *
                                          -tf.reduce_sum(linear_w * tf.log(
                                              node.variable + 1e-8)))

            traverse_graph(self._root, fun=regularize_node)
            return tf.add_n(losses) if losses else tf.constant(0.0)

    @staticmethod
    def _turn_off_dropconnect_root(dropconnect_keep_prob, learning_task_type):
        """Determines whether to turn off dropconnect for the root node. """
        return dropconnect_keep_prob is not None and \
            (not isinstance(dropconnect_keep_prob, (int, float)) or dropconnect_keep_prob == 1.0) \
            and learning_task_type == LearningTaskType.SUPERVISED

