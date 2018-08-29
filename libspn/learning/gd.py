# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

from collections import namedtuple
import tensorflow as tf
from libspn.inference.value import Value, LogValue
from libspn.graph.algorithms import traverse_graph
from libspn.learning.type import LearningType
from libspn.learning.type import LearningMethod
from libspn.learning.type import GradientType
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
    def __init__(self, root, value=None, value_inference_type=None,
                 log=True, learning_type=LearningType.SUPERVISED,
                 learning_method=LearningMethod.DISCRIMINATIVE,
                 gradient_type=GradientType.SOFT, learning_rate=0.0001,
                 dropconnect_keep_prob=None):
        self._root = root
        self._log = log
        if value is None:
            val_kwargs = dict(dropconnect_keep_prob=dropconnect_keep_prob)
            if log:
                self._value = LogValue(value_inference_type, **val_kwargs)
            else:
                self._value = Value(value_inference_type, **val_kwargs)
        else:
            self._value = value
            self._log = value.log()

        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be a positive number")
        else:
            self._learning_rate = learning_rate

        self._learning_type = learning_type
        self._learning_method = learning_method
        self._gradient_type = gradient_type

        # Create a name scope
        with tf.name_scope("GDLearning") as self._name_scope:
            pass

    @property
    def value(self):
        """Value or LogValue: Computed SPN values."""
        return self._value

    def learn(self, loss=None, gradient_type=None,
              optimizer=tf.train.GradientDescentOptimizer):
        """Assemble TF operations performing GD learning of the SPN."""

        # Traverse the graph and set gradient-type for all OpNodes
        self._root.set_gradient_types(self._gradient_type if gradient_type
                                      is None else gradient_type)

        # If a loss function is not provided, define the loss function based
        # on learning-type and learning-method
        if loss is None:
            if self._learning_method == LearningMethod.GENERATIVE:
                if self._learning_type == LearningType.UNSUPERVISED:
                    likelihood = self._value.get_value(
                        self._root, with_ivs=False)
                else:  # SUPERVISED
                    likelihood = self._value.get_value(
                        self._root, with_ivs=True)

                loss = -tf.reduce_mean(likelihood)
            else:  # DISCRIMINATIVE
                if self._learning_type == LearningType.UNSUPERVISED \
                        or self._root.ivs is None:
                    pass
                else:  # SUPERVISED
                    joint = self._value.get_value(
                        self._root, with_ivs=True)
                    marginalized = self._value.get_value(
                        self._root, with_ivs=False)
                loss = -tf.reduce_mean(joint - marginalized)

        # Assemble TF ops for optimizing and weights normalization
        with tf.name_scope(self._name_scope):
            minimize = optimizer(self._learning_rate).minimize(loss=loss)

            # After applying gradients to weights, normalize weights
            with tf.control_dependencies([minimize]):
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