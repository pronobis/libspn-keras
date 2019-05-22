import tensorflow as tf

from libspn import BlockSum
from libspn.graph.op.base_sum import BaseSum

from libspn.inference.value import LogValue
from libspn.graph.algorithms import traverse_graph
from libspn.exceptions import StructureError
from libspn.learning.type import LearningTaskType
from libspn.learning.type import LearningMethodType
from libspn.graph.leaf.location_scale import LocationScaleLeaf
from libspn.graph.op.sum import Sum
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
        marginalizing_root (Sum, ParallelSums, SumsLayer): A sum node without IndicatorLeafs attached to
            it (or IndicatorLeafs with a fixed no-evidence feed). If it is omitted here, the node
            will constructed internally once needed.
        name (str): The name given to this instance of GDLearning.
    """

    __logger = get_logger()

    def __init__(self, root, value=None, value_inference_type=None,
                 learning_task_type=LearningTaskType.SUPERVISED,
                 learning_method=LearningMethodType.DISCRIMINATIVE,
                 learning_rate=1e-4, marginalizing_root=None, name="GDLearning",
                 global_step=None, linear_w_minimum=1e-2):

        if learning_task_type == LearningTaskType.UNSUPERVISED and \
                learning_method == LearningMethodType.DISCRIMINATIVE:
            raise ValueError("It is not possible to do unsupervised learning discriminatively.")

        self._root = root
        self._marginalizing_root = marginalizing_root

        if value is not None and isinstance(value, LogValue):
            self._log_value = value
        else:
            if value is not None:
                GDLearning.__logger.warn(
                    "{}: Value instance is ignored since the current implementation does "
                    "not support gradients with non-log inference. Using a LogValue instance "
                    "instead.".format(name))
            self._log_value = LogValue(value_inference_type)
        self._learning_rate = learning_rate
        self._learning_task_type = learning_task_type
        self._learning_method = learning_method
        self._name = name
        self._global_step = global_step
        self._linear_w_minimum = linear_w_minimum

    def loss(self, learning_method=None, reduce_fn=tf.reduce_mean):
        """Assembles main objective operations. In case of generative learning it will select
        the MLE objective, whereas in discriminative learning it selects the cross entropy.

        Args:
            learning_method (LearningMethodType): The learning method (can be either generative
                or discriminative).

        Returns:
            An operation to compute the main loss function.
        """
        learning_method = learning_method or self._learning_method
        if learning_method == LearningMethodType.GENERATIVE:
            return self.negative_log_likelihood(reduce_fn=reduce_fn)
        return self.cross_entropy_loss(reduce_fn=reduce_fn)

    def learn(self, loss=None, optimizer=None, post_gradient_ops=True, name="LearnGD"):
        """Assemble TF operations performing GD learning of the SPN. This includes setting up
        the loss function (with regularization), setting up the optimizer and setting up
        post gradient-update ops.

        Args:
            loss (Tensor): The operation corresponding to the loss to minimize.
            optimizer (tf.train.Optimizer): A TensorFlow optimizer to use for minimizing the loss.
            post_gradient_ops (bool): Whether to use post-gradient ops such as normalization.

        Returns:
            A tuple of grouped update Ops and a loss Op.
        """
        if self._learning_task_type == LearningTaskType.SUPERVISED and self._root.latent_indicators is None:
            raise StructureError(
                "{}: the SPN rooted at {} does not have a latent IndicatorLeaf node, so cannot "
                "setup conditional class probabilities.".format(self._name, self._root))

        # If a loss function is not provided, define the loss function based
        # on learning-type and learning-method
        with tf.name_scope(name):
            with tf.name_scope("Loss"):
                if loss is None:
                    if self._learning_method == LearningMethodType.GENERATIVE:
                        loss = self.negative_log_likelihood()
                    else:
                        loss = self.cross_entropy_loss()
            # Assemble TF ops for optimizing and weights normalization
            with tf.name_scope("ParameterUpdate"):
                minimize = optimizer.minimize(loss=loss)
                if post_gradient_ops:
                    return self.post_gradient_update(minimize), loss
                else:
                    return minimize, loss

    def post_gradient_update(self, update_op):
        """Constructs post-parameter update ops such as normalization of weights and clipping of
        scale parameters of NormalLeaf nodes.

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

                    if isinstance(node, LocationScaleLeaf) and node._trainable_scale:
                        weight_norm_ops.append(tf.assign(node.scale_variable, tf.maximum(
                            node.scale_variable, node._min_scale)))

                with tf.name_scope("WeightNormalization"):
                    traverse_graph(self._root, fun=fun)
            return tf.group(*weight_norm_ops, name="weight_norm")

    def cross_entropy_loss(self, name="CrossEntropy", reduce_fn=tf.reduce_mean):
        """Sets up the cross entropy loss, which is equivalent to -log(p(Y|X)).

        Args:
            name (str): Name of the name scope for the Ops defined here
            reduce_fn (Op): An operation that reduces the losses for all samples to a scalar.

        Returns:
            A Tensor corresponding to the cross-entropy loss.
        """
        with tf.name_scope(name):
            log_prob_data_and_labels = LogValue().get_value(self._root)
            log_prob_data = self._log_likelihood()
            return -reduce_fn(log_prob_data_and_labels - log_prob_data)

    def negative_log_likelihood(self, name="NegativeLogLikelihood", reduce_fn=tf.reduce_mean):
        """Returns the maximum (log) likelihood estimate loss function which corresponds to
        -log(p(X)) in the case of unsupervised learning or -log(p(X,Y)) in the case of supservised
        learning.

        Args:
            name (str): The name for the name scope to use
            reduce_fn (function): An function that returns an operation that reduces the losses for
                all samples to a scalar.
        Returns:
            A Tensor corresponding to the MLE loss
        """
        with tf.name_scope(name):
            if self._learning_task_type == LearningTaskType.UNSUPERVISED:
                if self._root.latent_indicators is not None:
                    likelihood = self._log_likelihood()
                else:
                    likelihood = self._log_value.get_value(self._root)
            elif self._root.latent_indicators is None:
                raise StructureError("Root should have latent indicator node when doing supervised "
                                     "learning.")
            else:
                likelihood = self._log_value.get_value(self._root)
            return -reduce_fn(likelihood)

    def _log_likelihood(self):
        """Computes log(p(X)) by creating a copy of the root node without latent indicators.

        Returns:
            A Tensor of shape [batch, 1] corresponding to the log likelihood of the data.
        """
        if isinstance(self._root, BaseSum):
            marginalizing_root = self._marginalizing_root or Sum(
                *self._root.values, weights=self._root.weights)
        else:
            marginalizing_root = self._marginalizing_root or BlockSum(
                self._root.values[0], weights=self._root.weights, num_sums_per_block=1)
        return self._log_value.get_value(marginalizing_root)


