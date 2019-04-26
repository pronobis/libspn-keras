import abc

import numpy as np
import tensorflow as tf

from libspn import utils, conf
from libspn.graph.node import Node, VarNode
from libspn.graph.scope import Scope


class ContinuousLeafBase(VarNode, abc.ABC):

    """
    An abstract leaf node for modeling continuous variables. Each input variable can have multiple
    components, each with their own trainable parameters.

    Args:
        feed (Tensor): Tensor feeding this node or ``None``. If ``None``,
            an internal placeholder will be used to feed this node.
        num_vars (int): The number of variables this leaf models.
        num_components (int): The number of components per variable
        name (str): Name of this node
        evidence_indicator_feed (Tensor): Boolean Tensor with shape ``[batch_size, num_vars]``.
            The node will interpret ``True`` values as actual evidence, while others will be
            marginalized by setting the component outputs to 1 (or 0 in the log domain). If ``None``
            is given, the node will use a ``tf.placeholder_with_default`` containing only ``True``.
        dimensionality (int): The number of dimensions.
        component_axis (int): Which axes corresponds to components. Used only internally.
    """

    def __init__(self, feed=None, num_vars=1, num_components=2, name="ContinuousBase",
                 evidence_indicator_feed=None, dimensionality=1, component_axis=-1,
                 samplewise_normalization=False):
        self._dimensionality = dimensionality
        self._num_vars = num_vars
        self._num_components = num_components
        self._component_axis = component_axis
        self._sample_wise_normalization = samplewise_normalization
        super().__init__(feed=feed, name=name)
        self.attach_evidence_indicator(evidence_indicator_feed)
        self._dist = self._create_dist()

    @property
    def num_vars(self):
        """int: Number of random variables."""
        return self._num_vars

    @property
    def num_components(self):
        """int: Number of components per variable. """
        return self._num_components

    @property
    def evidence(self):
        """int: Tensor holding evidence placeholder. """
        return self._evidence_indicator

    @property
    def dimensionality(self):
        """int: The number of dimensions modeled."""
        return self._dimensionality

    @property
    def input_shape(self):
        """Tuple of either the number of variables (in case of univariate distributions) or a
        tuple of the number of variables and the number of dimensions."""
        if self._dimensionality == 1:
            return (self._num_vars,)
        return (self._num_vars, self._dimensionality)

    @property
    @abc.abstractmethod
    def variables(self):
        """Tuple of the variables contained by this node """

    @abc.abstractmethod
    def _create_dist(self):
        """Creates the tfp.Distribution instance """

    @utils.docinherit(VarNode)
    def _create_placeholder(self):
        return tf.placeholder(conf.dtype, (None,) + self.input_shape)

    def _create_evidence_indicator(self):
        """Creates a placeholder with default value. The default value is a ``Tensor`` of shape
        [batch, num_vars] filled with ``True``.

        Return:
            Evidence indicator placeholder: a placeholder ``Tensor`` set to True for each variable.
        """

        return tf.placeholder_with_default(
            tf.cast(tf.ones([tf.shape(self.feed)[0], self._num_vars]), tf.bool),
            shape=(None, self._num_vars))

    def attach_evidence_indicator(self, indicator):
        """Set a tensor that feeds the evidence indicators.

        Args:
           indicator (Tensor):  Tensor feeding this node or ``None``. If ``None``,
                                an internal placeholder will be used to feed this node.
        """
        if indicator is None:
            self._evidence_indicator = self._create_evidence_indicator()
        else:
            self._evidence_indicator = indicator

    @utils.docinherit(Node)
    def serialize(self):
        data = super().serialize()
        data['num_vars'] = self._num_vars
        data['num_components'] = self._num_components
        data['dimensionality'] = self._dimensionality
        return data

    @utils.docinherit(Node)
    def deserialize(self, data):
        self._num_vars = data['num_vars']
        self._num_components = data['num_components']
        self._dimensionality = data['dimensionality']
        super().deserialize(data)

    def _total_accumulates(self, init_val, shape):
        """Creates a ``Variable`` that holds the counts per component.

        Return:
              Counts per component: ``Variable`` holding counts per component.
        """
        # TODO shouldn't this go somewhere else?
        init = tf.broadcast_to(init_val, shape)
        return tf.Variable(init, name=self.name + "TotalCounts", trainable=False, dtype=tf.float32)

    @utils.docinherit(Node)
    def _compute_out_size(self):
        return self._num_vars * self._num_components

    @utils.lru_cache
    def _tile_num_components(self, tensor, axis=None):
        """Tiles a ``Tensor`` so that its last axis contains ``num_components`` repetitions of the
        original values. If the incoming tensor's last dim size equals 1, it will tile along this
        axis. If the incoming tensor's last dim size is not equal to 1, it will append a dimension
        of size 1 and then perform tiling.

        Args:
            tensor (Tensor): The tensor to tile ``num_components`` times.

        Return:
            Tiled tensor: Input tensor tiled ``num_components`` times along last axis.
        """
        axis = axis or self._component_axis

        if tensor.shape[axis].value != 1:
            tensor = tf.expand_dims(tensor, axis=axis)
        multiples = np.ones(len(tensor.shape))
        multiples[axis] = self._num_components
        return tf.tile(tensor, multiples)

    def _evidence_mask(self, value, no_evidence_fn):
        """Consists of selecting the (log) pdf of the input or ``1`` (``0`` for log) in case
        of lacking evidence.

        Args:
            value (Tensor): The (log) pdf.
            no_evidence_fn (function): A function ``fun(value)`` that takes in the tensor from the
                                       ``value`` and returns the corresponding output in case of
                                       lacking evidence.
        Returns:
            Evidence masked output: Tensor containing pdf or no evidence values.
        """
        out_shape = (-1, self._compute_out_size())
        # Now we can't rely on _component_axis, since value has shape
        # [batch, num_vars * num_components] (no dimensionality axis)
        evidence = tf.reshape(self._tile_num_components(self.evidence, axis=-1), out_shape)
        value = tf.reshape(value, out_shape)
        return tf.where(evidence, value, no_evidence_fn(value))

    @utils.lru_cache
    def _normalized_feed(self, epsilon=1e-10):
        feed_mean, feed_stddev = self._feed_mean_and_stddev()
        return (self._feed - feed_mean) / (feed_stddev + epsilon)

    @utils.lru_cache
    def _feed_mean_and_stddev(self):
        """Computes the mean and standard deviation of the input. """
        reduce_axes = list(range(1, len(self._feed.shape)))

        feed_mean = tf.reduce_mean(self._feed, axis=reduce_axes, keepdims=True)
        feed_stddev = tf.sqrt(tf.reduce_mean(
            tf.square(self._feed - feed_mean), keepdims=True, axis=reduce_axes))
        return feed_mean, feed_stddev

    @utils.lru_cache
    def _preprocessed_feed(self):
        """Returns the preprocessed feed (possibly normalized and tiled) """
        return self._tile_num_components(
            self._normalized_feed() if self._sample_wise_normalization else self._feed)

    @utils.docinherit(Node)
    @utils.lru_cache
    def _compute_log_value(self):
        return self._evidence_mask(self._dist.log_prob(self._preprocessed_feed()), tf.zeros_like)

    @utils.docinherit(Node)
    def _compute_scope(self):
        return [Scope(self, i) for i in range(self._num_vars) for _ in range(self._num_components)]

    @utils.docinherit(Node)
    @utils.lru_cache
    def _compute_mpe_state(self, counts):
        # MPE state can be found by taking the mean of the mixture components that are 'selected'
        # by the counts
        counts_reshaped = tf.reshape(counts, (-1, self._num_vars, self._num_components))
        indices = tf.argmax(counts_reshaped, axis=-1) + tf.expand_dims(
            tf.range(self._num_vars, dtype=tf.int64) * self._num_components, axis=0)
        flat_shape = (-1,) if self._dimensionality == 1 else (-1, self._dimensionality)
        mpe_state = tf.gather(tf.reshape(self.mode(), flat_shape), indices=indices, axis=0)
        evidence = tf.tile(tf.expand_dims(self.evidence, -1), (1, 1, self.dimensionality)) \
            if self.dimensionality > 1 else self.evidence
        return tf.where(evidence, self.feed, mpe_state)

    def impute_by_posterior_marginal(self, root, name="ImputeByPosteriorMarginal"):
        """ Impute data by multiplying posterior marginals of components that are not part of the
        evidence. See also http://reasoning.cs.ucla.edu/fetch.php?id=36&type=bib (Darwiche, 2003).

        Args:
            root (Node):    Root of the SPN.
            name (str):     Name of the TensorFlow subgraph

        Returns:
            A Tensor that holds the original evidence together with the imputed values.
        """
        with tf.name_scope(name):
            posterior_marginal = tf.gradients(root.get_log_value(), self.get_log_value())[0]
            posterior_marginal = tf.reshape(
                posterior_marginal, [-1, self.num_vars, self.num_components])
            weighted_modes = tf.reduce_sum(tf.expand_dims(self.mode(), 0) * posterior_marginal, axis=-1)
            weighted_modes /= tf.reduce_sum(posterior_marginal, axis=-1) + 1e-8
            if self._sample_wise_normalization:
                feed_mean, feed_stddev = self._feed_mean_and_stddev()
                weighted_modes = weighted_modes * feed_stddev + feed_mean
            return tf.where(self.evidence, self._feed, weighted_modes)

    def mode(self):
        return self._dist.mode()


def _softplus_inverse_np(x):
    return np.log(1 - np.exp(-x)) + x


def _logit(x, name="Logit"):
    with tf.name_scope(name):
        return -tf.log(tf.reciprocal(x) - 1)