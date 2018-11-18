# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------
import tensorflow as tf
from libspn.graph.scope import Scope
from libspn.graph.node import VarNode, Node
from libspn.utils.initializers import Equidistant
from libspn import conf
from libspn import utils
import numpy as np
import tensorflow.contrib.distributions as tfd
from tensorflow_probability import distributions as tfp
from tensorflow_probability import bijectors
from tensorflow.contrib.distributions.python.ops import distribution_util
import abc
from libspn.utils import SPNGraphKeys


class DistributionLeaf(VarNode, abc.ABC):

    def __init__(self, feed=None, num_vars=1, num_components=2, name="DistributionLeaf",
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
        return self._num_vars

    @property
    def num_components(self):
        """Number of components per variable. """
        return self._num_components

    @property
    def evidence(self):
        """Tensor holding evidence placeholder. """
        return self._evidence_indicator

    @property
    def dimensionality(self):
        return self._dimensionality

    @property
    def input_shape(self):
        if self._dimensionality == 1:
            return (self._num_vars,)
        return (self._num_vars, self._dimensionality)

    @property
    @abc.abstractmethod
    def variables(self):
        """Tuple of the variables contained by this node """

    @property
    def trainable(self):
        return any(v.trainable for v in self.variables)

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
        init = utils.broadcast_value(init_val, shape, dtype=conf.dtype)
        return tf.Variable(init, name=self.name + "TotalCounts", trainable=False)

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
        reduce_axes = list(range(1, len(self._feed.shape)))

        feed_mean = tf.reduce_mean(self._feed, axis=reduce_axes, keepdims=True)
        feed_stddev = tf.sqrt(tf.reduce_mean(
            tf.square(self._feed - feed_mean), keepdims=True, axis=reduce_axes))
        return feed_mean, feed_stddev
        # evidence_size = tf.reduce_sum(
        #     tf.to_float(self.evidence), axis=reduce_axes, keepdims=True)
        # zeros = tf.zeros_like(self._feed)
        # feed_mean = tf.reduce_sum(tf.where(self.evidence, self._feed, zeros),
        #     axis=reduce_axes, keepdims=True) / evidence_size
        # feed_stddev = tf.sqrt(tf.reduce_sum(
        #     tf.where(self.evidence, tf.square(self._feed - feed_mean), zeros),
        #     keepdims=True, axis=reduce_axes) / evidence_size)
        # return feed_mean, feed_stddev

    @utils.lru_cache
    def _preprocessed_feed(self):
        return self._tile_num_components(
            self._normalized_feed() if self._sample_wise_normalization else self._feed)

    @utils.docinherit(Node)
    @utils.lru_cache
    def _compute_value(self, step=None):
        return self._evidence_mask(self._dist.prob(self._preprocessed_feed()), tf.ones_like)

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

    def completion_by_posterior_marginal(self, root):
        grad_log_likelihood = tf.gradients(root.get_log_value(), self.get_log_value())[0]
        grad_log_likelihood = tf.reshape(
            grad_log_likelihood, [-1, self.num_vars, self.num_components])
        marginal = tf.reduce_sum(tf.expand_dims(self.mode(), 0) * grad_log_likelihood, axis=-1)
        marginal /= (tf.reduce_sum(grad_log_likelihood, axis=-1) + 1e-8)
        if self._sample_wise_normalization:
            feed_mean, feed_stddev = self._feed_mean_and_stddev()
            marginal = marginal * feed_stddev + feed_mean
        return tf.where(self.evidence, self._feed, marginal)

    def entropy(self):
        return self._dist.entropy()

    def kl_divergence(self, other):
        return self._dist.kl_divergence(other)

    def cross_entropy(self, other):
        return self._dist.cross_entropy(other)

    def mode(self):
        return self._dist.mode()


class BetaLocationPrecisionLeaf(DistributionLeaf):

    def __init__(self, feed=None, evidence_indicator_feed=None, num_vars=1, num_components=2,
                 total_counts_init=1, trainable_loc=True, trainable_precision=True,
                 loc_init=Equidistant(minval=0.01, maxval=0.99),
                 precision_init=10.0, min_precision=1e-2, softplus_precision=True,
                 name="BetaLocationPrecision",
                 share_locs_across_vars=False, share_precision=False, sigmoid_loc=True):
        self._softplus_precision = softplus_precision
        self._sigmoid_loc = sigmoid_loc
        # Initial value for means
        variable_shape = self._variable_shape(num_vars, num_components)
        self._min_scale = _softplus_inverse_np(min_precision) if softplus_precision else min_precision
        self.init_variables(variable_shape, loc_init, precision_init, softplus_precision)
        self._trainable_precision = trainable_precision
        self._trainable_loc = trainable_loc
        self._share_locs_across_vars = share_locs_across_vars
        self._share_precision = share_precision

        super().__init__(feed=feed, name=name, dimensionality=1,
                         num_components=num_components, num_vars=num_vars,
                         evidence_indicator_feed=evidence_indicator_feed,
                         component_axis=-1)
        self._total_count_variable = self._total_accumulates(
            total_counts_init, (num_vars, num_components))

    @property
    def variables(self):
        return self._loc_variable, self._precision_variable

    def _create_dist(self):
        loc = tf.nn.sigmoid(self._loc_variable) if self._sigmoid_loc else self._loc_variable
        precision = tf.nn.softplus(self._precision_variable) \
            if self._softplus_precision else self._precision_variable
        concentration0 = precision * loc
        concentration1 = precision * (1 - loc)
        return tfp.Beta(concentration0=concentration0, concentration1=concentration1)

    def _variable_shape(self, num_vars, num_components):
        return [num_vars, num_components]

    def init_variables(self, shape, loc_init, precision_init, softplus_scale):
        # Initial value for means
        if isinstance(loc_init, float):
            self._loc_init = np.ones(shape, dtype=np.float32) * loc_init
        else:
            self._loc_init = loc_init

        # Initial values for variances.
        self._precision_init = np.ones(shape, dtype=np.float32) * precision_init
        if softplus_scale:
            self._precision_init = _softplus_inverse_np(self._precision_init)

    def initialize(self):
        """Provide initializers for mean, variance and total counts """
        return (self._loc_variable.initializer, self._precision_variable.initializer,
                self._total_count_variable.initializer)

    @property
    def loc_variable(self):
        """Tensor holding mean variable. """
        return self._loc_variable

    @property
    def precision_variable(self):
        return self._precision_variable

    @utils.docinherit(Node)
    def _create(self):
        super()._create()
        with tf.variable_scope(self._name):
            # Initialize locations
            shape = self._variable_shape(
                1 if self._share_locs_across_vars else self._num_vars,
                self._num_components)
            shape_kwarg = dict(shape=shape) if callable(self._loc_init) else dict()

            def init_wrapper(shape, dtype=None, partition_info=None):
                ret = self._loc_init(shape=shape, dtype=dtype, partition_info=partition_info)
                return logit(ret) if self._sigmoid_loc else ret

            self._loc_variable = tf.get_variable(
                "Loc", initializer=init_wrapper, dtype=conf.dtype,
                collections=[SPNGraphKeys.DIST_LOC, SPNGraphKeys.DIST_PARAMETERS,
                             tf.GraphKeys.GLOBAL_VARIABLES],
                trainable=self._trainable_loc, **shape_kwarg)

            # Initialize precision
            shape = self._variable_shape(
                1 if self._share_precision else self._num_vars,
                1 if self._share_precision else self._num_vars)
            shape_kwarg = dict(shape=shape) if callable(self._precision_init) else dict()
            self._precision_variable = tf.get_variable(
                "Scale", initializer=tf.maximum(self._precision_init, self._min_scale),
                dtype=conf.dtype,
                collections=[SPNGraphKeys.DIST_SCALE, SPNGraphKeys.DIST_PARAMETERS,
                             tf.GraphKeys.GLOBAL_VARIABLES],
                trainable=self._trainable_precision, **shape_kwarg)

    @utils.docinherit(Node)
    def serialize(self):
        data = super().serialize()
        data['loc_init'] = self._loc_init
        data['scale_init'] = self._scale_init
        return data

    @utils.docinherit(Node)
    def deserialize(self, data):
        self._loc_init = data['loc_init']
        self._precision_init = data['precision_init']
        super().deserialize(data)


class LocationScaleLeaf(DistributionLeaf, abc.ABC):

    def __init__(self, feed=None, evidence_indicator_feed=None, num_vars=1, num_components=2,
                 total_counts_init=1, trainable_loc=True, trainable_scale=True,
                 loc_init=Equidistant(),
                 scale_init=1.0, min_scale=1e-2, softplus_scale=True,
                 dimensionality=1, name="LocationScaleLeaf", component_axis=-1,
                 share_locs_across_vars=False, share_scales=False, samplewise_normalization=False):
        self._softplus_scale = softplus_scale
        # Initial value for means
        variable_shape = self._variable_shape(num_vars, num_components, dimensionality)
        self._min_scale = min_scale if not softplus_scale else np.log(np.exp(min_scale) - 1)
        self.init_variables(variable_shape, loc_init, scale_init, softplus_scale)
        self._trainable_scale = trainable_scale
        self._trainable_loc = trainable_loc
        self._share_locs_across_vars = share_locs_across_vars
        self._share_scales = share_scales

        super().__init__(feed=feed, name=name, dimensionality=dimensionality,
                         num_components=num_components, num_vars=num_vars,
                         evidence_indicator_feed=evidence_indicator_feed,
                         component_axis=component_axis,
                         samplewise_normalization=samplewise_normalization)
        self._total_count_variable = self._total_accumulates(
            total_counts_init, (num_vars, num_components))

    def init_variables(self, shape, loc_init, scale_init, softplus_scale):
        # Initial value for means
        if isinstance(loc_init, float):
            self._loc_init = np.ones(shape, dtype=np.float32) * loc_init
        else:
            self._loc_init = loc_init

        # Initial values for variances.
        self._scale_init = np.ones(shape, dtype=np.float32) * scale_init
        if softplus_scale:
            self._scale_init = _softplus_inverse_np(self._scale_init)

    @utils.docinherit(Node)
    def _create(self):
        super()._create()
        with tf.variable_scope(self._name):
            # Initialize locations
            shape = self._variable_shape(
                1 if self._share_locs_across_vars else self._num_vars,
                self._num_components, self._dimensionality)
            shape_kwarg = dict(shape=shape) if callable(self._loc_init) else dict()
            self._loc_variable = tf.get_variable(
                "Loc", initializer=self._loc_init, dtype=conf.dtype,
                collections=[SPNGraphKeys.DIST_LOC, SPNGraphKeys.DIST_PARAMETERS,
                             tf.GraphKeys.GLOBAL_VARIABLES],
                trainable=self._trainable_loc, **shape_kwarg)

            # Initialize scale
            shape = self._variable_shape(
                1 if self._share_scales else self._num_vars,
                1 if self._share_scales else self._num_vars,
                self._dimensionality)
            shape_kwarg = dict(shape=shape) if callable(self._scale_init) else dict()
            self._scale_variable = tf.get_variable(
                "Scale", initializer=tf.maximum(self._scale_init, self._min_scale),
                dtype=conf.dtype,
                collections=[SPNGraphKeys.DIST_SCALE, SPNGraphKeys.DIST_PARAMETERS,
                             tf.GraphKeys.GLOBAL_VARIABLES],
                trainable=self._trainable_scale, **shape_kwarg)

    def _variable_shape(self, num_vars, num_components, dimensionality):
        """The shape of the variables within this node. """
        return [num_vars, num_components] + ([] if dimensionality == 1 else [dimensionality])

    def initialize(self):
        """Provide initializers for mean, variance and total counts """
        return (self._loc_variable.initializer, self._scale_variable.initializer,
                self._total_count_variable.initializer)

    @property
    def variables(self):
        """Returns mean and variance variables. """
        return self._loc_variable, self._scale_variable

    @property
    def loc_variable(self):
        """Tensor holding mean variable. """
        return self._loc_variable

    @property
    def scale_variable(self):
        """Tensor holding variance variable. """
        return self._scale_variable

    @utils.docinherit(Node)
    def serialize(self):
        data = super().serialize()
        data['loc_init'] = self._loc_init
        data['scale_init'] = self._scale_init
        return data

    @utils.docinherit(Node)
    def deserialize(self, data):
        self._loc_init = data['loc_init']
        self._scale_init = data['scale_init']
        super().deserialize(data)

    def assign(self, accum, sum_data, sum_data_squared):
        """
        Assigns new values to variables based on accumulated tensors. It updates the distribution
        parameters based on what can be found in "Online Structure Learning for Sum-Product Networks
        with Gaussian Leaves" by Hsu et al. (2017) https://arxiv.org/pdf/1701.05265.pdf

        Args:
            accum (Tensor): A ``Variable`` with accumulated counts per component.
            sum_data (Tensor): A ``Variable`` with the accumulated sum of data per component.
            sum_data_squared (Tensor): A ``Variable`` with the accumulated sum of squares of data
                                       per component.
        Returns:
            Tuple: A tuple containing assignment operations for the new total counts, the variance
            and the mean.
        """
        n = tf.maximum(self._total_count_variable, tf.ones_like(self._total_count_variable))
        k = accum
        mean = self._compute_hard_em_mean(k, n, sum_data)
        with tf.control_dependencies([n, mean]):
            return (
                tf.assign_add(self._total_count_variable, k),
                tf.assign(self._loc_variable, mean) if self._trainable_loc else tf.no_op())

    @utils.docinherit(Node)
    @utils.lru_cache
    def _compute_hard_em_update(self, counts):
        counts_reshaped = tf.reshape(counts, (-1, self._num_vars, self._num_components))
        # Determine accumulates per component
        accum = tf.reduce_sum(counts_reshaped, axis=0)

        # Tile the feed
        # tiled_feed = self._tile_num_components(self._feed)
        tiled_feed = self._preprocessed_feed()
        data_per_component = tf.multiply(counts_reshaped, tiled_feed, name="DataPerComponent")
        squared_data_per_component = tf.multiply(
            counts_reshaped, tf.square(tiled_feed), name="SquaredDataPerComponent")
        sum_data = tf.reduce_sum(data_per_component, axis=0)
        sum_data_squared = tf.reduce_sum(squared_data_per_component, axis=0)
        return {'accum': accum, "sum_data": sum_data, "sum_data_squared": sum_data_squared}

    def _compute_hard_em_mean(self, k, n, sum_data):
        return (n * self._loc_variable + sum_data) / (n + k)


class NormalLeaf(LocationScaleLeaf):
    """A node representing multiple uni-variate Gaussian distributions for continuous input
    variables. Each variable will have *k* Gaussian components. Each Gaussian component has its
    own location (mean) and scale (standard deviation). These parameters can be learned or fixed.
    Lack of evidence must be provided explicitly through feeding_
    :meth:`~libspn.NormalLeaf.evidence`.

    Args:
        feed (Tensor): Tensor feeding this node or ``None``. If ``None``,
                       an internal placeholder will be used to feed this node.
        num_vars (int): Number of random variables.
        num_components (int): Number of components per random variable.
        name (str): Name of the node
        initialization_data (numpy.ndarray): Numpy array containing the data for mean and variance
                                             initialization.
        estimate_variance_init (bool): Boolean marking whether to estimate variance from
                                       ``initialization_data``.
        loc_init (float or numpy.ndarray): If a float and there's no ``initialization_data``,
                                            all components are initialized with ``loc_init``. If
                                            an numpy.ndarray, must have shape
                                            ``[num_vars, num_components]``.
        scale_init (float): If a float and there's no ``initialization_data``, scales are
                            initialized with ``variance_init``.
        trainable_loc (bool): Whether to make the location ``Variable`` trainable.
        trainable_scale (bool): Whether to make the scale ``Variable`` trainable.
        use_prior (bool): Use prior when initializing variances from data.
                          See :meth:`~libspn.GaussianLeaf.initialize_from_quantiles`.
        prior_alpha (float): Alpha parameter for variance prior.
                             See :meth:`~libspn.GaussianLeaf.initialize_from_quantiles`.
        prior_beta (float): Beta parameter for variance prior.
                             See :meth:`~libspn.GaussianLeaf.initialize_from_quantiles`.
        min_stddev (float): Minimum value for standard devation. Used for avoiding numerical
                            instabilities when computing (log) pdfs.
        evidence_indicator_feed (Tensor): Tensor feeding this node's evidence indicator. If
                                          ``None``, an internal placeholder with default value will
                                          be created.
    """

    def __init__(self, feed=None, evidence_indicator_feed=None, num_vars=1, num_components=2,
                 initialization_data=None, estimate_scale=True, total_counts_init=1,
                 trainable_loc=True, trainable_scale=True,
                 loc_init=Equidistant(),
                 scale_init=1.0, use_prior=False, prior_alpha=2.0, prior_beta=3.0,
                 min_scale=1e-2, softplus_scale=True, name="NormalLeaf",
                 share_locs_across_vars=False, share_scales=False, samplewise_normalization=False):
        self._initialization_data = initialization_data
        self._estimate_scale_init = estimate_scale
        self._use_prior = use_prior
        self._prior_alpha = prior_alpha
        self._prior_beta = prior_beta
        super().__init__(
            feed=feed, name=name, dimensionality=1, num_components=num_components,
            num_vars=num_vars, evidence_indicator_feed=evidence_indicator_feed,
            softplus_scale=softplus_scale, loc_init=loc_init, trainable_loc=trainable_loc,
            trainable_scale=trainable_scale, scale_init=scale_init, min_scale=min_scale,
            total_counts_init=total_counts_init, share_locs_across_vars=share_locs_across_vars,
            share_scales=share_scales, samplewise_normalization=samplewise_normalization)
        self._initialization_data = None

    def init_variables(self, shape, loc_init, scale_init, softplus_scale):
        super().init_variables(shape, loc_init, scale_init, softplus_scale)
        if self._initialization_data is not None:
            if len(self._initialization_data.shape) != 2:
                raise ValueError("Initialization data must of rank 2")
            if self._initialization_data.shape[1] != shape[0]:
                raise ValueError("Initialization data samples must have as many components as "
                                 "there are variables in this NormalLeaf node.")
            self.initialize_from_quantiles(
                self._initialization_data, num_quantiles=shape[1],
                estimate_variance=self._estimate_scale_init, use_prior=self._use_prior,
                prior_alpha=self._prior_alpha, prior_beta=self._prior_beta)

    def _create_dist(self):
        if self._softplus_scale:
            return tfd.NormalWithSoftplusScale(self._loc_variable, self._scale_variable)
        return tfd.Normal(self._loc_variable, self._scale_variable)

    def initialize_from_quantiles(self, data, num_quantiles, estimate_variance=True,
                                  use_prior=False, prior_alpha=2.0, prior_beta=3.0):
        """Initializes the data from its quantiles per variable using the method described in
        Poon&Domingos UAI'11.

        Args:
            data (numpy.ndarray): Numpy array of shape [batch, num_vars] containing the data to
            initialize the means and variances.
            estimate_variance (bool): Whether to use the variance estimate.
            use_prior (False):  If ``True``, puts an inverse Gamma prior on the variance with
                                parameters ``prior_beta`` and ``prior_alpha``.
            prior_alpha (float): The alpha parameter of the inverse Gamma prior.
            prior_beta (float): The beta parameter of the inverse Gamma prior.
        """
        values_per_quantile = self._split_in_quantiles(data, num_quantiles)

        means = [np.mean(values, axis=0) for values in values_per_quantile]

        if use_prior:
            sum_sq = [np.sum((x - np.expand_dims(mu, 0)) ** 2, axis=0)
                      for x, mu in zip(values_per_quantile, means)]
            variance = [(2 * prior_beta + ssq) / (2 * prior_alpha + 2 + ssq.shape[0])
                        for ssq in sum_sq]
        else:
            variance = [np.var(values, axis=0) for values in values_per_quantile]

        self._loc_init = np.stack(means, axis=-1)
        variance = np.stack(variance, axis=-1)
        if estimate_variance:
            self._scale_init = _softplus_inverse_np(np.sqrt(variance)) if self._softplus_scale \
                else np.sqrt(variance)
            self._scale_init = np.maximum(self._scale_init, self._min_scale)

    def _split_in_quantiles(self, data, num_quantiles):
        """Given data, finds quantiles of it along zeroth axis. Each quantile is assigned to a
        component. Taken from "Sum-Product Networks: A New Deep Architecture"
        (Poon and Domingos 2012), https://arxiv.org/abs/1202.3732.

        Params:
            data (numpy.ndarray): Numpy array containing data to split into quantiles.

        Returns:
            Data per quantile: a list of numpy.ndarray corresponding to quantiles.
        """
        if self._sample_wise_normalization:
            reduce_axes = tuple(range(1, len(data.shape)))
            data = (data - np.mean(data, axis=reduce_axes)) / np.std(data, axis=reduce_axes)
        batch_size = data.shape[0]
        quantile_sections = np.arange(
            batch_size // num_quantiles, batch_size, int(np.ceil(batch_size / num_quantiles)))
        sorted_features = np.sort(data, axis=0).astype(tf.DType(conf.dtype).as_numpy_dtype())
        values_per_quantile = np.split(
            sorted_features, indices_or_sections=quantile_sections, axis=0)
        return values_per_quantile

    @utils.docinherit(Node)
    @utils.lru_cache
    def _compute_hard_em_update(self, counts):
        counts_reshaped = tf.reshape(counts, (-1, self._num_vars, self._num_components))
        # Determine accumulates per component
        accum = tf.reduce_sum(counts_reshaped, axis=0)

        # Tile the feed
        # tiled_feed = self._tile_num_components(self._feed)
        tiled_feed = self._preprocessed_feed()
        data_per_component = tf.multiply(counts_reshaped, tiled_feed, name="DataPerComponent")
        squared_data_per_component = tf.multiply(
            counts_reshaped, tf.square(tiled_feed), name="SquaredDataPerComponent")
        sum_data = tf.reduce_sum(data_per_component, axis=0)
        sum_data_squared = tf.reduce_sum(squared_data_per_component, axis=0)
        return {'accum': accum, "sum_data": sum_data, "sum_data_squared": sum_data_squared}

    def _compute_gradient(self, incoming_grad):
        """
        Computes gradients for location and scales of the distributions propagated gradients via
        chain rule. The incoming gradient is the summed gradient of the parents of this node.

        Args:
             incoming_grad (Tensor): A ``Tensor`` holding the summed gradient of the parents of this
                                     node
        Returns:
            Tuple: A ``Tensor`` holding the gradient of the locations and a ``Tensor`` holding the
                   gradient of the scales.
        """
        incoming_grad = tf.reshape(incoming_grad, (-1, self._num_vars, self._num_components))
        # tiled_feed = self._tile_num_components(self._feed)
        tiled_feed = self._preprocessed_feed()
        mean = tf.expand_dims(self._loc_variable, 0)

        # Compute the actual variance of the Gaussian without softplus
        if self._softplus_scale:
            scale = tf.nn.softplus(tf.expand_dims(self._scale_variable, 0))
        else:
            scale = tf.expand_dims(self._scale_variable, 0)
        var = tf.square(scale)

        # Compute the gradient
        one_over_var = tf.reciprocal(var)
        x_min_mu = tiled_feed - mean
        mean_grad_out = one_over_var * x_min_mu

        var_grad_out = tf.negative(0.5 * one_over_var * (1 - one_over_var * tf.square(x_min_mu)))
        loc_grad = tf.reduce_sum(mean_grad_out * incoming_grad, axis=0)
        var_grad = var_grad_out * incoming_grad

        if self._softplus_scale:
            scale_grad = 2 * var_grad * scale * tf.nn.sigmoid(self._scale_variable)
        else:
            scale_grad = 2 * var_grad * scale
        return loc_grad, tf.reduce_sum(scale_grad, axis=0)

    def assign(self, accum, sum_data, sum_data_squared):
        """
        Assigns new values to variables based on accumulated tensors. It updates the distribution
        parameters based on what can be found in "Online Structure Learning for Sum-Product Networks
        with Gaussian Leaves" by Hsu et al. (2017) https://arxiv.org/pdf/1701.05265.pdf

        Args:
            accum (Tensor): A ``Variable`` with accumulated counts per component.
            sum_data (Tensor): A ``Variable`` with the accumulated sum of data per component.
            sum_data_squared (Tensor): A ``Variable`` with the accumulated sum of squares of data
                                       per component.
        Returns:
            Tuple: A tuple containing assignment operations for the new total counts, the variance
            and the mean.
        """
        n = tf.maximum(self._total_count_variable, tf.ones_like(self._total_count_variable))
        k = accum
        mean = self._compute_hard_em_mean(k, n, sum_data)

        current_var = tf.square(self.scale_variable) if not self._softplus_scale else \
            tf.square(tf.nn.softplus(self._scale_variable))
        variance = (n * current_var + sum_data_squared -
                    2 * self.loc_variable * sum_data + k * tf.square(self.loc_variable)) / \
                   (n + k) - tf.square(mean - self._loc_variable)

        scale = tf.sqrt(variance)
        if self._softplus_scale:
            scale = tfd.softplus_inverse(scale)
        scale = tf.maximum(scale, self._min_scale)
        with tf.control_dependencies([n, mean, scale]):
            return (
                tf.assign_add(self._total_count_variable, k),
                tf.assign(self._scale_variable, scale) if self._trainable_scale else tf.no_op(),
                tf.assign(self._loc_variable, mean) if self._trainable_loc else tf.no_op())

    def assign_add(self, delta_loc, delta_scale):
        """
        Updates distribution parameters by adding a small delta value.

        Args:
            delta_loc (Tensor): A delta ``Tensor`` for the locations of the distributions.
            delta_scale (Tensor): A delta ``Tensor`` for the scales of the distributions.
        Returns:
             Tuple: An update ``Op`` for the locations and an update ``Op`` for the scales.
        """
        new_var = tf.maximum(self._scale_variable + delta_scale, self._min_scale)
        with tf.control_dependencies([new_var]):
            update_variance = tf.assign(self._scale_variable, new_var)
        return tf.assign_add(self._loc_variable, delta_loc), update_variance


class MultivariateNormalDiagLeaf(LocationScaleLeaf):
    """A node representing multiple uni-variate Gaussian distributions for continuous input
    variables. Each variable will have *k* Gaussian components. Each Gaussian component has its
    own location (mean) and scale (standard deviation). These parameters can be learned or fixed.
    Lack of evidence must be provided explicitly through feeding
    :meth:`~libspn.GaussianLeaf.evidence`.

    Args:
        feed (Tensor): Tensor feeding this node or ``None``. If ``None``,
                       an internal placeholder will be used to feed this node.
        num_vars (int): Number of random variables.
        num_components (int): Number of components per random variable.
        name (str): Name of the node
        loc_init (float or numpy.ndarray): If a float and there's no ``initialization_data``,
                                            all components are initialized with ``loc_init``. If
                                            an numpy.ndarray, must have shape
                                            ``[num_vars, num_components]``.
        scale_init (float): If a float and there's no ``initialization_data``, scales are
                            initialized with ``variance_init``.
        min_stddev (float): Minimum value for standard devation. Used for avoiding numerical
                            instabilities when computing (log) pdfs.
        evidence_indicator_feed (Tensor): Tensor feeding this node's evidence indicator. If
                                          ``None``, an internal placeholder with default value will
                                          be created.
    """

    def __init__(self, feed=None, num_vars=1, num_components=2, dimensionality=2,
                 name="MultivariateNormalDiagLeaf", total_counts_init=1,
                 trainable_scale=True, trainable_loc=True,
                 loc_init=tf.initializers.random_uniform(0.0, 1.0),
                 scale_init=1.0, min_scale=1e-2, evidence_indicator_feed=None,
                 softplus_scale=False, share_locs_across_vars=False, share_scales=False):
        super().__init__(
            feed=feed, name=name, dimensionality=dimensionality,
            num_components=num_components, num_vars=num_vars,
            evidence_indicator_feed=evidence_indicator_feed, component_axis=-2,
            total_counts_init=total_counts_init, loc_init=loc_init, trainable_loc=trainable_loc,
            trainable_scale=trainable_scale, scale_init=scale_init, min_scale=min_scale,
            softplus_scale=softplus_scale, share_locs_across_vars=share_locs_across_vars,
            share_scales=share_scales)

    def _create_dist(self):
        if self._softplus_scale:
            return tfd.MultivariateNormalDiag(self._loc_variable, self._scale_variable)
        return tfd.MultivariateNormalDiagWithSoftplusScale(self._loc_variable, self._scale_variable)

    @utils.docinherit(Node)
    @utils.lru_cache
    def _compute_hard_em_update(self, counts):
        counts_reshaped = tf.reshape(counts, (-1, self._num_vars, self._num_components, 1))
        # Determine accumulates per component
        accum = tf.reduce_sum(counts_reshaped, axis=0)

        # Tile the feed
        # tiled_feed = self._tile_num_components(self._feed)
        tiled_feed = self._preprocessed_feed()
        # Accumulate data (broadcast along dim axis)
        data_per_component = tf.multiply(counts_reshaped, tiled_feed, name="DataPerComponent")
        squared_data_per_component = tf.multiply(
            counts_reshaped, tf.square(tiled_feed), name="SquaredDataPerComponent")
        sum_data = tf.reduce_sum(data_per_component, axis=0)
        sum_data_squared = tf.reduce_sum(squared_data_per_component, axis=0)
        return {'accum': accum, "sum_data": sum_data, "sum_data_squared": sum_data_squared}

    def assign(self, accum, sum_data, sum_data_squared):
        """
        Assigns new values to variables based on accumulated tensors. It updates the distribution
        parameters based on what can be found in "Online Structure Learning for Sum-Product Networks
        with Gaussian Leaves" by Hsu et al. (2017) https://arxiv.org/pdf/1701.05265.pdf

        Args:
            accum (Tensor): A ``Variable`` with accumulated counts per component.
            sum_data (Tensor): A ``Variable`` with the accumulated sum of data per component.
            sum_data_squared (Tensor): A ``Variable`` with the accumulated sum of squares of data
                                       per component.
        Returns:
            Tuple: A tuple containing assignment operations for the new total counts, the variance
            and the mean.
        """
        n = tf.expand_dims(
            tf.maximum(self._total_count_variable, tf.ones_like(self._total_count_variable)),
            axis=-1)
        k = tf.expand_dims(accum, axis=-1)
        mean = (n * self._loc_variable + sum_data) / (n + k)

        current_var = tf.square(self.scale_variable) if not self._softplus_scale else \
            tf.square(tf.nn.softplus(self._scale_variable))
        variance = (n * current_var + sum_data_squared -
                    2 * self.loc_variable * sum_data + k * tf.square(self.loc_variable)) / \
                   (n + k) - tf.square(mean - self._loc_variable)

        scale = tf.sqrt(variance)
        if self._softplus_scale:
            scale = tfd.softplus_inverse(scale)
        scale = tf.maximum(scale, self._min_scale)
        with tf.control_dependencies([n, mean, scale]):
            return (
                tf.assign_add(self._total_count_variable, tf.squeeze(k, axis=-1)),
                tf.assign(self._scale_variable, scale) if self._trainable_scale else tf.no_op(),
                tf.assign(self._loc_variable, mean) if self._trainable_loc else tf.no_op())


class LaplaceLeaf(LocationScaleLeaf):

    def __init__(self, feed=None, num_vars=1, num_components=2, name="LaplaceLeaf",
                 trainable_scale=True, trainable_loc=True,
                 loc_init=Equidistant(), scale_init=1.0,
                 min_scale=1e-2, evidence_indicator_feed=None, softplus_scale=False,
                 share_locs_across_vars=False, share_scales=False):
        super().__init__(
            feed=feed, evidence_indicator_feed=evidence_indicator_feed,
            num_vars=num_vars, num_components=num_components, trainable_loc=trainable_loc,
            trainable_scale=trainable_scale, loc_init=loc_init, scale_init=scale_init,
            min_scale=min_scale, softplus_scale=softplus_scale, name=name, dimensionality=1,
            share_locs_across_vars=share_locs_across_vars, share_scales=share_scales)

    def _create_dist(self):
        if self._softplus_scale:
            return tfd.LaplaceWithSoftplusScale(self._loc_variable, self._scale_variable)
        return tfd.Laplace(self._loc_variable, self._scale_variable)


class MultivariateCauchyDiagLeaf(LocationScaleLeaf):

    def __init__(self, feed=None, num_vars=1, num_components=2, dimensionality=2,
                 name="MultivariateCauchyDiagLeaf", total_counts_init=1,
                 trainable_scale=True, trainable_loc=True,
                 loc_init=tf.initializers.random_uniform(0.0, 1.0),
                 scale_init=1.0, min_scale=1e-2, evidence_indicator_feed=None,
                 softplus_scale=False, share_locs_across_vars=False, share_scales=False):
        super().__init__(
            feed=feed, name=name, dimensionality=dimensionality,
            num_components=num_components, num_vars=num_vars,
            evidence_indicator_feed=evidence_indicator_feed, component_axis=-2,
            total_counts_init=total_counts_init, loc_init=loc_init, trainable_loc=trainable_loc,
            trainable_scale=trainable_scale, scale_init=scale_init, min_scale=min_scale,
            softplus_scale=softplus_scale, share_locs_across_vars=share_locs_across_vars,
            share_scales=share_scales)

    def _create_dist(self):
        scale_diag = tf.nn.softplus(self._scale_variable) if self._softplus_scale \
            else self._scale_variable
        scale = distribution_util.make_diag_scale(
            loc=self._loc_variable,
            scale_diag=scale_diag,
            validate_args=False,
            assert_positive=False)
        batch_shape, event_shape = distribution_util.shapes_from_loc_and_scale(
            self._loc_variable, scale)

        return tfp.TransformedDistribution(
            distribution=tfp.Cauchy(
                loc=tf.zeros([], dtype=scale.dtype),
                scale=tf.ones([], dtype=scale.dtype)),
            bijector=bijectors.AffineLinearOperator(
                shift=self._loc_variable, scale=scale),
            batch_shape=batch_shape,
            event_shape=event_shape,
            name="MultivariateCauchyDiag" + ("SoftplusScale" if self._softplus_scale else ""))

    def mode(self):
        return self._loc_variable


class CauchyLeaf(LocationScaleLeaf):

    def __init__(self, feed=None, num_vars=1, num_components=2, name="CauchyLeaf",
                 trainable_scale=True, trainable_loc=True,
                 loc_init=Equidistant(), scale_init=1.0,
                 min_scale=1e-2, evidence_indicator_feed=None, softplus_scale=False,
                 share_scales=False, share_locs_across_vars=False):
        super().__init__(
            feed=feed, evidence_indicator_feed=evidence_indicator_feed,
            num_vars=num_vars, num_components=num_components, trainable_loc=trainable_loc,
            trainable_scale=trainable_scale, loc_init=loc_init, scale_init=scale_init,
            min_scale=min_scale, softplus_scale=softplus_scale, name=name, dimensionality=1,
            share_scales=share_scales, share_locs_across_vars=share_locs_across_vars)

    def _create_dist(self):
        if self._softplus_scale:
            return tfd.Cauchy(self._loc_variable, tf.nn.softplus(self._scale_variable))
        return tfd.Cauchy(self._loc_variable, self._scale_variable)


class StudentTLeaf(LocationScaleLeaf):

    def __init__(self, feed=None, num_vars=1, num_components=2, name="StudentTLeaf",
                 trainable_scale=True, trainable_loc=True,
                 loc_init=Equidistant(), scale_init=1.0,
                 min_scale=1e-2, evidence_indicator_feed=None, softplus_scale=False,
                 trainable_df=False, df_init=tf.initializers.constant(1.0),
                 share_locs_across_vars=False, share_scales=False, share_dfs=False):
        self._trainable_df = trainable_df
        self._df_init = df_init
        self._share_dfs = share_dfs

        super().__init__(
            feed=feed, evidence_indicator_feed=evidence_indicator_feed,
            num_vars=num_vars, num_components=num_components, trainable_loc=trainable_loc,
            trainable_scale=trainable_scale, loc_init=loc_init, scale_init=scale_init,
            min_scale=min_scale, softplus_scale=softplus_scale, name=name, dimensionality=1,
            share_locs_across_vars=share_locs_across_vars, share_scales=share_scales)

    def _create_dist(self):
        if self._softplus_scale:
            return tfd.StudentTWithAbsDfSoftplusScale(
                self._df_variable, self._loc_variable, self._scale_variable)
        return tfd.StudentT(self._df_variable, self._loc_variable, self._scale_variable)

    @utils.docinherit(Node)
    def _create(self):
        super()._create()
        with tf.variable_scope(self._name):
            # Initialize locations
            shape = self._variable_shape(
                1 if self._share_dfs else self._num_vars,
                1 if self._share_dfs else self._num_components,
                self._dimensionality)
            shape_kwarg = dict(shape=shape) if callable(self._df_init) else dict()
            self._df_variable = tf.get_variable(
                "Df", initializer=self._df_init, dtype=conf.dtype,
                collections=[SPNGraphKeys.DIST_DF, SPNGraphKeys.DIST_PARAMETERS,
                             tf.GraphKeys.GLOBAL_VARIABLES],
                trainable=self._trainable_df, **shape_kwarg)

    @property
    def variables(self):
        """Returns mean and variance variables. """
        return self._df_variable, self._loc_variable, self._scale_variable


class TruncatedNormalLeaf(LocationScaleLeaf):

    def __init__(self, feed=None, num_vars=1, num_components=2, name="TruncatedNormalLeaf",
                 trainable_scale=True, trainable_loc=True,
                 loc_init=Equidistant(), scale_init=1.0,
                 min_scale=1e-2, evidence_indicator_feed=None, softplus_scale=False,
                 truncate_min=0.0, truncate_max=1.0, share_locs_across_vars=False,
                 share_scales=False):
        self._truncate_min = truncate_min
        self._truncate_max = truncate_max
        super().__init__(
            feed=feed, evidence_indicator_feed=evidence_indicator_feed,
            num_vars=num_vars, num_components=num_components, trainable_loc=trainable_loc,
            trainable_scale=trainable_scale, loc_init=loc_init, scale_init=scale_init,
            min_scale=min_scale, softplus_scale=softplus_scale, name=name, dimensionality=1,
            share_locs_across_vars=share_locs_across_vars, share_scales=share_scales)

    def _create_dist(self):
        if self._softplus_scale:
            return tfp.TruncatedNormal(
                self._loc_variable, tf.nn.softplus(self._scale_variable), low=self._truncate_min,
                high=self._truncate_max)
        return tfp.TruncatedNormal(
            self._loc_variable, self._scale_variable,
            low=self._truncate_min, high=self._truncate_max)


def _softplus_inverse_np(x):
    return np.log(1 - np.exp(-x)) + x


def logit(x, name="Logit"):
    with tf.name_scope(name):
        return -tf.log(tf.reciprocal(x) - 1)


