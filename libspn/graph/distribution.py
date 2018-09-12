# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------
import tensorflow as tf
from libspn.graph.scope import Scope
from libspn.graph.node import VarNode, Node
from libspn import conf
from libspn import utils
import numpy as np
import tensorflow.contrib.distributions as tfd


class GaussianLeaf(VarNode):
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
        train_mean (bool): Whether to make the mean ``Variable`` trainable.
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

    def __init__(self, feed=None, num_vars=1, num_components=2, name="GaussianLeaf",
                 initialization_data=None, estimate_variance_init=True, total_counts_init=1,
                 learn_dist_params=False, train_var=True, loc_init=0.0, scale_init=1.0,
                 train_mean=True, use_prior=False, prior_alpha=2.0, prior_beta=3.0, min_stddev=1e-2,
                 evidence_indicator_feed=None, softplus_scale=False, share_scales=False,
                 normalized=True):
        self._loc_variable = None
        self._scale_variable = None
        self._num_vars = num_vars
        self._num_components = num_components
        self._softplus_scale = softplus_scale
        self._train_var = train_var
        self._train_mean = train_mean
        self._share_scales = share_scales
        self._normalized = normalized

        # Initial value for means
        if isinstance(loc_init, float):
            self._loc_init = tf.ones((num_vars, num_components), dtype=conf.dtype) * loc_init
        else:
            self._loc_init = loc_init

        # Initial values for variances.
        self._scale_init = tf.ones([1, 1]) * scale_init if share_scales else \
            tf.ones((num_vars, num_components), dtype=conf.dtype) * scale_init
        self._learn_dist_params = learn_dist_params
        self._min_stddev = min_stddev if not softplus_scale else np.log(np.exp(min_stddev) - 1)
        if initialization_data is not None:
            self.initialize_from_quantiles(
                initialization_data, estimate_variance=estimate_variance_init, use_prior=use_prior,
                prior_alpha=prior_alpha, prior_beta=prior_beta)
        super().__init__(feed=feed, name=name)
        self.attach_evidence_indicator(evidence_indicator_feed)

        var_shape = (num_vars, num_components)
        self._total_count_variable = self._total_accumulates(total_counts_init, var_shape)

    def initialize(self):
        """Provide initializers for mean, variance and total counts """
        return (self._loc_variable.initializer, self._scale_variable.initializer,
                self._total_count_variable.initializer)

    @property
    def num_vars(self):
        return self._num_vars

    @property
    def num_components(self):
        """Number of components per variable. """
        return self._num_components

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

    @property
    def evidence(self):
        """Tensor holding evidence placeholder. """
        return self._evidence_indicator

    @property
    def learn_distribution_parameters(self):
        """Flag indicating whether this node learns its parameters. """
        return self._learn_dist_params

    @utils.docinherit(VarNode)
    def _create_placeholder(self):
        return tf.placeholder(conf.dtype, [None, self._num_vars])

    def _create_evidence_indicator(self):
        """Creates a placeholder with default value. The default value is a ``Tensor`` of shape
        [batch, num_vars] filled with ``True``.

        Return:
            Evidence indicator placeholder: a placeholder ``Tensor`` set to True for each variable.
        """
        return tf.placeholder_with_default(
            tf.cast(tf.ones_like(self.feed), tf.bool), shape=[None, self._num_vars])

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
    def _create(self):
        super()._create()
        self._loc_variable = tf.Variable(
            self._loc_init, dtype=conf.dtype, collections=['spn_distribution_parameters'],
            trainable=self._train_mean)
        self._scale_variable = tf.Variable(
            tf.maximum(self._scale_init, self._min_stddev), dtype=conf.dtype,
            collections=['spn_distribution_parameters'], trainable=self._train_var)
        if self._softplus_scale:
            self._dist = tfd.NormalWithSoftplusScale(self._loc_variable, self._scale_variable)
        else:
            self._dist = tfd.Normal(self._loc_variable, self._scale_variable)

    def initialize_from_quantiles(self, data, estimate_variance=True, use_prior=False,
                                  prior_alpha=2.0, prior_beta=3.0):
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
        if len(data.shape) != 2 or data.shape[1] != self._num_vars:
            raise ValueError("Data should be of rank 2 and contain equally many variables as this "
                             "GaussianLeaf node.")

        values_per_quantile = self._split_in_quantiles(data)

        means = [np.mean(values, axis=0) for values in values_per_quantile]

        if use_prior:
            sum_sq = [np.sum((x - np.expand_dims(mu, 0)) ** 2, axis=0)
                      for x, mu in zip(values_per_quantile, means)]
            variance = [(2 * prior_beta + ssq) / (2 * prior_alpha + 2 + ssq.shape[0])
                        for ssq in sum_sq]
        else:
            variance = [np.var(values, axis=0) for values in values_per_quantile]

        self._loc_init = np.stack(means, axis=-1)
        if estimate_variance:
            if self._softplus_scale:
                self._scale_init = np.log(np.exp(np.sqrt(np.stack(variance, axis=-1))) - 1)
            else:
                self._scale_init = np.maximum(
                    np.stack(np.sqrt(variance), axis=-1), self._min_stddev)

    @property
    def scale(self):
        return tf.nn.softplus(self.scale_variable) if self._softplus_scale else self.scale_variable

    @utils.docinherit(Node)
    def serialize(self):
        data = super().serialize()
        data['num_vars'] = self._num_vars
        data['num_components'] = self._num_components
        data['mean_init'] = self._loc_init
        data['variance_init'] = self._scale_init
        return data

    @utils.docinherit(Node)
    def deserialize(self, data):
        self._num_vars = data['num_vars']
        self._num_components = data['num_components']
        self._loc_init = data['mean_init']
        self._scale_init = data['variance_init']
        super().deserialize(data)

    def _total_accumulates(self, init_val, shape):
        """Creates a ``Variable`` that holds the counts per component.

        Return:
              Counts per component: ``Variable`` holding counts per component.
        """
        init = utils.broadcast_value(init_val, shape, dtype=conf.dtype)
        return tf.Variable(init, name=self.name + "TotalCounts", trainable=False)

    def _split_in_quantiles(self, data):
        """Given data, finds quantiles of it along zeroth axis. Each quantile is assigned to a
        component. Taken from "Sum-Product Networks: A New Deep Architecture"
        (Poon and Domingos 2012), https://arxiv.org/abs/1202.3732.

        Params:
            data (numpy.ndarray): Numpy array containing data to split into quantiles.

        Returns:
            Data per quantile: a list of numpy.ndarray corresponding to quantiles.
        """
        batch_size = data.shape[0]
        quantile_sections = np.arange(
            batch_size // self._num_components, batch_size,
            int(np.ceil(batch_size / self._num_components)))
        sorted_features = np.sort(data, axis=0).astype(tf.DType(conf.dtype).as_numpy_dtype())
        values_per_quantile = np.split(
            sorted_features, indices_or_sections=quantile_sections, axis=0)
        return values_per_quantile

    @utils.docinherit(Node)
    def _compute_out_size(self):
        return self._num_vars * self._num_components

    @utils.lru_cache
    def _tile_num_components(self, tensor):
        """Tiles a ``Tensor`` so that its last axis contains ``num_components`` repetitions of the
        original values. If the incoming tensor's last dim size equals 1, it will tile along this
        axis. If the incoming tensor's last dim size is not equal to 1, it will append a dimension
        of size 1 and then perform tiling.

        Args:
            tensor (Tensor): The tensor to tile ``num_components`` times.

        Return:
            Tiled tensor: Input tensor tiled ``num_components`` times along last axis.
        """
        if tensor.shape[-1].value != 1:
            tensor = tf.expand_dims(tensor, axis=-1)
        rank = len(tensor.shape)
        return tf.tile(tensor, [1 for _ in range(rank - 1)] + [self._num_components])

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
        evidence = tf.reshape(self._tile_num_components(self.evidence), out_shape)
        value = tf.reshape(value, out_shape)
        return tf.where(evidence, value, no_evidence_fn(value))

    @utils.docinherit(Node)
    @utils.lru_cache
    def _compute_value(self, step=None):
        return self._evidence_mask(
            self._dist.prob(self._tile_num_components(self._feed)), tf.ones_like)

    @utils.docinherit(Node)
    @utils.lru_cache
    def _compute_log_value(self):
        if self._normalized:
            return self._evidence_mask(
                self._dist.log_prob(self._tile_num_components(self._feed)), tf.zeros_like)
        else:
            return self._evidence_mask(
                self._dist._log_unnormalized_prob(self._tile_num_components(self._feed)),
                tf.zeros_like)

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
        return tf.gather(tf.reshape(self._loc_variable, (-1,)), indices=indices, axis=0)

    @utils.docinherit(Node)
    @utils.lru_cache
    def _compute_hard_em_update(self, counts):
        counts_reshaped = tf.reshape(counts, (-1, self._num_vars, self._num_components))
        # Determine accumulates per component
        accum = tf.reduce_sum(counts_reshaped, axis=0)

        # Tile the feed
        tiled_feed = self._tile_num_components(self._feed)
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
        tiled_feed = self._tile_num_components(self._feed)
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
        parameters based on what can be found in "Online Algorithms for Sum-Product Networks with
        Continuous Variables" by Jaini et al. (2016) http://proceedings.mlr.press/v52/jaini16.pdf

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
        mean = (n * self._loc_variable + sum_data) / (n + k)

        current_var = tf.square(self.scale_variable) if not self._softplus_scale else \
            tf.square(tf.nn.softplus(self._scale_variable))
        variance = (n * current_var + sum_data_squared -
                    2 * self.loc_variable * sum_data + k * tf.square(self.loc_variable)) / \
                   (n + k) - tf.square(mean - self._loc_variable)

        scale = tf.sqrt(variance)
        if self._softplus_scale:
            scale = tfd.softplus_inverse(scale)
        scale = tf.maximum(scale, self._min_stddev)
        with tf.control_dependencies([n, mean, scale]):
            return (
                tf.assign_add(self._total_count_variable, k),
                tf.assign(self._scale_variable, scale) if self._train_var else tf.no_op(),
                tf.assign(self._loc_variable, mean) if self._train_mean else tf.no_op())

    def assign_add(self, delta_loc, delta_scale):
        """
        Updates distribution parameters by adding a small delta value.

        Args:
            delta_loc (Tensor): A delta ``Tensor`` for the locations of the distributions.
            delta_scale (Tensor): A delta ``Tensor`` for the scales of the distributions.
        Returns:
             Tuple: An update ``Op`` for the locations and an update ``Op`` for the scales.
        """
        new_var = tf.maximum(self._scale_variable + delta_scale, self._min_stddev)
        with tf.control_dependencies([new_var]):
            update_variance = tf.assign(self._scale_variable, new_var)
        return tf.assign_add(self._loc_variable, delta_loc), update_variance

    @utils.lru_cache
    def entropy(self):
        return self._dist.entropy()

    @utils.lru_cache
    def cross_entropy(self, normal_dist):
        return self._dist.cross_entropy(normal_dist)

    @utils.lru_cache
    def kl_divergence(self, normal_dist):
        return self._dist.kl_divergence(normal_dist)