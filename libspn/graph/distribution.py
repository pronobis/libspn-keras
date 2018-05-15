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
    own mean and variance. These parameters can be learned or fixed. Lack of evidence must be
    provided explicitly through feeding :meth:`~libspn.GaussianLeaf.evidence`.

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
        mean_init (float or numpy.ndarray): If a float and there's no ``initialization_data``,
                                            all components are initialized with ``mean_init``. If
                                            an numpy.ndarray, must have shape
                                            ``[num_vars, num_components]``.
        variance_init (float): If a float and there's no ``initialization_data``, variances are
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
                 learn_dist_params=False, train_var=True, mean_init=0.0, variance_init=1.0,
                 train_mean=True, use_prior=False, prior_alpha=2.0, prior_beta=3.0, min_stddev=1e-2,
                 evidence_indicator_feed=None):
        self._mean_variable = None
        self._variance_variable = None
        self._num_vars = num_vars
        self._num_components = num_components

        # Initial value for means
        if isinstance(mean_init, float):
            self._mean_init = tf.ones((num_vars, num_components), dtype=conf.dtype) * mean_init
        else:
            self._mean_init = mean_init

        # Initial values for variances.
        self._variance_init = tf.ones((num_vars, num_components), dtype=conf.dtype) * variance_init
        self._learn_dist_params = learn_dist_params
        self._min_stddev = min_stddev
        self._min_var = np.square(min_stddev)
        if initialization_data is not None:
            self.initialize_from_quantiles(
                initialization_data, estimate_variance=estimate_variance_init, use_prior=use_prior,
                prior_alpha=prior_alpha, prior_beta=prior_beta)
        super().__init__(feed=feed, name=name)
        self.attach_evidence_indicator(evidence_indicator_feed)

        var_shape = (num_vars, num_components)
        self._total_count_variable = self._total_accumulates(total_counts_init, var_shape)
        self._train_var = train_var
        self._train_mean = train_mean

    def initialize(self):
        """Provide initializers for mean, variance and total counts """
        return (self._mean_variable.initializer, self._variance_variable.initializer,
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
        return self._mean_variable, self._variance_variable

    @property
    def mean_variable(self):
        """Tensor holding mean variable. """
        return self._mean_variable

    @property
    def variance_variable(self):
        """Tensor holding variance variable. """
        return self._variance_variable

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
            tf.cast(tf.ones_like(self._placeholder), tf.bool), shape=[None, self._num_vars])

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
        self._mean_variable = tf.Variable(
            self._mean_init, dtype=conf.dtype, collections=['spn_distribution_parameters'])
        self._variance_variable = tf.Variable(
            tf.maximum(self._variance_init, self._min_var), dtype=conf.dtype,
            collections=['spn_distribution_parameters'])
        self._dist = tfd.Normal(
            self._mean_variable, tf.maximum(tf.sqrt(self._variance_variable), self._min_stddev))

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

        self._mean_init = np.stack(means, axis=-1)
        if estimate_variance:
            self._variance_init = np.maximum(np.stack(variance, axis=-1), self._min_var)

    @utils.docinherit(Node)
    def serialize(self):
        data = super().serialize()
        data['num_vars'] = self._num_vars
        data['num_components'] = self._num_components
        data['mean_init'] = self._mean_init
        data['variance_init'] = self._variance_init
        return data

    @utils.docinherit(Node)
    def deserialize(self, data):
        self._num_vars = data['num_vars']
        self._num_components = data['num_components']
        self._mean_init = data['mean_init']
        self._variance_init = data['variance_init']
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
    def _compute_value(self, step=None):
        return self._evidence_mask(
            self._dist.prob(self._tile_num_components(self._feed)), tf.ones_like)

    @utils.docinherit(Node)
    def _compute_log_value(self):
        return self._evidence_mask(
            self._dist.log_prob(self._tile_num_components(self._feed)), tf.zeros_like)

    @utils.docinherit(Node)
    def _compute_scope(self):
        return [Scope(self, i) for i in range(self._num_vars) for _ in range(self._num_components)]

    @utils.docinherit(Node)
    def _compute_mpe_state(self, counts):
        # MPE state can be found by taking the mean of the mixture components that are 'selected'
        # by the counts
        counts_reshaped = tf.reshape(counts, (-1, self._num_vars, self._num_components))
        indices = tf.argmax(counts_reshaped, axis=-1) + tf.expand_dims(
            tf.range(self._num_vars, dtype=tf.int64) * self._num_components, axis=0)
        return tf.gather(tf.reshape(self._mean_variable, (-1,)), indices=indices, axis=0)

    @utils.docinherit(Node)
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
        mean = (n * self._mean_variable + sum_data) / (n + k)

        variance = (n * self._variance_variable +
                    sum_data_squared - 2 * self.mean_variable * sum_data +
                    k * tf.square(self.mean_variable)) / \
                   (n + k) - tf.square(mean - self._mean_variable)
        variance = tf.maximum(variance, self._min_var)
        with tf.control_dependencies([n, mean, variance]):
            return (
                tf.assign_add(self._total_count_variable, k),
                tf.assign(self._variance_variable, variance) if self._train_var else tf.no_op(),
                tf.assign(self._mean_variable, mean) if self._train_mean else tf.no_op())
