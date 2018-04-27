# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------
import tensorflow as tf
from libspn.graph.scope import Scope
from libspn.graph.node import VarNode
from libspn import conf, ContVars
from libspn import utils
import numpy as np
import tensorflow.contrib.distributions as tfd

# Some good sources:
# https://github.com/PhDP/spn/blob/1b837f1293e1098e6d7d908f4647a1d368308833/code/src/spn/SPN.java#L263
# https://github.com/whsu/spn/tree/master/spn


class GaussianLeaf(VarNode):

    def __init__(self, feed=None, num_vars=1, num_components=2, name="GaussianLeaf", data=None,
                 learn_scale=True, total_counts_init=1, learn_dist_params=False,
                 train_var=True, mean_init=0.0, variance_init=1.0, train_mean=True,
                 use_prior=False, prior_alpha=2.0, prior_beta=3.0, min_stddev=1e-4):
        self._external_feed = isinstance(feed, ContVars)
        self._mean_variable = None
        self._variance_variable = None
        self._num_vars = feed.num_vars if self._external_feed else num_vars
        self._num_components = num_components
        if isinstance(mean_init, float):
            self._mean_init = tf.ones((num_vars, num_components), dtype=conf.dtype) * mean_init
        else:
            self._mean_init = mean_init
        self._variance_init = tf.ones((num_vars, num_components), dtype=conf.dtype) * variance_init
        self._learn_dist_params = learn_dist_params
        self._min_stddev = min_stddev
        self._min_var = np.square(min_stddev)
        if data is not None:
            self.learn_from_data(data, learn_scale=learn_scale, use_prior=use_prior,
                                 prior_beta=prior_beta, prior_alpha=prior_alpha)
        super().__init__(feed=feed, name=name)

        var_shape = (num_vars, num_components)
        self._total_count_variable = self._total_accumulates(
            total_counts_init, var_shape)
        self._train_var = train_var
        self._train_mean = train_mean

    def initialize(self):
        return (self._mean_variable.initializer, self._variance_variable.initializer,
                self._total_count_variable.initializer)

    @property
    def variables(self):
        return self._mean_variable, self._variance_variable

    @property
    def mean_variable(self):
        return self._mean_variable

    @property
    def variance_variable(self):
        return self._variance_variable

    @property
    def evidence(self):
        return self._evidence_indicator

    @property
    def learn_distribution_parameters(self):
        return self._learn_dist_params

    def _create_placeholder(self):
        return tf.placeholder(conf.dtype, [None, self._num_vars])

    def _create_evidence_indicator(self):
        return tf.placeholder_with_default(
            tf.cast(tf.ones_like(self._placeholder), tf.bool), shape=[None, self._num_vars])

    def attach_feed(self, feed):
        super().attach_feed(feed)
        if self._external_feed:
            self._evidence_indicator = self._feed.evidence
        else:
            self._evidence_indicator = self._create_evidence_indicator()

    def _create(self):
        super()._create()
        self._mean_variable = tf.Variable(
            self._mean_init, dtype=conf.dtype, collections=['spn_distribution_parameters'])
        self._variance_variable = tf.Variable(
            tf.maximum(self._variance_init, self._min_var), dtype=conf.dtype,
            collections=['spn_distribution_parameters'])
        self._dist = tfd.Normal(
            self._mean_variable, tf.maximum(tf.sqrt(self._variance_variable), self._min_stddev))

    def learn_from_data(self, data, learn_scale=True, use_prior=False, prior_beta=3.0,
                        prior_alpha=2.0):
        """Learns the distribution parameters from data
        Params:
            data: numpy.ndarray of shape [batch, num_vars]
        """
        if len(data.shape) != 2 or data.shape[1] != self._num_vars:
            raise ValueError("Data should be of rank 2 and contain equally many variables as this "
                             "GaussianLeaf node.")

        values_per_quantile = self._values_per_quantile(data)

        means = [np.mean(values, axis=0) for values in values_per_quantile]

        if use_prior:
            sum_sq = [np.sum((x - np.expand_dims(mu, 0)) ** 2, axis=0)
                      for x, mu in zip(values_per_quantile, means)]
            variance = [(2 * prior_beta + ssq) / (2 * prior_alpha + 2 + ssq.shape[0])
                        for ssq in sum_sq]
        else:
            variance = [np.var(values, axis=0) for values in values_per_quantile]

        self._mean_init = np.stack(means, axis=-1)
        if learn_scale:
            self._variance_init = np.maximum(np.stack(variance, axis=-1), self._min_var)

    def serialize(self):
        data = super().serialize()
        data['num_vars'] = self._num_vars
        data['num_components'] = self._num_components
        data['mean_init'] = self._mean_init
        data['variance_init'] = self._variance_init
        return data

    def deserialize(self, data):
        self._num_vars = data['num_vars']
        self._num_components = data['num_components']
        self._mean_init = data['mean_init']
        self._variance_init = data['variance_init']
        super().deserialize(data)

    def _total_accumulates(self, init_val, shape):
        """Creates tensor to accumulate total counts """
        init = utils.broadcast_value(init_val, shape, dtype=conf.dtype)
        return tf.Variable(init, name=self.name + "TotalCounts", trainable=False)

    def _values_per_quantile(self, data):
        """Given data, finds quantiles of it along zeroth axis. Each quantile is assigned to a
        component. Taken from "Sum-Product Networks: A New Deep Architecture"
        (Poon and Domingos 2012), https://arxiv.org/abs/1202.3732

        Params:
            data: Numpy array containing data to split into quantiles.

        Returns:
            A list of numpy arrays corresponding to quantiles.
        """
        batch_size = data.shape[0]
        quantile_sections = np.arange(
            batch_size // self._num_components, batch_size,
            int(np.ceil(batch_size / self._num_components)))
        sorted_features = np.sort(data, axis=0).astype(tf.DType(conf.dtype).as_numpy_dtype())
        values_per_quantile = np.split(
            sorted_features, indices_or_sections=quantile_sections, axis=0)
        return values_per_quantile

    def _compute_out_size(self):
        """Size of output """
        return self._num_vars * self._num_components

    def _get_feed(self):
        return self._feed._as_graph_element() if self._external_feed else self._feed

    def _tile_num_components(self, tensor):
        if tensor.shape[-1].value != 1:
            tensor = tf.expand_dims(tensor, axis=-1)
        rank = len(tensor.shape)
        return tf.tile(tensor, [1 for _ in range(rank - 1)] + [self._num_components])

    def _compute_value_common(self, value, no_evidence_fn):
        out_shape = (-1, self._compute_out_size())
        evidence = tf.reshape(self._tile_num_components(self.evidence), out_shape)
        value = tf.reshape(value, out_shape)
        return tf.where(evidence, value, no_evidence_fn(value))

    def _compute_value(self, step=None):
        """Computes value in non-log space """
        return self._compute_value_common(
            self._dist.prob(self._tile_num_components(self._get_feed())), tf.ones_like)

    def _compute_log_value(self):
        """Computes value in log-space """
        return self._compute_value_common(
            self._dist.log_prob(self._tile_num_components(self._get_feed())), tf.zeros_like)

    def _compute_scope(self):
        """Computes scope """
        # Potentially copy scope from external feed
        node = self._feed if self._external_feed else self
        return [Scope(node, i) for i in range(self._num_vars) for _ in range(self._num_components)]

    def _compute_mpe_state(self, counts):
        # MPE state can be found by taking the mean of the mixture components that are 'selected'
        # by the counts
        counts_reshaped = tf.reshape(counts, (-1, self._num_vars, self._num_components))
        indices = tf.argmax(counts_reshaped, axis=-1) + tf.expand_dims(
            tf.range(self._num_vars, dtype=tf.int64) * self._num_components, axis=0)
        return tf.gather(tf.reshape(self._mean_variable, (-1,)), indices=indices, axis=0)

    def _compute_hard_em_update(self, counts):
        counts_reshaped = tf.reshape(counts, (-1, self._num_vars, self._num_components))
        # Determine accumulates per component
        accum = tf.reduce_sum(counts_reshaped, axis=0)

        # Tile the feed
        tiled_feed = self._tile_num_components(self._get_feed())
        data_per_component = tf.multiply(counts_reshaped, tiled_feed, name="DataPerComponent")
        squared_data_per_component = tf.multiply(counts_reshaped, tf.square(tiled_feed),
                                                 name="SquaredDataPerComponent")
        sum_data = tf.reduce_sum(data_per_component, axis=0)
        sum_data_squared = tf.reduce_sum(squared_data_per_component, axis=0)
        return {'accum': accum, "sum_data": sum_data, "sum_data_squared": sum_data_squared}

    def assign(self, accum, sum_data, sum_data_squared):
        """
        Assigns new values to variables based on accumulated tensors. It updates the distribution
        parameters based on what can be found in "Online Algorithms for Sum-Product Networks with
        Continuous Variables" by Jaini et al. (2016) http://proceedings.mlr.press/v52/jaini16.pdf

        Parameters:
            :param accum: Accumulated counts per component
            :param sum_data: Accumulated sum of data per component
            :param sum_data_squared: Accumulated sum of squares of data per component
        Returns
            :return A tuple containing assignment operations for the new total counts, the variance
            and the mean
        """
        n = tf.maximum(self._total_count_variable, tf.ones_like(self._total_count_variable))
        k = accum
        mean = (n * self._mean_variable + sum_data) / (n + k)

        variance = (n * self._variance_variable +
                    sum_data_squared - 2 * self.mean_variable * sum_data +
                    k * tf.square(self.mean_variable)) / \
                   (n + k) - tf.square(mean - self._mean_variable)
        variance = tf.maximum(variance, self._min_var)
        with tf.control_dependencies([n, k, mean, variance]):
            return (
                tf.assign_add(self._total_count_variable, k),
                tf.assign(self._variance_variable, variance) if self._train_var else tf.no_op(),
                tf.assign(self._mean_variable, mean) if self._train_mean else tf.no_op())