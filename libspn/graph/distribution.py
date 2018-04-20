# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------
from collections import OrderedDict

import tensorflow as tf
from libspn.graph.scope import Scope
from libspn.graph.node import VarNode
from libspn import conf
from libspn import utils
from libspn.exceptions import StructureError
import numpy as np
import tensorflow.contrib.distributions as tfd

# Some good sources:
# https://github.com/PhDP/spn/blob/1b837f1293e1098e6d7d908f4647a1d368308833/code/src/spn/SPN.java#L263
# https://github.com/whsu/spn/tree/master/spn


class NormalLeaf(VarNode):

    def __init__(self, feed=None, num_vars=1, num_components=2, name="GaussianQuantile", data=None,
                 learn_scale=True, total_counts_init=1):
        self._num_components = num_components
        self._mean_init = tf.zeros((num_vars, num_components), dtype=conf.dtype)
        self._variance_init = tf.ones((num_vars, num_components), dtype=conf.dtype)
        if data is not None:
            self.learn_from_data(data, learn_scale=learn_scale)

        var_shape = (num_vars, num_components)
        self._total_count_variable = self._total_accumulates(total_counts_init, var_shape)
        self._mean_variable = None
        self._variance_variable = None
        super().__init__(feed=feed, name=name)

    @property
    def initialize(self):
        return tf.group(*[var.initializer for var in self.variables])

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

    def _create_placeholder(self):
        return tf.placeholder(conf.dtype, [None, self._num_vars])

    def _create_evidence_indicator(self):
        return tf.placeholder_with_default(
            tf.cast(tf.ones_like(self._placeholder), tf.bool), shape=[None, self._num_vars])

    def _create(self):
        super()._create()
        self._mean_variable = tf.Variable(
            self._mean_init, dtype=conf.dtype, collections=['spn_distribution_parameters'])
        self._variance_variable = tf.Variable(
            self._variance_init, dtype=conf.dtype, collections=['spn_distribution_parameters'])
        self._evidence_indicator = self._create_evidence_indicator()

    def learn_from_data(self, data, learn_scale=True):
        """Learns the distribution parameters from data
        Params:
            data: numpy.ndarray of shape [batch, num_vars]
        """
        if len(data.shape) != 2 or data.shape[1] != self._num_vars:
            raise ValueError("Data should be of rank 2 and contain equally many variables as this "
                             "GaussianQuantile node.")

        values_per_quantile = self._values_per_quantile(data)

        self._mean_init = np.stack(
            [np.mean(values, axis=0) for values in values_per_quantile], axis=-1)
        if learn_scale:
            self._variance_init = np.stack(
                [np.var(values, axis=0) for values in values_per_quantile], axis=-1)
        else:
            self._variance_init = np.ones_like(self._mean_init)

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
        init = utils.broadcast_value(init_val, shape, dtype=conf.dtype)
        return tf.Variable(init, name=self.name + "TotalCounts", trainable=False),

    def _values_per_quantile(self, data):
        batch_size = data.shape[0]
        quantile_sections = np.arange(
            batch_size // self._num_components, batch_size, batch_size // self._num_components)
        sorted_features = np.sort(data, axis=0).astype(tf.DType(conf.dtype).as_numpy_dtype())
        values_per_quantile = np.split(
            sorted_features, indices_or_sections=quantile_sections, axis=0)
        return values_per_quantile

    def _compute_out_size(self):
        return self._num_vars * self._num_components

    def _tile_feed(self):
        return tf.tile(tf.expand_dims(self._feed, -1), [1, 1, self._num_components])

    def _tile_evidence(self):
        return tf.tile(self.evidence, [1, self._num_components])

    def _compute_value(self):
        # self._assert_built()
        dist = tfd.Normal(loc=self._mean_variable, scale=tf.sqrt(self._variance_variable))
        evidence_probs = tf.reshape(
            dist.prob(self._tile_feed()), (-1, self._compute_out_size()))
        return tf.where(self._tile_evidence(), evidence_probs, tf.ones_like(evidence_probs))

    def _compute_log_value(self):
        # self._assert_built()
        dist = tfd.Normal(loc=self._mean_variable, scale=tf.sqrt(self._variance_variable))
        evidence_probs = tf.reshape(
            dist.log_prob(self._tile_feed()), (-1, self._compute_out_size()))
        return tf.where(self._tile_evidence(), evidence_probs, tf.zeros_like(evidence_probs))

    def _compute_scope(self):
        return [Scope(self, i) for i in range(self._num_vars) for _ in range(self._num_components)]

    def _compute_mpe_state(self, counts):
        # self._assert_built()
        counts_reshaped = tf.reshape(counts, (-1, self._num_vars, self._num_components))
        indices = tf.argmax(counts_reshaped, axis=-1) + tf.expand_dims(
            tf.range(self._num_vars, dtype=tf.int64) * self._num_components, axis=0)
        return tf.gather(tf.reshape(self._mean_variable, (-1,)), indices=indices, axis=0)

    def _compute_hard_em_update(self, counts):
        counts_reshaped = tf.reshape(counts, (-1, self._num_vars, self._num_components))
        accum = tf.reduce_sum(counts_reshaped, axis=0)
        tiled_feed = self._tile_feed()
        sum_data = tf.reduce_sum(counts_reshaped * tiled_feed, axis=0)
        sum_data_squared = tf.reduce_sum(counts_reshaped * tf.square(tiled_feed), axis=0)
        return {'accum': accum, "sum_data": sum_data, "sum_data_squared": sum_data_squared}

    def assign(self, accum, sum_data, sum_data_squared):
        total_counts = self._total_count_variable + accum
        mean = (self._total_count_variable * self._mean_variable + sum_data) / total_counts
        # dx = x - mean
        # dx.dot(dx) == \sum x^2 - 2 * mean * \sum x + n * mean^2
        variance = (self._total_count_variable * self._variance_variable +
                    sum_data_squared - 2 * mean * sum_data + accum * mean) / total_counts - \
            tf.square((mean - self._mean_variable))

        return (
            tf.assign(self._total_count_variable, total_counts),
            tf.assign(self._variance_variable, variance),
            tf.assign(self._mean_variable, mean)
        )

#
# class GaussianQuantile(DistributionNode):
#
#     def __init__(self, feed=None, num_vars=1, num_components=2, name="GaussianQuantile"):
#         self._num_components = num_components
#         self._means = None
#         self._stddevs = None
#         self._dist = None
#         self._built = False
#         super().__init__(feed, num_vars=num_vars, name=name)
#
#     def learn_from_data(self, data):
#         """Learns the distribution parameters from data
#         Params:
#             data: numpy.ndarray of shape [batch, num_vars]
#         """
#         if len(data.shape) != 2 or data.shape[1] != self._num_vars:
#             raise ValueError("Data should be of rank 2 and contain equally many variables as this "
#                              "GaussianQuantile node.")
#
#         values_per_quantile = self._values_per_quantile(data)
#
#         self._means = [np.mean(values, axis=0) for values in values_per_quantile]
#         self._stddevs = [np.std(values, axis=0) for values in values_per_quantile]
#         self._dist = tfd.Normal(
#             loc=np.stack(self._means, axis=-1),
#             scale=np.stack(self._stddevs, axis=-1)
#         )
#         self._built = True
#
#     def serialize(self):
#         data = super().serialize()
#         data['num_vars'] = self._num_vars
#         data['num_components'] = self._num_components
#         data['means'] = self._means
#         data['stddevs'] = self._stddevs
#         return data
#
#     def deserialize(self, data):
#         self._num_vars = data['num_vars']
#         self._num_components = data['num_components']
#         self._means = data['means']
#         self._stddevs = data['stddevs']
#         super().deserialize(data)
#
#     def _assert_built(self):
#         if not self._built:
#             raise StructureError(self.name + " has not been built. Use learn_from_data(...) to "
#                                              "learn distribution parameters.")
#
#     def _values_per_quantile(self, data):
#         batch_size = data.shape[0]
#         quantile_sections = np.arange(
#             batch_size // self._num_components, batch_size, batch_size // self._num_components)
#         sorted_features = np.sort(data, axis=0).astype(tf.DType(conf.dtype).as_numpy_dtype())
#         values_per_quantile = np.split(
#             sorted_features, indices_or_sections=quantile_sections, axis=0)
#         return values_per_quantile
#
#     def _compute_out_size(self):
#         return self._num_vars * self._num_components
#
#     def _tile_feed(self):
#         return tf.tile(tf.expand_dims(self._feed, -1), [1, 1, self._num_components])
#
#     def _tile_evidence(self):
#         return tf.tile(self.evidence, [1, self._num_components])
#
#     def _compute_value(self):
#         self._assert_built()
#         evidence_probs = tf.reshape(
#             self._dist.prob(self._tile_feed()), (-1, self._compute_out_size()))
#         return tf.where(self._tile_evidence(), evidence_probs, tf.ones_like(evidence_probs))
#
#     def _compute_log_value(self):
#         self._assert_built()
#         evidence_probs = tf.reshape(
#             self._dist.log_prob(self._tile_feed()), (-1, self._compute_out_size()))
#         return tf.where(self._tile_evidence(), evidence_probs, tf.zeros_like(evidence_probs))
#
#     def _compute_scope(self):
#         return [Scope(self, i) for i in range(self._num_vars) for _ in range(self._num_components)]
#
#     def _compute_mpe_state(self, counts):
#         self._assert_built()
#         counts_reshaped = tf.reshape(counts, (-1, self._num_vars, self._num_components))
#         indices = tf.argmax(counts_reshaped, axis=-1) + tf.expand_dims(
#             tf.range(self._num_vars, dtype=tf.int64) * self._num_components, axis=0)
#         return tf.gather(tf.reshape(self._dist.loc, (-1,)), indices=indices, axis=0)
