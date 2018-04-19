# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

import tensorflow as tf
from libspn.graph.scope import Scope
from libspn.graph.node import DistributionNode, ParameterizedDistributionNode
from libspn import conf
from libspn import utils
from libspn.exceptions import StructureError
import numpy as np
import tensorflow.contrib.distributions as tfd

# Some good sources:
# https://github.com/PhDP/spn/blob/1b837f1293e1098e6d7d908f4647a1d368308833/code/src/spn/SPN.java#L263
# https://github.com/whsu/spn/tree/master/spn


class NormalLeaf(ParameterizedDistributionNode):

    def __init__(self, feed=None, num_vars=1, num_components=2, name="GaussianQuantile", data=None,
                 learn_scale=True, trainable=True, total_counts_init=1):
        self._num_components = num_components
        self._means = tf.zeros((num_vars, num_components), dtype=conf.dtype)
        self._stddevs = tf.ones((num_vars, num_components), dtype=conf.dtype)
        if data is not None:
            self.learn_from_data(data, learn_scale=learn_scale)
        self._dist = None

        var_shape = (num_vars, num_components)
        mean = super().Parameter("mean", var_shape, self._means)
        stddev = super().Parameter("stddev", var_shape, self._stddevs)

        self._total_count_variable = self._total_accumulates(total_counts_init, var_shape)
        super().__init__(parameters=[mean, stddev], feed=feed, num_vars=num_vars, name=name,
                         trainable=trainable)

    def learn_from_data(self, data, learn_scale=True):
        """Learns the distribution parameters from data
        Params:
            data: numpy.ndarray of shape [batch, num_vars]
        """
        if len(data.shape) != 2 or data.shape[1] != self._num_vars:
            raise ValueError("Data should be of rank 2 and contain equally many variables as this "
                             "GaussianQuantile node.")

        values_per_quantile = self._values_per_quantile(data)

        self._means = np.stack(
            [np.mean(values, axis=0) for values in values_per_quantile], axis=-1)
        if learn_scale:
            self._stddevs = np.stack(
                [np.std(values, axis=0) for values in values_per_quantile], axis=-1)
        else:
            self._stddevs = np.ones_like(self._means)

    def serialize(self):
        data = super().serialize()
        data['num_vars'] = self._num_vars
        data['num_components'] = self._num_components
        data['means'] = self._means
        data['stddevs'] = self._stddevs
        return data

    def deserialize(self, data):
        self._num_vars = data['num_vars']
        self._num_components = data['num_components']
        self._means = data['means']
        self._stddevs = data['stddevs']
        super().deserialize(data)

    def _total_accumulates(self, init_val, shape):
        init = utils.broadcast_value(init_val, shape, dtype=conf.dtype)
        return tf.Variable(init, name=self.name + "TotalCounts", trainable=False),
            # tf.Variable(self.)

    def _assert_built(self):
        if not self._built:
            raise StructureError(self.name + " has not been built. Use learn_from_data(...) to "
                                             "learn distribution parameters.")

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
        self._assert_built()
        evidence_probs = tf.reshape(
            self._dist.prob(self._tile_feed()), (-1, self._compute_out_size()))
        return tf.where(self._tile_evidence(), evidence_probs, tf.ones_like(evidence_probs))

    def _compute_log_value(self):
        self._assert_built()
        evidence_probs = tf.reshape(
            self._dist.log_prob(self._tile_feed()), (-1, self._compute_out_size()))
        return tf.where(self._tile_evidence(), evidence_probs, tf.zeros_like(evidence_probs))

    def _compute_scope(self):
        return [Scope(self, i) for i in range(self._num_vars) for _ in range(self._num_components)]

    def _compute_mpe_state(self, counts):
        self._assert_built()
        counts_reshaped = tf.reshape(counts, (-1, self._num_vars, self._num_components))
        indices = tf.argmax(counts_reshaped, axis=-1) + tf.expand_dims(
            tf.range(self._num_vars, dtype=tf.int64) * self._num_components, axis=0)
        return tf.gather(tf.reshape(self._dist.loc, (-1,)), indices=indices, axis=0)

    def _compute_hard_em_update(self, counts):
        return tf.reduce_sum(tf.reshape(counts, (-1, self._num_vars, self._num_components)), axis=0)

    def assign(self, accum):
        update_total_counts = tf.assign_add(self._total_count_variable, accum)
        mean = (self._total_count_variable * self.variables['mean'] + accum)


class GaussianQuantile(DistributionNode):

    def __init__(self, feed=None, num_vars=1, num_components=2, name="GaussianQuantile"):
        self._num_components = num_components
        self._means = None
        self._stddevs = None
        self._dist = None
        self._built = False
        super().__init__(feed, num_vars=num_vars, name=name)

    def learn_from_data(self, data):
        """Learns the distribution parameters from data
        Params:
            data: numpy.ndarray of shape [batch, num_vars]
        """
        if len(data.shape) != 2 or data.shape[1] != self._num_vars:
            raise ValueError("Data should be of rank 2 and contain equally many variables as this "
                             "GaussianQuantile node.")

        values_per_quantile = self._values_per_quantile(data)

        self._means = [np.mean(values, axis=0) for values in values_per_quantile]
        self._stddevs = [np.std(values, axis=0) for values in values_per_quantile]
        self._dist = tfd.Normal(
            loc=np.stack(self._means, axis=-1),
            scale=np.stack(self._stddevs, axis=-1)
        )
        self._built = True

    def serialize(self):
        data = super().serialize()
        data['num_vars'] = self._num_vars
        data['num_components'] = self._num_components
        data['means'] = self._means
        data['stddevs'] = self._stddevs
        return data

    def deserialize(self, data):
        self._num_vars = data['num_vars']
        self._num_components = data['num_components']
        self._means = data['means']
        self._stddevs = data['stddevs']
        super().deserialize(data)

    def _assert_built(self):
        if not self._built:
            raise StructureError(self.name + " has not been built. Use learn_from_data(...) to "
                                             "learn distribution parameters.")

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
        self._assert_built()
        evidence_probs = tf.reshape(
            self._dist.prob(self._tile_feed()), (-1, self._compute_out_size()))
        return tf.where(self._tile_evidence(), evidence_probs, tf.ones_like(evidence_probs))

    def _compute_log_value(self):
        self._assert_built()
        evidence_probs = tf.reshape(
            self._dist.log_prob(self._tile_feed()), (-1, self._compute_out_size()))
        return tf.where(self._tile_evidence(), evidence_probs, tf.zeros_like(evidence_probs))

    def _compute_scope(self):
        return [Scope(self, i) for i in range(self._num_vars) for _ in range(self._num_components)]

    def _compute_mpe_state(self, counts):
        self._assert_built()
        counts_reshaped = tf.reshape(counts, (-1, self._num_vars, self._num_components))
        indices = tf.argmax(counts_reshaped, axis=-1) + tf.expand_dims(
            tf.range(self._num_vars, dtype=tf.int64) * self._num_components, axis=0)
        return tf.gather(tf.reshape(self._dist.loc, (-1,)), indices=indices, axis=0)
