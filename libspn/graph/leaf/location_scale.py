import abc

import numpy as np
import tensorflow as tf

from libspn import utils, conf
from libspn.graph.node import Node
from libspn.graph.leaf.continuous_base import ContinuousLeafBase, _softplus_inverse_np
from libspn.utils.initializers import Equidistant
from libspn.utils.graphkeys import SPNGraphKeys


class LocationScaleLeaf(ContinuousLeafBase, abc.ABC):

    def __init__(self, feed=None, evidence_indicator_feed=None, num_vars=1, num_components=2,
                 total_counts_init=1.0, trainable_loc=True, trainable_scale=True,
                 loc_init=Equidistant(),
                 scale_init=1.0, min_scale=1e-2, softplus_scale=True,
                 dimensionality=1, name="LocationScaleLeaf", component_axis=-1,
                 share_locs_across_vars=False, share_scales=False, samplewise_normalization=False):
        self._softplus_scale = softplus_scale
        # Initial value for means
        variable_shape = self._variable_shape(num_vars, num_components, dimensionality)
        self._min_scale = min_scale if not softplus_scale else np.log(np.exp(min_scale) - 1)
        self._sample_wise_normalization = samplewise_normalization
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

    @property
    def trainable_loc(self):
        return self._trainable_loc

    @property
    def trainable_scale(self):
        return self._trainable_scale
