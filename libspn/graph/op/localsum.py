# ------------------------------------------------------------------------
# Copyright (C) 2016 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

from libspn.inference.type import InferenceType
from libspn.graph.op.basesum import BaseSum
import libspn.utils as utils
import tensorflow as tf
import numpy as np
from libspn.graph.op.spatialsum import SpatialSum


@utils.register_serializable
class LocalSum(SpatialSum):
    """A container representing convolutional sums (which share the same input) in an SPN.

    Args:
        *values (input_like): Inputs providing input values to this container.
            See :meth:`~libspn.Input.as_input` for possible values.
        num_sums (int): Number of Sum ops modelled by this container.
        weights (input_like): Input providing weights container to this sum container.
            See :meth:`~libspn.Input.as_input` for possible values. If set
            to ``None``, the input is disconnected.
        name (str): Name of the container.

    Attributes:
        inference_type(InferenceType): Flag indicating the preferred inference
                                       type for this container that will be used
                                       during value calculation and learning.
                                       Can be changed at any time and will be
                                       used during the next inference/learning
                                       op generation.
    """

    @utils.docinherit(SpatialSum)
    def _num_inner_sums(self):
        return int(np.prod(self._grid_dim_sizes) * self._num_channels)

    @utils.docinherit(SpatialSum)
    def _spatial_weight_shape(self):
        return [1] + self._grid_dim_sizes + [self._num_channels, self._max_sum_size]

    @utils.docinherit(BaseSum)
    def _get_sum_sizes(self, num_sums):
        num_values = sum(self._get_input_num_channels())  # Skip ivs, weights
        return num_sums * [num_values]

    @property
    def _tile_unweighted_size(self):
        return self._num_channels

    def _accumulate_weight_counts(self, counts_spatial):
        return tf.reshape(counts_spatial,
                          (-1, int(np.prod(self.output_shape_spatial)), self._max_sum_size))
