# ------------------------------------------------------------------------
# Copyright (C) 2016 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

from libspn.inference.type import InferenceType
from libspn.graph.op.base_sum import BaseSum
import libspn.utils as utils
import tensorflow as tf
import numpy as np
from libspn.graph.op.spatial_sums import SpatialSums


@utils.register_serializable
class ConvSums(SpatialSums):
    """A container representing convolutional sums (which share the same input) in an SPN.

    Args:
        *values (input_like): Inputs providing input values to this container.
            See :meth:`~libspn.Input.as_input` for possible values.
        num_channels (int): Number of channels modeled by this spatial sum.
        spatial_dim_sizes (tuple or list of ints): Sizes of spatial dimensions
        weights (input_like): Input providing weights node to this sum node.
            See :meth:`~libspn.Input.as_input` for possible values. If set
            to ``None``, the input is disconnected.
        latent_indicators (input_like): Input providing IndicatorLeafs of an explicit latent
            variable associated with this sum node. See :meth:`~libspn.Input.as_input`
            for possible values. If set to ``None``, the input is disconnected.
        name (str): Name of the container.

    Attributes:
        inference_type(InferenceType): Flag indicating the preferred inference
                                       type for this container that will be used
                                       during value calculation and learning.
                                       Can be changed at any time and will be
                                       used during the next inference/learning
                                       op generation.
    """

    def __init__(self, *values, num_channels=1, weights=None, latent_indicators=None,
                 inference_type=InferenceType.MARGINAL, name="ConvSums", spatial_dim_sizes=None):
        super().__init__(
            *values, weights=weights, latent_indicators=latent_indicators, num_channels=num_channels,
            inference_type=inference_type, name=name, spatial_dim_sizes=spatial_dim_sizes)

    def _num_inner_sums(self):
        return self._num_channels

    @utils.docinherit(SpatialSums)
    def _spatial_weight_shape(self):
        return [1] * 3 + [self._num_channels, self._max_sum_size]

    @utils.docinherit(BaseSum)
    def _get_sum_sizes(self, num_sums):
        num_values = sum(self._get_input_num_channels())  # Skip latent indicators, weights
        return num_sums * int(np.prod(self._spatial_dim_sizes)) * [num_values]

    def _accumulate_weight_counts(self, counts_spatial):
        return tf.reduce_sum(counts_spatial, axis=self._op_axis)