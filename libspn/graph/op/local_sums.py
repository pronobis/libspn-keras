from libspn.inference.type import InferenceType
from libspn.graph.op.base_sum import BaseSum
import libspn.utils as utils
import tensorflow as tf
import numpy as np
from libspn.graph.op.spatial_sums import SpatialSums


@utils.register_serializable
class LocalSums(SpatialSums):
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
                 inference_type=InferenceType.MARGINAL, name="LocalSums", spatial_dim_sizes=None):
        super().__init__(
            *values, weights=weights, latent_indicators=latent_indicators, num_channels=num_channels,
            inference_type=inference_type, name=name, spatial_dim_sizes=spatial_dim_sizes)

    @utils.docinherit(SpatialSums)
    def _num_inner_sums(self):
        return int(np.prod(self._spatial_dim_sizes) * self._num_channels)

    @utils.docinherit(SpatialSums)
    def _spatial_weight_shape(self):
        return [1] + self._spatial_dim_sizes + [self._num_channels, self._max_sum_size]

    @utils.docinherit(BaseSum)
    def _get_sum_sizes(self, num_sums):
        num_values = sum(self._get_input_num_channels())  # Skip latent indicators, weights
        return num_sums * [num_values]

    @property
    def _tile_unweighted_size(self):
        return self._num_channels

    def _accumulate_weight_counts(self, counts_spatial):
        return tf.reshape(counts_spatial,
                          (-1, int(np.prod(self.output_shape_spatial)), self._max_sum_size))
