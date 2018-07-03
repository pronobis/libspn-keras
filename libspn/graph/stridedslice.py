# ------------------------------------------------------------------------
# Copyright (C) 2016 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------
import itertools

from libspn.inference.type import InferenceType
import libspn.utils as utils
import tensorflow as tf
from libspn.exceptions import StructureError
import numpy as np
from libspn.graph.node import OpNode
from libspn.log import get_logger
from tensorflow.python.ops import array_ops


@utils.register_serializable
class StridedSlice2D(OpNode):
    """A container representing convolutional products in an SPN.

    Args:
        *values (input_like): Inputs providing input values to this container.
            See :meth:`~libspn.Input.as_input` for possible values.
        num_sums (int): Number of Sum ops modelled by this container.
        weights (input_like): Input providing weights container to this sum container.
            See :meth:`~libspn.Input.as_input` for possible values. If set
            to ``None``, the input is disconnected.
        ivs (input_like): Input providing IVs of an explicit latent variable
            associated with this sum container. See :meth:`~libspn.Input.as_input`
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

    logger = get_logger()

    def __init__(self, *values, name="StridedSlice2D", begin=(0, 0), end=None, strides=(2, 2),
                 grid_dim_sizes=None):
        self._begin = begin
        self._end = end
        self._strides = strides
        super().__init__(name=name)
        self.set_values(*values)

        if grid_dim_sizes is None:
            raise NotImplementedError(
                "{}: Must also provide grid_dim_sizes at this point.".format(self))
        self._grid_dim_sizes = grid_dim_sizes
        self._batch_axis = 0
        self._channel_axis = 3

    def set_values(self, *values):
        """Set the inputs providing input values to this node. If no arguments
        are given, all existing value inputs get disconnected.

        Args:
            *values (input_like): Inputs providing input values to this node.
                See :meth:`~libspn.Input.as_input` for possible values.
        """
        self._values = self._parse_inputs(*values)

    @utils.lru_cache
    def _spatial_concat(self, *input_tensors):
        """Concatenates input tensors spatially. Makes sure to reshape them before.

        Args:
            input_tensors (tuple): A tuple of `Tensor`s to concatenate along the channel axis.

        Returns:
            The concatenated tensor.
        """
        input_tensors = [self._spatial_reshape(t) for t in input_tensors]
        return utils.concat_maybe(input_tensors, axis=self._channel_axis)

    def _spatial_reshape(self, t, forward=True):
        """Reshapes a Tensor ``t``` to one that represents the spatial dimensions.

        Args:
            t (Tensor): The ``Tensor`` to reshape.
            forward (bool): Whether to reshape for forward inference. If True, reshapes to
                ``[batch, rows, cols, 1, input_channels]``. Otherwise, reshapes to
                ``[batch, rows, cols, output_channels, input_channels]``.
        Returns:
             A reshaped ``Tensor``.
        """
        non_batch_dim_size = self._non_batch_dim_prod(t)
        if forward:
            input_channels = non_batch_dim_size // np.prod(self._grid_dim_sizes)
            return tf.reshape(t, [-1] + self._grid_dim_sizes + [input_channels])
        return tf.reshape(t, [-1] + self._grid_dim_sizes + [self._num_input_channels()])

    def _non_batch_dim_prod(self, t):
        """Computes the product of the non-batch dimensions to be used for reshaping purposes.

        Args:
            t (Tensor): A ``Tensor`` for which to compute the product.

        Returns:
            An ``int``: product of non-batch dimensions.
        """
        non_batch_dim_size = np.prod([ds for i, ds in enumerate(t.shape.as_list())
                                      if i != self._batch_axis])
        return int(non_batch_dim_size)

    def _num_channels_per_input(self):
        """Returns a list of number of input channels for each value Input.

        Returns:
            A list of ints containing the number of channels.
        """
        input_sizes = self.get_input_sizes()
        return [int(s // np.prod(self._grid_dim_sizes)) for s in input_sizes]

    def _num_input_channels(self):
        """Computes the total number of input channels.

        Returns:
            An int indicating the number of input channels for the convolution operation.
        """
        return sum(self._num_channels_per_input())

    @utils.lru_cache
    def _batch_size(self, t):
        return tf.shape(t)[self._batch_axis]

    def _slice_end(self, batch_size):
        row, col = self._slice_end_row_col()
        return [batch_size, row, col, self._num_input_channels()]

    def _slice_end_row_col(self):
        if self._end is None:
            row, col = self._grid_dim_sizes
        else:
            row, col = self._end
        if row < 0:
            row = self._grid_dim_sizes[0] + row
        if col < 0:
            col = self._grid_dim_sizes[1] + col
        return row, col

    def _slice_start(self):
        return [0, self._begin[0], self._begin[1], 0]

    def _slice_stride(self):
        return [1, self._strides[0], self._strides[1],  1]
    
    def _compute_out_size_spatial(self, *input_out_sizes):
        """Computes spatial output shape.

        Returns:
            A tuple with (num_rows, num_cols, num_channels).
        """
        end_row, end_col = self._slice_end_row_col()
        begin_row, begin_col = self._begin
        strides_row, strides_col = self._strides
        out_rows = int(np.ceil((end_row - begin_row) / strides_row))
        out_cols = int(np.ceil((end_col - begin_col) / strides_col))

        return out_rows, out_cols, self._num_input_channels()

    def _compute_out_size(self, *input_out_sizes):
        return int(np.prod(self._compute_out_size_spatial(*input_out_sizes)))

    @property
    def output_shape_spatial(self):
        """tuple: The spatial shape of this node, formatted as (rows, columns, channels). """
        return self._compute_out_size_spatial()

    @utils.lru_cache
    def _compute_value(self, *input_tensors):
        inp_concat = self._spatial_concat(*input_tensors)
        batch_size_tensor = self._batch_size(inp_concat)
        val = tf.strided_slice(
            inp_concat, 
            begin=self._slice_start(), 
            end=self._slice_end(batch_size=batch_size_tensor),
            strides=self._slice_stride())
        val.set_shape((None,) + self.output_shape_spatial)
        return self._flatten(val)

    def _compute_log_value(self, *input_tensors):
        return self._compute_value(*input_tensors)

    def _compute_mpe_value(self, *input_tensors):
        return self._compute_value(*input_tensors)

    def _compute_log_mpe_value(self, *input_tensors):
        return self._compute_value(*input_tensors)

    def _compute_mpe_path_common(self, counts, *input_values):
        if not self._values:
            raise StructureError("{} is missing input values.".format(self))
        # Concatenate inputs along channel axis, should already be done during forward pass
        inp_concat = self._spatial_concat(*input_values)
        batch_size_tensor = self._batch_size(inp_concat)
        inp_counts = array_ops.strided_slice_grad(
            shape=tf.shape(inp_concat),
            begin=self._slice_start(),
            end=self._slice_end(batch_size=batch_size_tensor),
            strides=self._slice_stride(),
            dy=tf.reshape(counts, (-1,) + self.output_shape_spatial))
        inp_counts.set_shape(
            (None, self._grid_dim_sizes[0], self._grid_dim_sizes[1], self._num_input_channels()))
        return self._split_to_children(inp_counts)

    def _compute_log_mpe_path(self, counts, *input_values, add_random=False,
                              use_unweighted=False, with_ivs=False, sample=False, sample_prob=None):
        return self._compute_mpe_path_common(counts, *input_values)

    def _compute_mpe_path(self, counts, *input_values, add_random=False,
                          use_unweighted=False, with_ivs=False, sample=False, sample_prob=None):
        return self._compute_mpe_path_common(counts, *input_values)

    @utils.lru_cache
    def _split_to_children(self, x):
        if len(self.inputs) == 1:
            return [self._flatten(x)]
        x_split = tf.split(x, num_or_size_splits=self._num_channels_per_input(),
                           axis=self._channel_axis)
        return [self._flatten(t) for t in x_split]

    @utils.lru_cache
    def _flatten(self, t):
        """Flattens a Tensor ``t`` so that the resulting shape is [batch, non_batch]

        Args:
            t (Tensor): A ``Tensor```to flatten

        Returns:
            A flattened ``Tensor``.
        """
        if self._batch_axis != 0:
            raise NotImplementedError("{}: Cannot flatten if batch axis isn't equal to zero."
                                      .format(self))
        non_batch_dim_size = self._non_batch_dim_prod(t)
        return tf.reshape(t, (-1, non_batch_dim_size))

    def _compute_gradient(self, *input_tensors, with_ivs=True):
        raise NotImplementedError("{}: No gradient implementation available.".format(self))

    def _compute_log_gradient(
            self, *value_tensors, with_ivs=True, sum_weight_grads=False):
        raise NotImplementedError("{}: No log-gradient implementation available.".format(self))

    @utils.docinherit(OpNode)
    def _compute_scope(self, *value_scopes):
        flat_value_scopes = self._gather_input_scopes(*value_scopes)

        value_scopes_grid = [
            np.asarray(vs).reshape(self._grid_dim_sizes + [-1]) for vs in flat_value_scopes]
        value_scopes_concat = np.concatenate(value_scopes_grid, axis=2)

        row_b, col_b = self._begin
        row_e, col_e = self._end
        row_s, col_s = self._strides
        return value_scopes_concat[:, row_b:row_e:row_s, col_b:col_e:col_s, :].ravel().tolist()
        

    @utils.docinherit(OpNode)
    def _compute_valid(self, *value_scopes):
        return self._compute_scope(*value_scopes)

    def deserialize_inputs(self, data, nodes_by_name):
        pass

    @property
    def inputs(self):
        return self._values

    def serialize(self):
        pass

    def deserialize(self, data):
        pass

    @property
    def _const_out_size(self):
        return True