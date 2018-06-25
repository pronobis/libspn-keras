# ------------------------------------------------------------------------
# Copyright (C) 2016 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

from libspn.inference.type import InferenceType
import libspn.utils as utils
import tensorflow as tf
from libspn.exceptions import StructureError
import numpy as np
from libspn.graph.scope import Scope
from libspn.graph.node import OpNode
from libspn import conf
from libspn.log import get_logger


@utils.register_serializable
class ConvProd(OpNode):
    """A container representing convolutional sums (which share the same input) in an SPN.

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

    def __init__(self, *values, num_channels, padding='same', dilation_rate=1, strides=1,
                 kernel_size=(2, 2),
                 inference_type=InferenceType.MARGINAL, name="ConvProds",
                 sparse_connections=None, dense_connections=None, grid_dim_sizes=None,
                 num_spatial_dims=2):
        self._batch_axis = 0
        self._channel_axis = 3
        super().__init__(inference_type=inference_type, name=name)
        self.set_values(*values)
        if num_spatial_dims != 2:
            raise NotImplementedError("{}: Currently only supports 2D convolutions.")

        if not grid_dim_sizes:
            raise NotImplementedError(
                "{}: Must also provide grid_dim_sizes at this point.".format(self))

        self._grid_dim_sizes = grid_dim_sizes or [-1] * num_spatial_dims
        self._padding = padding
        self._dilation_rate = [dilation_rate] * num_spatial_dims \
            if isinstance(dilation_rate, int) else dilation_rate
        self._strides = [strides] * num_spatial_dims \
            if isinstance(strides, int) else strides
        self._num_channels = num_channels
        self._kernel_size = [kernel_size] * num_spatial_dims if isinstance(kernel_size, int) \
            else list(kernel_size)

        if sparse_connections is not None:
            if dense_connections is not None:
                raise ValueError("{}: Must provide either spare connections or dense connections, "
                                 "not both.".format(self))
            self._sparse_connections = sparse_connections
            self._dense_connections = self.sparse_connections_to_dense(sparse_connections)
        elif dense_connections is not None:
            self._dense_connections = dense_connections
            self._sparse_connections = self.dense_connections_to_sparse(dense_connections)
        else:
            self._sparse_connections = self.generate_sparse_connections(num_channels)
            self._dense_connections = self.sparse_connections_to_dense(self._sparse_connections)

    def set_values(self, *values):
        """Set the inputs providing input values to this node. If no arguments
        are given, all existing value inputs get disconnected.

        Args:
            *values (input_like): Inputs providing input values to this node.
                See :meth:`~libspn.Input.as_input` for possible values.
        """
        self._values = self._parse_inputs(*values)

    def sparse_connections_to_dense(self, sparse):
        # Sparse has shape [rows, cols, out_channels]
        # Dense will have shape [rows, cols, in_channels, out_channels]
        num_input_channels = self._input_channels()
        batch_grid_prod = int(np.prod(self._kernel_size + [self._num_channels]))
        dense = np.zeros([batch_grid_prod, num_input_channels])

        dense[np.arange(batch_grid_prod), sparse.ravel()] = 1.0
        dense = dense.reshape(self._kernel_size + [self._num_channels, num_input_channels])
        # [rows, cols, out_channels, in_channels] to [rows, cols, in_channels, out_channels]
        return np.transpose(dense, (0, 1, 3, 2)).astype(conf.dtype.as_numpy_dtype())

    def dense_connections_to_sparse(self, dense):
        return np.argmax(dense, axis=2)

    def generate_sparse_connections(self, num_channels):
        num_input_channels = self._input_channels()
        kernel_surface = int(np.prod(self._kernel_size))
        total_possibilities = num_input_channels ** kernel_surface
        if num_channels >= total_possibilities:
            if num_channels > total_possibilities:
                self.logger.warn("Number of channels exceeds total number of combinations.")
                self._num_channels = total_possibilities
            p = np.arange(total_possibilities)
            kernel_cells = []
            for _ in range(kernel_surface):
                kernel_cells.append(p % num_input_channels)
                p //= num_input_channels
            return np.stack(kernel_cells, axis=0).reshape(self._kernel_size + [total_possibilities])

        size = int(np.prod(self._grid_dim_sizes + [num_channels]))
        return np.random.randint(num_input_channels, size=size).reshape(
            self._grid_dim_sizes + [num_channels])

    @utils.lru_cache
    def _concatenate_input_tensors(self, *input_tensors):
        input_tensors = [self._spatial_reshape(t) for t in input_tensors]
        reducible_inputs = utils.concat_maybe(input_tensors, axis=self._channel_axis)
        return reducible_inputs

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
        return tf.reshape(t, [-1] + self._grid_dim_sizes + [self._num_channels])

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

    def _get_input_num_channels(self):
        """Returns a list of number of input channels for each value Input.

        Returns:
            A list of ints containing the number of channels.
        """
        input_sizes = self.get_input_sizes()
        return [int(s // np.prod(self._grid_dim_sizes)) for s in input_sizes]

    def _input_channels(self):
        return sum(self._get_input_num_channels())

    def _compute_out_size(self, *input_out_sizes):
        cols_pad_left, cols_pad_right, rows_pad_left, rows_pad_right = self.pad_sizes()
        total_rows = self._grid_dim_sizes[0] + rows_pad_left + rows_pad_right
        total_cols = self._grid_dim_sizes[1] + cols_pad_left + cols_pad_right
        out_rows = total_rows // self._strides[0]
        out_cols = total_cols // self._strides[1]
        return int(out_rows * out_cols * self._num_channels)

    def _compute_value(self, *input_tensors):
        raise NotImplementedError("{}: No linear value implementation for ConvProd".format(self))

    @utils.lru_cache
    def _compute_log_value(self, *input_tensors):
        concat_inp = self._concatenate_input_tensors(*input_tensors)
        conv_out = tf.nn.convolution(
            concat_inp, filter=self._dense_connections, padding=self._padding.upper(),
            strides=self._strides, dilation_rate=self._dilation_rate)
        return tf.reshape(conv_out, (-1, self._compute_out_size()))

    def _compute_mpe_value(self, *input_tensors):
        raise NotImplementedError("{}: No linear MPE value implementation for ConvProd"
                                  .format(self))

    def _compute_log_mpe_value(self, *input_tensors):
        return self._compute_log_value(*input_tensors)

    def _compute_mpe_path_common(self, counts, *input_values):
        if not self._values:
            raise StructureError("{} is missing input values.".format(self))
        inp_concat = self._concatenate_input_tensors(*input_values)
        input_counts = tf.nn.conv2d_backprop_input(
            input_sizes=tf.shape(inp_concat),
            filter=self._dense_connections,
            out_backprop=counts,
            strides=[1] + self._strides + [1],
            padding=self._padding.upper(),
            use_cudnn_on_gpu=True,
            data_format="NHWC",
            dilations=[1] + self._dilation_rate + [1])
        return self._split_to_children(input_counts)

    def _compute_log_mpe_path(self, counts, *input_values, add_random=False,
                              use_unweighted=False, with_ivs=False, sample=False, sample_prob=None):
        return self._compute_mpe_path_common(counts, *input_values)

    def _compute_mpe_path(self, counts, *input_values, add_random=False,
                          use_unweighted=False, with_ivs=False, sample=False, sample_prob=None):
        return self._compute_mpe_path_common(counts, *input_values)

    @utils.lru_cache
    def _split_to_children(self, x):
        x_split = tf.split(x, num_or_size_splits=self._get_input_num_channels(),
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

        dilate = self._dilation_rate
        kernel_size = self._kernel_size
        grid_dims = self._grid_dim_sizes
        strides = self._strides
        input_channels = self._input_channels()

        if self._padding.lower() == "same":
            cols_pad_left, cols_pad_right, rows_pad_left, rows_pad_right = self.pad_sizes()

            padded_value_scopes_concat = np.empty(
                (rows_pad_left + grid_dims[0] + rows_pad_right,
                 cols_pad_left + grid_dims[1] + cols_pad_right,
                 input_channels))
            padded_value_scopes_concat[:, :cols_pad_left] = Scope.__empty()
            padded_value_scopes_concat[:rows_pad_left, :] = Scope.__empty()
            padded_value_scopes_concat[-rows_pad_right:, :] = Scope.__empty()
            padded_value_scopes_concat[:, -cols_pad_right:] = Scope.__empty()
            padded_value_scopes_concat \
                [rows_pad_left:-rows_pad_right, cols_pad_left:-cols_pad_right] = value_scopes_concat
            value_scopes_concat = padded_value_scopes_concat

        scope_list = []
        for row in range(0, grid_dims[0] - kernel_size[0] * dilate[0] + 1, strides[0]):
            row_indices = list(range(row, row + kernel_size[0] * dilate[0], dilate[0]))
            for col in range(0, grid_dims[1] - kernel_size[1] * dilate[1] + 1, strides[1]):
                col_indices = list(range(col, col + kernel_size[1] * dilate[1], dilate[1]))
                for channel in range(self._num_channels):
                    single_scope = []
                    for im_row, kernel_row in zip(row_indices, range(kernel_size[0])):
                        for im_col, kernel_col in zip(col_indices, range(kernel_size[1])):
                            single_scope.append(
                                value_scopes_concat[
                                    im_row, im_col,
                                    self._sparse_connections[kernel_row, kernel_col, channel]])
                    scope_list.append(Scope.merge_scopes(single_scope))

        return scope_list

    def pad_sizes(self):
        if self._padding.lower() == "valid":
            return 0, 0, 0, 0

        dilate = self._dilation_rate
        kernel_size = self._kernel_size
        rows_pad = kernel_size[0] * dilate[0] - 1
        cols_pad = kernel_size[1] * dilate[1] - 1
        if rows_pad % 2 != 0:
            rows_pad_left = rows_pad // 2
            rows_pad_right = rows_pad // 2 + 1
        else:
            rows_pad_left = rows_pad_right = rows_pad // 2
        if cols_pad % 2 != 0:
            cols_pad_left = cols_pad // 2
            cols_pad_right = cols_pad // 2 + 1
        else:
            cols_pad_left = cols_pad_right = cols_pad // 2
        return cols_pad_left, cols_pad_right, rows_pad_left, rows_pad_right

    @utils.docinherit(OpNode)
    def _compute_valid(self, weight_scopes, ivs_scopes, *value_scopes):
        self.logger.warn("{}: validity is skipped for convolutional sum layer.".format(self))
        return self._compute_scope(weight_scopes, ivs_scopes, *value_scopes)

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