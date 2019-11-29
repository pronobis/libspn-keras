import tensorflow as tf
from tensorflow import keras
from tensorflow import initializers
import operator
import functools
import numpy as np


class PatchWiseProduct(keras.layers.Layer):

    def __init__(
        self, strides, dilations, kernel_size, num_channels=None, depthwise=False, padding='valid'
    ):
        super(PatchWiseProduct, self).__init__()
        self._strides = strides
        self._dilations = dilations
        self._num_channels = num_channels
        self._padding = padding
        self._depthwise = depthwise
        self._kernel_size = kernel_size

    @property
    def num_factors(self):
        return self._num_factors

    def build(self, input_shape):
        num_batch, num_scopes_vertical, num_scopes_horizontal, num_channels_in = input_shape

        sparse_kernels = self._create_sparse_kernels(num_channels_in, self._num_channels)

        onehot_kernels = self.sparse_kernels_to_onehot(sparse_kernels, num_channels_in)

        self._onehot_kernels = self.add_weight(
            "onehot_kernel", initializer=initializers.Constant(onehot_kernels), trainable=False)

        super(PatchWiseProduct, self).build(input_shape)

    def sparse_kernels_to_onehot(self, sparse_kernels, num_channels_in):
        """Converts an index-based representation of sparse kernels to a dense onehot
        representation.

        Args:
            sparse (numpy.ndarray): A sparse kernel representation of shape
                [rows, cols, output_channel] containing the indices for which the kernel equals 1.

        Returns:
            A onehot representation of the same kernel with shape
            [rows, cols, input_channel, output_channel].
        """
        kernels_height, kernels_width, num_channels_out = sparse_kernels.shape
        onehot_kernels = np.zeros(
            (kernels_height, kernels_width, num_channels_out, num_channels_in))
        onehot_kernels[sparse_kernels] = 1
        return np.transpose(onehot_kernels, (0, 1, 3, 20))

    def _create_sparse_kernels(self, num_channels_in, num_channels_out):
        """Generates sparse kernels kernels. These kernels only contain '1' on a single channel
        per row and column. The remaining values are all zero. This method returns the sparse
        representation, containing only the indices for which the kernels are 1 along the input
        channel axis.

        Args:
            num_channels_out (int): The number of channels. In case the number of channels given is
                larger than the number of possible one-hot assignments, a warning is given and the
                number of channels is set accordingly before generating the connections.
        Returns:
            A `numpy.ndarray` containing the 'sparse' representation of the kernels with shape
            `[row, column, channel]`, containing the indices of the input channel for which the
            kernel is 1.
        """
        kernel_surface = int(np.prod(self._kernel_size))
        total_possibilities = num_channels_in ** kernel_surface
        if num_channels_out >= total_possibilities:
            if num_channels_out > total_possibilities:
                self.logger.warn("Number of channels exceeds total number of combinations.")
                self._num_channels = total_possibilities
            p = np.arange(total_possibilities)
            kernel_cells = []
            for _ in range(kernel_surface):
                kernel_cells.append(p % num_channels_in)
                p //= num_channels_in
            return np.stack(kernel_cells, axis=0).reshape(self._kernel_size + [total_possibilities])

        if num_channels_out >= num_channels_in:
            kernel_cells = []
            for _ in range(kernel_surface):
                ind = np.arange(self._num_channels) % num_channels_in
                np.random.shuffle(ind)
                kernel_cells.append(ind)
            return np.asarray(kernel_cells).reshape(self._kernel_size + [self._num_channels])

        sparse_shape = self._kernel_size + [num_channels_out]
        size = int(np.prod(sparse_shape))
        return np.random.randint(num_channels_in, size=size).reshape(sparse_shape)

    def _effective_kernel_size(self):
        """Computes the 'effective' kernel size by also taking into account the dilation rate.

        Returns:
            tuple: A tuple with (num_kernel_rows, num_kernel_cols)
        """
        return [
            (self._kernel_size[0] - 1) * self._dilations[0] + 1,
            (self._kernel_size[1] - 1) * self._dilations[1] + 1
        ]

    def pad_sizes(self):
        """Determines the pad sizes. Possibly adds up explicit padding and padding through SAME
        padding algorithm of `tf.nn.convolution`.

        Returns:
            A tuple of left, right, top and bottom padding sizes.
        """
        if self._padding == 'valid':
            return 0, 0, 0, 0
        if self._padding == 'full':
            kernel_height, kernel_width = self._effective_kernel_size()
            pad_top = pad_bottom = kernel_height - 1
            pad_left = pad_right = kernel_width - 1
            return pad_left, pad_right, pad_top, pad_bottom
        if self._padding == 'wicker_top':
            kernel_height, kernel_width = self._effective_kernel_size()
            pad_top = (kernel_height - 1) * 2 - self._spatial_dim_sizes[0]
            pad_left = (kernel_width - 1) * 2 - self._spatial_dim_sizes[1]
            return 0, pad_left, 0, pad_top
        raise ValueError(
            "{}: invalid padding algorithm. Use 'valid', 'full' or 'wicker_top', got '{}'"
            .format(self, self._padding))

    def call(self, x):
        # Split in list of tensors which will be added up using outer products

        conv_out = tf.nn.conv2d(
            x, self._onehot_kernels, strides=self._strides, padding=self._padding,

        )

        return tf.reshape(
            functools.reduce(operator.add, log_prob_per_in_scope),
            [self._num_scopes, self._num_decomps, -1, self._num_products]
        )

    def compute_output_shape(self, input_shape):
        num_scopes_in, num_decomps, _, num_nodes_in = input_shape
        return (
            num_scopes_in // self._num_factors,
            self._num_decomps,
            None,
            num_nodes_in ** self._num_factors
        )
