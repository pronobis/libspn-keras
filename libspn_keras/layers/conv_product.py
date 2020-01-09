import tensorflow as tf
from tensorflow import keras
from tensorflow import initializers
import operator
import functools
import numpy as np


class ConvProduct(keras.layers.Layer):

    def __init__(self, strides, dilations, kernel_size, num_channels=None, padding='valid', **kwargs):
        super(ConvProduct, self).__init__(**kwargs)
        self.strides = strides
        self.dilations = dilations
        self.num_channels = num_channels
        self.padding = padding
        self.kernel_size = kernel_size
        self._spatial_dim_sizes = None

    def build(self, input_shape):
        num_batch, num_scopes_vertical, num_scopes_horizontal, num_channels_in = input_shape

        self._spatial_dim_sizes = num_scopes_vertical, num_scopes_horizontal

        sparse_kernels = self._create_sparse_kernels(num_channels_in, self.num_channels)

        onehot_kernels = self.sparse_kernels_to_onehot(sparse_kernels, num_channels_in)

        self._onehot_kernels = self.add_weight(
            "onehot_kernel", initializer=initializers.Constant(onehot_kernels), trainable=False,
            shape=onehot_kernels.shape
        )

        super(ConvProduct, self).build(input_shape)

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
        onehot_kernels = np.ones(
            (kernels_height, kernels_width, num_channels_out, num_channels_in))
        onehot_kernels *= np.arange(num_channels_in).reshape([1, 1, 1, num_channels_in])
        onehot_kernels = np.equal(
            onehot_kernels, np.expand_dims(sparse_kernels, -1)).astype(np.float32)
        return np.transpose(onehot_kernels, (0, 1, 3, 2))

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
        kernel_surface = int(np.prod(self.kernel_size))
        total_possibilities = num_channels_in ** kernel_surface
        if num_channels_out >= total_possibilities:
            if num_channels_out > total_possibilities:
                self.logger.warn("Number of channels exceeds total number of combinations.")
                self.num_channels = total_possibilities
            p = np.arange(total_possibilities)
            kernel_cells = []
            for _ in range(kernel_surface):
                kernel_cells.append(p % num_channels_in)
                p //= num_channels_in
            return np.stack(kernel_cells, axis=0).reshape(self.kernel_size + [total_possibilities])

        if num_channels_out >= num_channels_in:
            kernel_cells = []
            for _ in range(kernel_surface):
                ind = np.arange(self.num_channels) % num_channels_in
                np.random.shuffle(ind)
                kernel_cells.append(ind)
            return np.asarray(kernel_cells).reshape(self.kernel_size + [self.num_channels])

        sparse_shape = self.kernel_size + [num_channels_out]
        size = int(np.prod(sparse_shape))
        return np.random.randint(num_channels_in, size=size).reshape(sparse_shape)

    def _effective_kernel_size(self):
        """Computes the 'effective' kernel size by also taking into account the dilation rate.

        Returns:
            tuple: A tuple with (num_kernel_rows, num_kernel_cols)
        """
        kernel_sizes = [(ks - 1) * d + 1 for ks, d in zip(self.kernel_size, self.dilations)]
        return kernel_sizes

    def pad_sizes(self):
        """Determines the pad sizes. Possibly adds up explicit padding and padding through SAME
        padding algorithm of `tf.nn.convolution`.

        Returns:
            A tuple of left, right, top and bottom padding sizes.
        """
        if self.padding == 'valid':
            return 0, 0, 0, 0
        if self.padding == 'full':
            kernel_height, kernel_width = self._effective_kernel_size()
            pad_top = pad_bottom = kernel_height - 1
            pad_left = pad_right = kernel_width - 1
            return pad_left, pad_right, pad_top, pad_bottom
        if self.padding == 'final':
            kernel_height, kernel_width = self._effective_kernel_size()
            pad_top = (kernel_height - 1) * 2 - self._spatial_dim_sizes[0]
            pad_left = (kernel_width - 1) * 2 - self._spatial_dim_sizes[1]
            return pad_left, 0, pad_top, 0
        raise ValueError(
            "{}: invalid padding algorithm. Use 'valid', 'full' or 'wicker_top', got '{}'"
            .format(self, self.padding))

    def _compute_out_size_spatial(self, num_scopes_vertical_in, num_scopes_horizontal_in):
        """Computes spatial output shape.

        Returns:
            A tuple with (num_rows, num_cols, num_channels).
        """
        kernel_size0, kernel_size1 = self._effective_kernel_size()

        pad_left, pad_right, pad_top, pad_bottom = self.pad_sizes()

        rows_post_pad = pad_top + pad_bottom + num_scopes_vertical_in - kernel_size0 + 1
        cols_post_pad = pad_left + pad_right + num_scopes_horizontal_in - kernel_size1 + 1
        return tuple(
            int(np.ceil(post_pad / s))
            for post_pad, s in zip([rows_post_pad, cols_post_pad], self.strides)
        )

    def call(self, x):
        # Split in list of tensors which will be added up using outer products
        pad_left, pad_right, pad_top, pad_bottom = self.pad_sizes()
        x_padded = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
        out = tf.nn.conv2d(
            x_padded, self._onehot_kernels, strides=self.strides, padding='VALID',
            dilations=self.dilations
        )
        return out

    def compute_output_shape(self, input_shape):
        num_batch, num_scopes_vertical_in, num_scopes_horizontal_in, _ = input_shape
        num_scopes_vertical_out, num_scopes_horizontal_out = self._compute_out_size_spatial(
            num_scopes_vertical_in, num_scopes_horizontal_in
        )
        return num_batch, num_scopes_vertical_out, num_scopes_horizontal_out, self.num_channels

