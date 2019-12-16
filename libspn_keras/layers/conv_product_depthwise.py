import tensorflow as tf
from tensorflow import initializers
from libspn_keras.layers.conv_product import ConvProduct


class ConvProductDepthwise(ConvProduct):
    """
    A 2D 'convolutional product', which internally is implemented as a convolution with unit
    kernels. Since we're operating in log-space, we can consider locally applied sum operations in
    these convolutions as multiplications in linear space.
    """

    def __init__(self, strides, dilations, kernel_size, padding='valid'):
        super().__init__(strides, dilations, kernel_size, padding=padding)

    def build(self, input_shape):
        num_batch, num_scopes_vertical, num_scopes_horizontal, num_channels_in = input_shape

        self._spatial_dim_sizes = num_scopes_vertical, num_scopes_horizontal
        self.num_channels = num_channels_in

        sparse_kernels = self._create_sparse_kernels(1, 1)

        onehot_kernels = self.sparse_kernels_to_onehot(sparse_kernels, 1)

        self._onehot_kernels = self.add_weight(
            "onehot_kernel", initializer=initializers.Constant(onehot_kernels), trainable=False,
            shape=onehot_kernels.shape
        )

    def call(self, x):
        # Split in list of tensors which will be added up using outer products
        pad_left, pad_right, pad_top, pad_bottom = self.pad_sizes()
        channels_first = tf.reshape(
            tf.transpose(x, (0, 3, 1, 2)),
            (-1,) + self._spatial_dim_sizes + (1,)
        )
        x_padded = tf.pad(
            channels_first, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
        out = tf.nn.conv2d(
            x_padded, self._onehot_kernels, strides=self.strides, padding='VALID',
            dilations=self.dilations
        )

        spatial_dim_sizes_out = self._compute_out_size_spatial(*self._spatial_dim_sizes)

        return tf.transpose(
            tf.reshape(out, (-1, self.num_channels) + spatial_dim_sizes_out),
            (0, 2, 3, 1)
        )

    def compute_output_shape(self, input_shape):
        num_batch, num_scopes_vertical_in, num_scopes_horizontal_in, _ = input_shape
        num_scopes_vertical_out, num_scopes_horizontal_out = self._compute_out_size_spatial(
            num_scopes_vertical_in, num_scopes_horizontal_in
        )
        return num_batch, num_scopes_vertical_out, num_scopes_horizontal_out, self.num_channels

