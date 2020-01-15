from libspn.inference.type import InferenceType
import libspn.utils as utils
import tensorflow as tf
from libspn.exceptions import StructureError
from libspn.log import get_logger
from libspn.graph.op.conv_products import ConvProducts


@utils.register_serializable
class ConvProductsDepthwise(ConvProducts):
    """A container representing convolutional products in an SPN.

    Args:
        *values (input_like): Inputs providing input values to this container.
            See :meth:`~libspn.Input.as_input` for possible values.
        num_channels (int): Number of channels modeled by this node. This parameter is optional.
            If ``None``, the layer will attempt to generate all possible permutations of channels
            under a patch as long as it is under ``num_channels_max``.
        padding (str): Type of padding used. Can be either, 'full', 'valid' or 'wicker_top'.
            For building Wicker CSPNs, 'full' padding is necessary in all but the very last
            ConvProducts node. The last ConvProducts node should take the 'wicker_top' padding algorithm
        dilation_rate (int or tuple of ints): Dilation rate of the convolution.
        strides (int or tuple of ints): Strides used for the convolution.
        spatial_dim_sizes (list or tuple of ints): Dim sizes of spatial dimensions (height and width)
        num_channels_max (int): The maximum number of channels when automatically generating
            permutations.
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

    def __init__(self, *values, padding='valid', dilation_rate=1,
                 strides=2, kernel_size=2, inference_type=InferenceType.MARGINAL,
                 name="ConvProductsDepthwise", spatial_dim_sizes=None):
        super().__init__(
            *values, inference_type=inference_type, name=name, spatial_dim_sizes=spatial_dim_sizes,
            strides=strides, kernel_size=kernel_size, padding=padding, dilation_rate=dilation_rate)
        self._num_channels = self._num_input_channels()

    @utils.lru_cache
    def _compute_log_value(self, *input_tensors):
        # Concatenate along channel axis
        concat_inp = self._prepare_convolutional_processing(*input_tensors)

        # This the quickest workaround for TensorFlow's apparent optimization whenever
        # part of the kernel computation involves a -inf:
        concat_inp = tf.where(
            tf.is_inf(concat_inp), tf.fill(tf.shape(concat_inp), value=-1e20), concat_inp)
        # Convolve
        conv_out = tf.nn.conv2d(
            input=self._channels_to_batch(concat_inp),
            filter=tf.ones(self._kernel_size + [1, 1]),
            padding='VALID',
            strides=[1] + self._strides + [1],
            dilations=[1] + self._dilation_rate + [1],
            data_format='NHWC'
        )
        conv_out = self._batch_to_channels(conv_out)
        return self._flatten(conv_out)

    @utils.lru_cache
    def _channels_to_batch(self, t):
        gd = t.shape.as_list()[1:3]
        return tf.reshape(self._transpose_channel_last_to_first(t), [-1] + gd + [1])

    @utils.lru_cache
    def _batch_to_channels(self, t):
        gd = t.shape.as_list()[1:3]
        return self._transpose_channel_first_to_last(tf.reshape(t, [-1, self._num_channels] + gd))

    def _compute_mpe_path_common(self, counts, *input_values):
        if not self._values:
            raise StructureError("{} is missing input values.".format(self))
        # Concatenate inputs along channel axis, should already be done during forward pass
        inp_concat = self._prepare_convolutional_processing(*input_values)
        spatial_counts = tf.reshape(counts, (-1,) + self.output_shape_spatial)

        inp_concat = self._channels_to_batch(inp_concat)
        spatial_counts = self._channels_to_batch(spatial_counts)

        input_counts = tf.nn.conv2d_backprop_input(
            input_sizes=tf.shape(inp_concat),
            filter=tf.ones(self._kernel_size + [1, 1]),
            out_backprop=spatial_counts,
            strides=[1] + self._strides + [1],
            padding='VALID',
            dilations=[1] + self._dilation_rate + [1],
            data_format="NHWC")

        input_counts = self._batch_to_channels(input_counts)

        # In case we have explicitly padded the tensor before forward convolution, we should
        # slice the counts now
        pad_left, pad_right, pad_top, pad_bottom = self.pad_sizes()
        if not any([pad_left, pad_right, pad_top, pad_bottom]):
            return self._split_to_children(input_counts)
        return self._split_to_children(input_counts[:, pad_top:-pad_bottom, pad_left:-pad_right, :])

