from libspn.tests.test import argsprod
import tensorflow as tf
import numpy as np
import libspn as spn
from libspn.log import get_logger

logger = get_logger()


class TestConvProd(tf.test.TestCase):

    @argsprod([4, 2, 1], [4, 2, 1], [2, 3])
    def test_value(self, stride, dilate, kernel_size):
        indicator_rows, indicator_columns = 16, 16
        batch_size = 32
        conv_prod, depthwise, feed_dict, indicators = self.build_test_spn(
            batch_size, dilate, indicator_columns, indicator_rows, kernel_size, stride)

        val_convprod = conv_prod.get_log_value()
        val_depthwise = depthwise.get_log_value()

        with self.test_session() as sess:
            conv_prod_out, depthwise_out = sess.run(
                [val_convprod, val_depthwise], feed_dict=feed_dict)

        self.assertAllClose(np.exp(depthwise_out), np.exp(conv_prod_out))


    def build_test_spn(self, batch_size, dilate, indicator_cols, indicator_rows, kernel_size, stride):
        num_vars = indicator_rows * indicator_cols
        indicator_leaf = spn.RawLeaf(num_vars=num_vars * 4)
        dense_connections = np.zeros((kernel_size, kernel_size, 4, 4), dtype=np.float32)
        for i in range(4):
            dense_connections[:, :, i, i] = 1.0
        grid_dims = [indicator_rows, indicator_cols]
        conv_prod = spn.ConvProducts(indicator_leaf, num_channels=4, dense_connections=dense_connections,
                                     strides=stride, dilation_rate=dilate, spatial_dim_sizes=grid_dims,
                                     kernel_size=kernel_size)
        depthwise = spn.ConvProductsDepthwise(indicator_leaf, spatial_dim_sizes=grid_dims, strides=stride,
                                              dilation_rate=dilate, kernel_size=kernel_size)
        feed_dict = {indicator_leaf: np.random.rand(batch_size, num_vars * 4)}
        return conv_prod, depthwise, feed_dict, indicator_leaf



