from libspn.graph.basesum import BaseSum
from libspn.tests.test import argsprod
import tensorflow as tf
import numpy as np
import libspn as spn
from libspn.log import get_logger
from libspn.generation.spatial import ConvSPN
import itertools
import random
logger = get_logger()


class TestConvProd(tf.test.TestCase):

    @argsprod([4, 2, 1], [4, 2, 1], [2, 3])
    def test_value(self, stride, dilate, kernel_size):
        ivs_rows, ivs_cols = 16, 16
        batch_size = 32
        conv_prod, depthwise, feed_dict, ivs = self.build_test_spn(batch_size, dilate, ivs_cols,
                                                                   ivs_rows, kernel_size, stride)

        val_convprod = conv_prod.get_log_value()
        val_depthwise = depthwise.get_log_value()

        with self.test_session() as sess:
            conv_prod_out, depthwise_out = sess.run(
                [val_convprod, val_depthwise], feed_dict=feed_dict)

        self.assertAllClose(np.exp(depthwise_out), np.exp(conv_prod_out))

    @argsprod([4, 2, 1], [4, 2, 1], [2, 3])
    def test_mpe_path(self, stride, dilate, kernel_size):
        ivs_rows, ivs_cols = 16, 16
        batch_size = 32
        conv_prod, depthwise, feed_dict, ivs = self.build_test_spn(batch_size, dilate, ivs_cols,
                                                                   ivs_rows, kernel_size, stride)
        out_shape = conv_prod.output_shape_spatial
        counts = np.random.rand(batch_size, *out_shape).astype(np.float32)
        counts_convprod = conv_prod._compute_log_mpe_path(counts, ivs.get_log_value())[0]
        counts_depthwise = depthwise._compute_log_mpe_path(counts, ivs.get_log_value())[0]

        with self.test_session() as sess:
            conv_prod_out, depthwise_out = sess.run(
                [counts_convprod, counts_depthwise], feed_dict=feed_dict)

        self.assertAllClose(depthwise_out, conv_prod_out)

    def build_test_spn(self, batch_size, dilate, ivs_cols, ivs_rows, kernel_size, stride):
        num_vars = ivs_rows * ivs_cols
        ivs = spn.ContVars(num_vars=num_vars * 4)
        dense_connections = np.zeros((kernel_size, kernel_size, 4, 4), dtype=np.float32)
        for i in range(4):
            dense_connections[:, :, i, i] = 1.0
        grid_dims = [ivs_rows, ivs_cols]
        conv_prod = spn.ConvProd2D(ivs, num_channels=4, dense_connections=dense_connections,
                                   strides=stride, dilation_rate=dilate, grid_dim_sizes=grid_dims,
                                   kernel_size=kernel_size)
        depthwise = spn.ConvProdDepthWise(ivs, grid_dim_sizes=grid_dims, strides=stride,
                                          dilation_rate=dilate, kernel_size=kernel_size)
        feed_dict = {ivs: np.random.rand(batch_size, num_vars * 4)}
        return conv_prod, depthwise, feed_dict, ivs



