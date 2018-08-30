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

    @argsprod([4, 2, 1], [4, 2, 1])
    def test_value(self, stride, dilate):
        ivs_rows, ivs_cols = 16, 16
        batch_size = 32
        num_vars = ivs_rows * ivs_cols
        ivs = spn.ContVars(num_vars=num_vars * 4)

        dense_connections = np.zeros((2, 2, 4, 4), dtype=np.float32)
        for i in range(4):
            dense_connections[:, :, i, i] = 1.0

        grid_dims = [ivs_rows, ivs_cols]
        conv_prod = spn.ConvProd2D(ivs, num_channels=4, dense_connections=dense_connections,
                                   strides=stride, dilation_rate=dilate, grid_dim_sizes=grid_dims)
        depthwise = spn.ConvProdDepthWise(ivs, grid_dim_sizes=grid_dims, strides=stride,
                                          dilation_rate=dilate)

        val_convprod = conv_prod.get_log_value()
        val_depthwise = depthwise.get_log_value()

        feed_dict = {ivs: np.random.rand(batch_size, num_vars * 4)}

        with self.test_session() as sess:
            conv_prod_out, depthwise_out = sess.run(
                [val_convprod, val_depthwise], feed_dict=feed_dict)

        self.assertAllClose(np.exp(depthwise_out), np.exp(conv_prod_out))
