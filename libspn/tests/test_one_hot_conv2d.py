#!/usr/bin/env python3

# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

from context import libspn as spn
from test import TestCase
import tensorflow as tf
import numpy as np
import collections
from random import shuffle
from parameterized import parameterized
import itertools
from libspn.tests.test import argsprod


class TestMath(tf.test.TestCase):

    @argsprod([8], [8], [7], [13], [1], [1], [1], [1])
    def test_one_hot_conv2d(self, in_rows, in_cols, in_depth, out_depth, stride_row, stride_col,
                            dilate_row, dilate_col):
        batch_size = 8
        # stride_row, stride_col = 1, 1
        # dilate_row, dilate_col = 1, 1
        # in_rows, in_cols = 4, 4
        # in_depth = 7
        # out_depth = 13

        strides = [stride_row, stride_col]
        dilations = [dilate_row, dilate_col]
        input_data = np.random.rand(batch_size, in_rows, in_cols, in_depth)
        input_ph = tf.placeholder(tf.float32, [None, in_rows, in_cols, in_depth])
        filter_one_hot = np.random.randint(in_depth, size=2 * 2 * out_depth).reshape(
            (2, 2, out_depth))
        filter_dense = np.zeros((2, 2, in_depth, out_depth))
        for i in range(2):
            for j in range(2):
                for c in range(out_depth):
                    filter_dense[i, j, filter_one_hot[i, j, c], c] = 1.0
        
        input_nchw = tf.transpose(input_ph, (0, 3, 1, 2))
        conv_dense = tf.nn.conv2d(input_nchw, filter=filter_dense, strides=[1, 1] + strides,
                                  dilations=[1, 1] + dilations, padding="VALID", data_format="NCHW")
        conv_dense = tf.transpose(conv_dense, (0, 2, 3, 1))
        conv_onehot = spn.utils.one_hot_conv2d(
            input_ph, filter_one_hot, strides=strides, dilations=dilations, padding="VALID")
        print(conv_dense, conv_onehot)
        with tf.Session() as sess:
            conv_dense_out, conv_onehot_out = sess.run(
                [conv_dense, conv_onehot], feed_dict={input_ph: input_data})

        self.assertAllClose(conv_dense_out, conv_onehot_out)

    @argsprod([8], [8], [2], [4], [1, 2], [1, 2], [1, 2], [1, 2])
    def test_one_hot_conv2d_grad(
            self, in_rows, in_cols, in_depth, out_depth, stride_row, stride_col, dilate_row,
            dilate_col):
        print("strides", stride_row, stride_col)
        print("dilates", dilate_row, dilate_col)
        batch_size = 8

        strides = [stride_row, stride_col]
        dilations = [dilate_row, dilate_col]
        input_data = np.random.rand(batch_size, in_rows, in_cols, in_depth)
        input_ph = tf.placeholder(tf.float32, [None, in_rows, in_cols, in_depth])
        filter_one_hot = np.random.randint(in_depth, size=2 * 2 * out_depth).reshape(
            (2, 2, out_depth))
        filter_dense = np.zeros((2, 2, in_depth, out_depth))
        for i in range(2):
            for j in range(2):
                for c in range(out_depth):
                    filter_dense[i, j, filter_one_hot[i, j, c], c] = 1.0


        conv_onehot = spn.utils.one_hot_conv2d(
            input_ph, filter_one_hot, strides=strides, dilations=dilations, padding="VALID")
        conv_onehot_shape = [batch_size] + conv_onehot.shape.as_list()[1:]
        conv_onehot = conv_onehot
        with self.test_session():
            grad_theoretical, grad_numerical = tf.test.compute_gradient(
                input_ph, input_data.shape, conv_onehot, conv_onehot_shape,
                extra_feed_dict={input_ph: input_data}, delta=2e-3)
        self.assertAllClose(grad_numerical, grad_theoretical, rtol=1e-4)
