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

    @argsprod([2, 0], [None, -2], [2, 0], [None, -2], [1, 2], [1, 2])
    def test_value(self, row_b, row_e, col_b, col_e, stride_row, stride_col):
        ivs_rows, ivs_cols = 16, 16
        batch_size = 32
        num_vars = ivs_rows * ivs_cols
        ivs = spn.ContVars(num_vars=num_vars * 2)
        ivs2 = spn.ContVars(num_vars=num_vars * 2)

        if row_e is None:
            if col_e is None:
                end = None
            else:
                end = (ivs_rows, col_e)
        elif col_e is None:
            end = (row_e, ivs_cols)
        else:
            end = (row_e, col_e)

        strided_slice = spn.StridedSlice2D(
            ivs, ivs2, begin=(row_b, col_b), end=end, strides=(stride_row, stride_col),
            grid_dim_sizes=[ivs_rows, ivs_cols])

        val_v1 = strided_slice.get_log_value()

        ivs_feed = np.random.rand(batch_size, num_vars * 2)
        ivs_feed2 = np.random.rand(batch_size, num_vars * 2)

        ivs_feed_spatial_concat = np.concatenate([
            ivs_feed.reshape((batch_size, ivs_rows, ivs_cols, -1)),
            ivs_feed2.reshape((batch_size, ivs_rows, ivs_cols, -1))], axis=3)

        if end is None:
            ivs_feed_sliced = ivs_feed_spatial_concat[
                              :, row_b::stride_row, col_b::stride_col, :]
        else:
            row_e, col_e = end
            ivs_feed_sliced = ivs_feed_spatial_concat[
                              :, row_b:row_e:stride_row, col_b:col_e:stride_col, :]

        with self.test_session() as sess:
            val_v1_out = sess.run(val_v1, {ivs: ivs_feed, ivs2: ivs_feed2})

        self.assertAllClose(np.exp(val_v1_out), ivs_feed_sliced.reshape((batch_size, -1)))

    @argsprod([2, 0], [None, -2], [2, 0], [None, -2], [1, 2], [1, 2])
    def test_counts(self, row_b, row_e, col_b, col_e, stride_row, stride_col):
        ivs_rows, ivs_cols = 16, 16
        batch_size = 32
        num_vars = ivs_rows * ivs_cols
        ivs = spn.ContVars(num_vars=num_vars * 2)
        ivs2 = spn.ContVars(num_vars=num_vars * 2)

        if row_e is None:
            if col_e is None:
                end = None
            else:
                end = (ivs_rows, col_e)
        elif col_e is None:
            end = (row_e, ivs_cols)
        else:
            end = (row_e, col_e)

        strided_slice = spn.StridedSlice2D(
            ivs, ivs2, begin=(row_b, col_b), end=end, strides=(stride_row, stride_col),
            grid_dim_sizes=[ivs_rows, ivs_cols])
        root = spn.Concat(strided_slice)
        counts_np = np.random.rand(batch_size, *strided_slice.output_shape_spatial)

        mpe_path_gen = spn.MPEPath()
        mpe_path_gen.get_mpe_path(root)
        sum_counts = mpe_path_gen.counts[root]


        ivs_feed = np.random.rand(batch_size, num_vars * 2)
        ivs_feed2 = np.random.rand(batch_size, num_vars * 2)

        counts_ivs1, counts_ivs2 = np.split(counts_np, axis=3, indices_or_sections=2)

        truth_ivs1 = np.zeros_like(ivs_feed).reshape((batch_size, ivs_rows, ivs_cols, 2))
        truth_ivs2 = np.zeros_like(ivs_feed2).reshape((batch_size, ivs_rows, ivs_cols, 2))

        if end is None:
            truth_ivs1[:, row_b::stride_row, col_b::stride_col, :] = counts_ivs1
            truth_ivs2[:, row_b::stride_row, col_b::stride_col, :] = counts_ivs2
        else:
            row_e, col_e = end
            truth_ivs1[:, row_b:row_e:stride_row, col_b:col_e:stride_col, :] = counts_ivs1
            truth_ivs2[:, row_b:row_e:stride_row, col_b:col_e:stride_col, :] = counts_ivs2
        with self.test_session() as sess:
            counts_ivs1_out, counts_ivs2_out = sess.run(
                [mpe_path_gen.counts[ivs], mpe_path_gen.counts[ivs2]],
                {ivs: ivs_feed, ivs2: ivs_feed2, sum_counts: counts_np.reshape((batch_size, -1))})

        self.assertAllClose(
            counts_ivs1_out.reshape((batch_size, ivs_rows, ivs_cols, 2)), truth_ivs1)

        self.assertAllClose(
            counts_ivs2_out.reshape((batch_size, ivs_rows, ivs_cols, 2)), truth_ivs2)