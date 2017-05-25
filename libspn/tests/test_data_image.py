#!/usr/bin/env python3

# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

import unittest
import os
import tensorflow as tf
import numpy as np
from context import libspn as spn


class TestImageDataset(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data_dir = os.path.realpath(os.path.join(os.getcwd(),
                                                     os.path.dirname(__file__),
                                                     "data"))

    def tearDown(self):
        tf.reset_default_graph()

    @staticmethod
    def data_path(p):
        if isinstance(p, list):
            return [os.path.join(TestImageDataset.data_dir, i) for i in p]
        else:
            return os.path.join(TestImageDataset.data_dir, p)

    def generic_dataset_test(self, dataset, correct_batches, tol=0.0):
        data = dataset.get_data()
        batches = []
        with spn.session() as (sess, run):
            while run():
                out = sess.run(data)
                batches.append(out)
        self.assertEqual(len(batches), len(correct_batches))
        for b, cb in zip(batches, correct_batches):
            if isinstance(b, list):
                self.assertEqual(len(b), len(cb))
                for bb, cbcb in zip(b, cb):
                    if cbcb is None:
                        self.assertIs(bb, None)
                    else:
                        self.assertEqual(bb.dtype, cbcb.dtype)
                        if (issubclass(bb.dtype.type, np.floating) or
                                issubclass(bb.dtype.type, np.integer)):
                            np.testing.assert_allclose(bb, cbcb, atol=tol)
                        else:
                            np.testing.assert_equal(bb, cbcb)
            else:
                if cb is None:
                    self.assertIs(b, None)
                else:
                    self.assertEqual(b.dtype, cb.dtype)
                    if (issubclass(b.dtype.type, np.floating) or
                            issubclass(b.dtype.type, np.integer)):
                        np.testing.assert_allclose(b, cb, atol=tol)
                    else:
                        np.testing.assert_equal(b, cb)

    def test_image_dataset_pnggray_int_noproc_smaller(self):
        dataset = spn.ImageDataset(image_files=self.data_path("img_dir1/*-{*}.png"),
                                   format=spn.ImageFormat.INT,
                                   num_epochs=3, batch_size=2, shuffle=False,
                                   ratio=1, crop=0, accurate=True,
                                   allow_smaller_final_batch=True)
        batches = [
            [np.array([[0, 0, 255, 0, 0,     # A
                        0, 255, 0, 255, 0,
                        0, 255, 255, 255, 0,
                        0, 255, 0, 255, 0,
                        0, 255, 0, 255, 170],
                       [0, 0, 255, 255, 0,   # C
                        0, 255, 0, 0, 0,
                        0, 255, 0, 0, 0,
                        0, 255, 0, 0, 0,
                        0, 0, 255, 255, 170]], dtype=np.uint8),
             np.array([b'A', b'C'], dtype=object)],
            [np.array([[0, 255, 255, 0, 0,   # B
                        0, 255, 0, 255, 0,
                        0, 255, 255, 255, 0,
                        0, 255, 0, 255, 0,
                        0, 255, 255, 0, 170],
                       [0, 0, 255, 0, 0,     # A
                        0, 255, 0, 255, 0,
                        0, 255, 255, 255, 0,
                        0, 255, 0, 255, 0,
                        0, 255, 0, 255, 170]], dtype=np.uint8),
             np.array([b'B', b'A'], dtype=object)],
            [np.array([[0, 0, 255, 255, 0,   # C
                        0, 255, 0, 0, 0,
                        0, 255, 0, 0, 0,
                        0, 255, 0, 0, 0,
                        0, 0, 255, 255, 170],
                       [0, 255, 255, 0, 0,   # B
                        0, 255, 0, 255, 0,
                        0, 255, 255, 255, 0,
                        0, 255, 0, 255, 0,
                        0, 255, 255, 0, 170]], dtype=np.uint8),
             np.array([b'C', b'B'], dtype=object)],
            [np.array([[0, 0, 255, 0, 0,     # A
                        0, 255, 0, 255, 0,
                        0, 255, 255, 255, 0,
                        0, 255, 0, 255, 0,
                        0, 255, 0, 255, 170],
                       [0, 0, 255, 255, 0,   # C
                        0, 255, 0, 0, 0,
                        0, 255, 0, 0, 0,
                        0, 255, 0, 0, 0,
                        0, 0, 255, 255, 170]], dtype=np.uint8),
             np.array([b'A', b'C'], dtype=object)],
            [np.array([[0, 255, 255, 0, 0,   # B
                        0, 255, 0, 255, 0,
                        0, 255, 255, 255, 0,
                        0, 255, 0, 255, 0,
                        0, 255, 255, 0, 170]], dtype=np.uint8),
             np.array([b'B'], dtype=object)]]

        self.generic_dataset_test(dataset, batches)

    def test_image_dataset_pnggray_int_noproc_nosmaller(self):
        dataset = spn.ImageDataset(image_files=self.data_path("img_dir1/*-{*}.png"),
                                   format=spn.ImageFormat.INT,
                                   num_epochs=3, batch_size=2, shuffle=False,
                                   ratio=1, crop=0, accurate=True,
                                   allow_smaller_final_batch=False)
        batches = [
            [np.array([[0, 0, 255, 0, 0,     # A
                        0, 255, 0, 255, 0,
                        0, 255, 255, 255, 0,
                        0, 255, 0, 255, 0,
                        0, 255, 0, 255, 170],
                       [0, 0, 255, 255, 0,   # C
                        0, 255, 0, 0, 0,
                        0, 255, 0, 0, 0,
                        0, 255, 0, 0, 0,
                        0, 0, 255, 255, 170]], dtype=np.uint8),
             np.array([b'A', b'C'], dtype=object)],
            [np.array([[0, 255, 255, 0, 0,   # B
                        0, 255, 0, 255, 0,
                        0, 255, 255, 255, 0,
                        0, 255, 0, 255, 0,
                        0, 255, 255, 0, 170],
                       [0, 0, 255, 0, 0,     # A
                        0, 255, 0, 255, 0,
                        0, 255, 255, 255, 0,
                        0, 255, 0, 255, 0,
                        0, 255, 0, 255, 170]], dtype=np.uint8),
             np.array([b'B', b'A'], dtype=object)],
            [np.array([[0, 0, 255, 255, 0,   # C
                        0, 255, 0, 0, 0,
                        0, 255, 0, 0, 0,
                        0, 255, 0, 0, 0,
                        0, 0, 255, 255, 170],
                       [0, 255, 255, 0, 0,   # B
                        0, 255, 0, 255, 0,
                        0, 255, 255, 255, 0,
                        0, 255, 0, 255, 0,
                        0, 255, 255, 0, 170]], dtype=np.uint8),
             np.array([b'C', b'B'], dtype=object)],
            [np.array([[0, 0, 255, 0, 0,     # A
                        0, 255, 0, 255, 0,
                        0, 255, 255, 255, 0,
                        0, 255, 0, 255, 0,
                        0, 255, 0, 255, 170],
                       [0, 0, 255, 255, 0,   # C
                        0, 255, 0, 0, 0,
                        0, 255, 0, 0, 0,
                        0, 255, 0, 0, 0,
                        0, 0, 255, 255, 170]], dtype=np.uint8),
             np.array([b'A', b'C'], dtype=object)]]

        self.generic_dataset_test(dataset, batches)

    def test_image_dataset_pnggray_binary_noproc_smaller(self):
        dataset = spn.ImageDataset(image_files=self.data_path("img_dir1/*-{*}.png"),
                                   format=spn.ImageFormat.BINARY,
                                   num_epochs=1, batch_size=2, shuffle=False,
                                   ratio=1, crop=0, accurate=True,
                                   allow_smaller_final_batch=True)
        batches = [
            [np.array([[0, 0, 1, 0, 0,     # A
                        0, 1, 0, 1, 0,
                        0, 1, 1, 1, 0,
                        0, 1, 0, 1, 0,
                        0, 1, 0, 1, 1],
                       [0, 0, 1, 1, 0,   # C
                        0, 1, 0, 0, 0,
                        0, 1, 0, 0, 0,
                        0, 1, 0, 0, 0,
                        0, 0, 1, 1, 1]], dtype=np.uint8),
             np.array([b'A', b'C'], dtype=object)],
            [np.array([[0, 1, 1, 0, 0,   # B
                        0, 1, 0, 1, 0,
                        0, 1, 1, 1, 0,
                        0, 1, 0, 1, 0,
                        0, 1, 1, 0, 1]], dtype=np.uint8),
             np.array([b'B'], dtype=object)]]

        self.generic_dataset_test(dataset, batches)

    def test_image_dataset_pnggray_float_noproc_smaller(self):
        dataset = spn.ImageDataset(image_files=self.data_path("img_dir1/*-{*}.png"),
                                   format=spn.ImageFormat.FLOAT,
                                   num_epochs=1, batch_size=2, shuffle=False,
                                   ratio=1, crop=0, accurate=True,
                                   allow_smaller_final_batch=True)
        batches = [
            [np.array([[0., 0., 1., 0., 0.,     # A
                        0., 1., 0., 1., 0.,
                        0., 1., 1., 1., 0.,
                        0., 1., 0., 1., 0.,
                        0., 1., 0., 1., 2 / 3],
                       [0., 0., 1., 1., 0.,   # C
                        0., 1., 0., 0., 0.,
                        0., 1., 0., 0., 0.,
                        0., 1., 0., 0., 0.,
                        0., 0., 1., 1., 2 / 3]], dtype=np.float32),
             np.array([b'A', b'C'], dtype=object)],
            [np.array([[0., 1., 1., 0., 0.,   # B
                        0., 1., 0., 1., 0.,
                        0., 1., 1., 1., 0.,
                        0., 1., 0., 1., 0.,
                        0., 1., 1., 0., 2 / 3]], dtype=np.float32),
             np.array([b'B'], dtype=object)]]

        self.generic_dataset_test(dataset, batches)

    def test_image_dataset_pnggray_rgbint_noproc_smaller(self):
        dataset = spn.ImageDataset(image_files=self.data_path("img_dir1/*-{*}.png"),
                                   format=spn.ImageFormat.RGB_INT,
                                   num_epochs=1, batch_size=2, shuffle=False,
                                   ratio=1, crop=0, accurate=True,
                                   allow_smaller_final_batch=True)
        batches = [[
            np.array(
                [[0, 0, 0, 0, 0, 0, 255, 255, 255, 0, 0, 0, 0, 0, 0,     # A
                  0, 0, 0, 255, 255, 255, 0, 0, 0, 255, 255, 255, 0, 0, 0,
                  0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0,
                  0, 0, 0, 255, 255, 255, 0, 0, 0, 255, 255, 255, 0, 0, 0,
                  0, 0, 0, 255, 255, 255, 0, 0, 0, 255, 255, 255, 170, 170, 170],
                 [0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 255, 0, 0, 0,   # C
                    0, 0, 0, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 255, 170, 170, 170]],
                dtype=np.uint8),
            np.array([b'A', b'C'], dtype=object)],
            [np.array(
                [[0, 0, 0, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0,   # B
                  0, 0, 0, 255, 255, 255, 0, 0, 0, 255, 255, 255, 0, 0, 0,
                  0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0,
                  0, 0, 0, 255, 255, 255, 0, 0, 0, 255, 255, 255, 0, 0, 0,
                  0, 0, 0, 255, 255, 255, 255, 255, 255, 0, 0, 0, 170, 170, 170]],
                dtype=np.uint8),
             np.array([b'B'], dtype=object)]]

        self.generic_dataset_test(dataset, batches)

    def test_image_dataset_pnggray_rgbbinary_noproc_smaller(self):
        dataset = spn.ImageDataset(image_files=self.data_path("img_dir1/*-{*}.png"),
                                   format=spn.ImageFormat.RGB_BINARY,
                                   num_epochs=1, batch_size=2, shuffle=False,
                                   ratio=1, crop=0, accurate=True,
                                   allow_smaller_final_batch=True)
        batches = [[
            np.array(
                [[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0,     # A
                  0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0,
                  0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
                  0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0,
                  0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                 [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0,   # C
                    0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
                dtype=np.uint8),
            np.array([b'A', b'C'], dtype=object)],
            [np.array(
                [[0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,   # B
                  0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0,
                  0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
                  0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0,
                  0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1]],
                dtype=np.uint8),
             np.array([b'B'], dtype=object)]]

        self.generic_dataset_test(dataset, batches)

    def test_image_dataset_pnggray_rgbfloat_noproc_smaller(self):
        dataset = spn.ImageDataset(image_files=self.data_path("img_dir1/*-{*}.png"),
                                   format=spn.ImageFormat.RGB_FLOAT,
                                   num_epochs=1, batch_size=2, shuffle=False,
                                   ratio=1, crop=0, accurate=True,
                                   allow_smaller_final_batch=True)
        batches = [[
            np.array(
                [[0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0.,   # A
                  0., 0., 0., 1., 1., 1., 0., 0., 0., 1., 1., 1., 0., 0., 0.,
                  0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0.,
                  0., 0., 0., 1., 1., 1., 0., 0., 0., 1., 1., 1., 0., 0., 0.,
                  0., 0., 0., 1., 1., 1., 0., 0., 0., 1., 1., 1., 2 / 3, 2 / 3, 2 / 3],
                 [0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 0., 0., 0.,   # C
                    0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                    0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                    0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                    0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 2 / 3, 2 / 3, 2 / 3]],
                dtype=np.float32),
            np.array([b'A', b'C'], dtype=object)],
            [np.array(
                [[0., 0., 0., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0.,   # B
                  0., 0., 0., 1., 1., 1., 0., 0., 0., 1., 1., 1., 0., 0., 0.,
                  0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0.,
                  0., 0., 0., 1., 1., 1., 0., 0., 0., 1., 1., 1., 0., 0., 0.,
                  0., 0., 0., 1., 1., 1., 1., 1., 1., 0., 0., 0., 2 / 3, 2 / 3, 2 / 3]],
                dtype=np.float32),
             np.array([b'B'], dtype=object)]]

        self.generic_dataset_test(dataset, batches)

    def test_image_dataset_pngrgb_int_noproc_smaller(self):
        dataset = spn.ImageDataset(image_files=self.data_path("img_dir2/*-{*}.png"),
                                   format=spn.ImageFormat.INT,
                                   num_epochs=1, batch_size=2, shuffle=False,
                                   ratio=1, crop=0, accurate=True,
                                   allow_smaller_final_batch=True)
        batches = [
            [np.array([[0, 0, 255, 0, 0,     # A
                        0, 255, 0, 255, 0,
                        0, 255, 255, 255, 0,
                        0, 255, 0, 255, 0,
                        0, 255, 0, 255, 181],
                       [0, 0, 255, 255, 0,   # C
                        0, 255, 0, 0, 0,
                        0, 255, 0, 0, 0,
                        0, 255, 0, 0, 0,
                        0, 0, 255, 255, 181]], dtype=np.uint8),
             np.array([b'A', b'C'], dtype=object)],
            [np.array([[0, 255, 255, 0, 0,   # B
                        0, 255, 0, 255, 0,
                        0, 255, 255, 255, 0,
                        0, 255, 0, 255, 0,
                        0, 255, 255, 0, 181]], dtype=np.uint8),
             np.array([b'B'], dtype=object)]]

        self.generic_dataset_test(dataset, batches)

    def test_image_dataset_pngrgb_binary_noproc_smaller(self):
        dataset = spn.ImageDataset(image_files=self.data_path("img_dir2/*-{*}.png"),
                                   format=spn.ImageFormat.BINARY,
                                   num_epochs=1, batch_size=2, shuffle=False,
                                   ratio=1, crop=0, accurate=True,
                                   allow_smaller_final_batch=True)
        batches = [
            [np.array([[0, 0, 1, 0, 0,     # A
                        0, 1, 0, 1, 0,
                        0, 1, 1, 1, 0,
                        0, 1, 0, 1, 0,
                        0, 1, 0, 1, 1],
                       [0, 0, 1, 1, 0,   # C
                        0, 1, 0, 0, 0,
                        0, 1, 0, 0, 0,
                        0, 1, 0, 0, 0,
                        0, 0, 1, 1, 1]], dtype=np.uint8),
             np.array([b'A', b'C'], dtype=object)],
            [np.array([[0, 1, 1, 0, 0,   # B
                        0, 1, 0, 1, 0,
                        0, 1, 1, 1, 0,
                        0, 1, 0, 1, 0,
                        0, 1, 1, 0, 1]], dtype=np.uint8),
             np.array([b'B'], dtype=object)]]

        self.generic_dataset_test(dataset, batches)

    def test_image_dataset_pngrgb_float_noproc_smaller(self):
        dataset = spn.ImageDataset(image_files=self.data_path("img_dir2/*-{*}.png"),
                                   format=spn.ImageFormat.FLOAT,
                                   num_epochs=1, batch_size=2, shuffle=False,
                                   ratio=1, crop=0, accurate=True,
                                   allow_smaller_final_batch=True)
        batches = [
            [np.array([[0., 0., 1., 0., 0.,     # A
                        0., 1., 0., 1., 0.,
                        0., 1., 1., 1., 0.,
                        0., 1., 0., 1., 0.,
                        0., 1., 0., 1., 0.70870972],
                       [0., 0., 1., 1., 0.,   # C
                        0., 1., 0., 0., 0.,
                        0., 1., 0., 0., 0.,
                        0., 1., 0., 0., 0.,
                        0., 0., 1., 1., 0.70870972]], dtype=np.float32),
             np.array([b'A', b'C'], dtype=object)],
            [np.array([[0., 1., 1., 0., 0.,   # B
                        0., 1., 0., 1., 0.,
                        0., 1., 1., 1., 0.,
                        0., 1., 0., 1., 0.,
                        0., 1., 1., 0., 0.70870972]], dtype=np.float32),
             np.array([b'B'], dtype=object)]]

        self.generic_dataset_test(dataset, batches)

    def test_image_dataset_pngrgb_rgbint_noproc_smaller(self):
        dataset = spn.ImageDataset(image_files=self.data_path("img_dir2/*-{*}.png"),
                                   format=spn.ImageFormat.RGB_INT,
                                   num_epochs=1, batch_size=2, shuffle=False,
                                   ratio=1, crop=0, accurate=True,
                                   allow_smaller_final_batch=True)
        batches = [[
            np.array(
                [[0, 0, 0, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0,     # A
                  0, 0, 0, 255, 0, 0, 0, 0, 0, 255, 0, 0, 0, 0, 0,
                  0, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 0, 0, 0,
                  0, 0, 0, 255, 0, 0, 0, 0, 0, 255, 0, 0, 0, 0, 0,
                  0, 0, 0, 255, 0, 0, 0, 0, 0, 255, 0, 0, 0, 42, 255],
                 [0, 0, 0, 0, 0, 0, 255, 0, 0, 255, 0, 0, 0, 0, 0,   # C
                    0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 255, 0, 0, 255, 0, 0, 0, 42, 255]],
                dtype=np.uint8),
            np.array([b'A', b'C'], dtype=object)],
            [np.array(
                [[0, 0, 0, 255, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0,   # B
                  0, 0, 0, 255, 0, 0, 0, 0, 0, 255, 0, 0, 0, 0, 0,
                  0, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 0, 0, 0,
                  0, 0, 0, 255, 0, 0, 0, 0, 0, 255, 0, 0, 0, 0, 0,
                  0, 0, 0, 255, 0, 0, 255, 0, 0, 0, 0, 0, 0, 42, 255]],
                dtype=np.uint8),
             np.array([b'B'], dtype=object)]]

        self.generic_dataset_test(dataset, batches)

    def test_image_dataset_pngrgb_rgbbinary_noproc_smaller(self):
        dataset = spn.ImageDataset(image_files=self.data_path("img_dir2/*-{*}.png"),
                                   format=spn.ImageFormat.RGB_BINARY,
                                   num_epochs=1, batch_size=2, shuffle=False,
                                   ratio=1, crop=0, accurate=True,
                                   allow_smaller_final_batch=True)
        batches = [[
            np.array(
                [[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,     # A
                  0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                  0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0,
                  0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                  0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0,   # C
                    0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1]],
                dtype=np.uint8),
            np.array([b'A', b'C'], dtype=object)],
            [np.array(
                [[0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,   # B
                  0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                  0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0,
                  0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                  0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1]],
                dtype=np.uint8),
             np.array([b'B'], dtype=object)]]

        self.generic_dataset_test(dataset, batches)

    def test_image_dataset_pngrgb_rgbfloat_noproc_smaller(self):
        dataset = spn.ImageDataset(image_files=self.data_path("img_dir2/*-{*}.png"),
                                   format=spn.ImageFormat.RGB_FLOAT,
                                   num_epochs=1, batch_size=2, shuffle=False,
                                   ratio=1, crop=0, accurate=True,
                                   allow_smaller_final_batch=True)
        batches = [[
            np.array(
                [[0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,   # A
                  0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
                  0., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0.,
                  0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
                  0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1 / 6, 1.],
                 [0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0.,   # C
                    0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                    0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                    0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                    0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1 / 6, 1.]],
                dtype=np.float32),
            np.array([b'A', b'C'], dtype=object)],
            [np.array(
                [[0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,   # B
                  0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
                  0., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0.,
                  0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
                  0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1 / 6, 1.]],
                dtype=np.float32),
             np.array([b'B'], dtype=object)]]

        self.generic_dataset_test(dataset, batches)

    def test_image_dataset_jpggray_int_noproc_smaller(self):
        dataset = spn.ImageDataset(image_files=self.data_path("img_dir1/*-{*}.jpg"),
                                   format=spn.ImageFormat.INT,
                                   num_epochs=3, batch_size=2, shuffle=False,
                                   ratio=1, crop=0, accurate=True,
                                   allow_smaller_final_batch=True)
        batches = [
            [np.array([[0, 0, 255, 0, 0,     # A
                        0, 255, 0, 255, 0,
                        0, 255, 255, 255, 0,
                        0, 255, 0, 255, 0,
                        0, 255, 0, 255, 170],
                       [0, 0, 255, 255, 0,   # C
                        0, 255, 0, 0, 0,
                        0, 255, 0, 0, 0,
                        0, 255, 0, 0, 0,
                        0, 0, 255, 255, 170]], dtype=np.uint8),
             np.array([b'A', b'C'], dtype=object)],
            [np.array([[0, 255, 255, 0, 0,   # B
                        0, 255, 0, 255, 0,
                        0, 255, 255, 255, 0,
                        0, 255, 0, 255, 0,
                        0, 255, 255, 0, 170],
                       [0, 0, 255, 0, 0,     # A
                        0, 255, 0, 255, 0,
                        0, 255, 255, 255, 0,
                        0, 255, 0, 255, 0,
                        0, 255, 0, 255, 170]], dtype=np.uint8),
             np.array([b'B', b'A'], dtype=object)],
            [np.array([[0, 0, 255, 255, 0,   # C
                        0, 255, 0, 0, 0,
                        0, 255, 0, 0, 0,
                        0, 255, 0, 0, 0,
                        0, 0, 255, 255, 170],
                       [0, 255, 255, 0, 0,   # B
                        0, 255, 0, 255, 0,
                        0, 255, 255, 255, 0,
                        0, 255, 0, 255, 0,
                        0, 255, 255, 0, 170]], dtype=np.uint8),
             np.array([b'C', b'B'], dtype=object)],
            [np.array([[0, 0, 255, 0, 0,     # A
                        0, 255, 0, 255, 0,
                        0, 255, 255, 255, 0,
                        0, 255, 0, 255, 0,
                        0, 255, 0, 255, 170],
                       [0, 0, 255, 255, 0,   # C
                        0, 255, 0, 0, 0,
                        0, 255, 0, 0, 0,
                        0, 255, 0, 0, 0,
                        0, 0, 255, 255, 170]], dtype=np.uint8),
             np.array([b'A', b'C'], dtype=object)],
            [np.array([[0, 255, 255, 0, 0,   # B
                        0, 255, 0, 255, 0,
                        0, 255, 255, 255, 0,
                        0, 255, 0, 255, 0,
                        0, 255, 255, 0, 170]], dtype=np.uint8),
             np.array([b'B'], dtype=object)]]

        self.generic_dataset_test(dataset, batches, tol=70)

    def test_image_dataset_jpggray_int_noproc_nosmaller(self):
        dataset = spn.ImageDataset(image_files=self.data_path("img_dir1/*-{*}.jpg"),
                                   format=spn.ImageFormat.INT,
                                   num_epochs=3, batch_size=2, shuffle=False,
                                   ratio=1, crop=0, accurate=True,
                                   allow_smaller_final_batch=False)
        batches = [
            [np.array([[0, 0, 255, 0, 0,     # A
                        0, 255, 0, 255, 0,
                        0, 255, 255, 255, 0,
                        0, 255, 0, 255, 0,
                        0, 255, 0, 255, 170],
                       [0, 0, 255, 255, 0,   # C
                        0, 255, 0, 0, 0,
                        0, 255, 0, 0, 0,
                        0, 255, 0, 0, 0,
                        0, 0, 255, 255, 170]], dtype=np.uint8),
             np.array([b'A', b'C'], dtype=object)],
            [np.array([[0, 255, 255, 0, 0,   # B
                        0, 255, 0, 255, 0,
                        0, 255, 255, 255, 0,
                        0, 255, 0, 255, 0,
                        0, 255, 255, 0, 170],
                       [0, 0, 255, 0, 0,     # A
                        0, 255, 0, 255, 0,
                        0, 255, 255, 255, 0,
                        0, 255, 0, 255, 0,
                        0, 255, 0, 255, 170]], dtype=np.uint8),
             np.array([b'B', b'A'], dtype=object)],
            [np.array([[0, 0, 255, 255, 0,   # C
                        0, 255, 0, 0, 0,
                        0, 255, 0, 0, 0,
                        0, 255, 0, 0, 0,
                        0, 0, 255, 255, 170],
                       [0, 255, 255, 0, 0,   # B
                        0, 255, 0, 255, 0,
                        0, 255, 255, 255, 0,
                        0, 255, 0, 255, 0,
                        0, 255, 255, 0, 170]], dtype=np.uint8),
             np.array([b'C', b'B'], dtype=object)],
            [np.array([[0, 0, 255, 0, 0,     # A
                        0, 255, 0, 255, 0,
                        0, 255, 255, 255, 0,
                        0, 255, 0, 255, 0,
                        0, 255, 0, 255, 170],
                       [0, 0, 255, 255, 0,   # C
                        0, 255, 0, 0, 0,
                        0, 255, 0, 0, 0,
                        0, 255, 0, 0, 0,
                        0, 0, 255, 255, 170]], dtype=np.uint8),
             np.array([b'A', b'C'], dtype=object)]]

        self.generic_dataset_test(dataset, batches, tol=70)

    def test_image_dataset_jpggray_binary_noproc_smaller(self):
        dataset = spn.ImageDataset(image_files=self.data_path("img_dir1/*-{*}.jpg"),
                                   format=spn.ImageFormat.BINARY,
                                   num_epochs=1, batch_size=2, shuffle=False,
                                   ratio=1, crop=0, accurate=True,
                                   allow_smaller_final_batch=True)
        batches = [
            [np.array([[0, 0, 1, 0, 0,     # A
                        0, 1, 0, 1, 0,
                        0, 1, 1, 1, 0,
                        0, 1, 0, 1, 0,
                        0, 1, 0, 1, 1],
                       [0, 0, 1, 1, 0,   # C
                        0, 1, 0, 0, 0,
                        0, 1, 0, 0, 0,
                        0, 1, 0, 0, 0,
                        0, 0, 1, 1, 1]], dtype=np.uint8),
             np.array([b'A', b'C'], dtype=object)],
            [np.array([[0, 1, 1, 0, 0,   # B
                        0, 1, 0, 1, 0,
                        0, 1, 1, 1, 0,
                        0, 1, 0, 1, 0,
                        0, 1, 1, 0, 1]], dtype=np.uint8),
             np.array([b'B'], dtype=object)]]

        self.generic_dataset_test(dataset, batches)

    def test_image_dataset_jpggray_float_noproc_smaller(self):
        dataset = spn.ImageDataset(image_files=self.data_path("img_dir1/*-{*}.jpg"),
                                   format=spn.ImageFormat.FLOAT,
                                   num_epochs=1, batch_size=2, shuffle=False,
                                   ratio=1, crop=0, accurate=True,
                                   allow_smaller_final_batch=True)
        batches = [
            [np.array([[0., 0., 1., 0., 0.,     # A
                        0., 1., 0., 1., 0.,
                        0., 1., 1., 1., 0.,
                        0., 1., 0., 1., 0.,
                        0., 1., 0., 1., 2 / 3],
                       [0., 0., 1., 1., 0.,   # C
                        0., 1., 0., 0., 0.,
                        0., 1., 0., 0., 0.,
                        0., 1., 0., 0., 0.,
                        0., 0., 1., 1., 2 / 3]], dtype=np.float32),
             np.array([b'A', b'C'], dtype=object)],
            [np.array([[0., 1., 1., 0., 0.,   # B
                        0., 1., 0., 1., 0.,
                        0., 1., 1., 1., 0.,
                        0., 1., 0., 1., 0.,
                        0., 1., 1., 0., 2 / 3]], dtype=np.float32),
             np.array([b'B'], dtype=object)]]

        self.generic_dataset_test(dataset, batches, tol=0.3)

    def test_image_dataset_jpggray_rgbint_noproc_smaller(self):
        dataset = spn.ImageDataset(image_files=self.data_path("img_dir1/*-{*}.jpg"),
                                   format=spn.ImageFormat.RGB_INT,
                                   num_epochs=1, batch_size=2, shuffle=False,
                                   ratio=1, crop=0, accurate=True,
                                   allow_smaller_final_batch=True)
        batches = [[
            np.array(
                [[0, 0, 0, 0, 0, 0, 255, 255, 255, 0, 0, 0, 0, 0, 0,     # A
                  0, 0, 0, 255, 255, 255, 0, 0, 0, 255, 255, 255, 0, 0, 0,
                  0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0,
                  0, 0, 0, 255, 255, 255, 0, 0, 0, 255, 255, 255, 0, 0, 0,
                  0, 0, 0, 255, 255, 255, 0, 0, 0, 255, 255, 255, 170, 170, 170],
                 [0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 255, 0, 0, 0,   # C
                    0, 0, 0, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 255, 170, 170, 170]],
                dtype=np.uint8),
            np.array([b'A', b'C'], dtype=object)],
            [np.array(
                [[0, 0, 0, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0,   # B
                  0, 0, 0, 255, 255, 255, 0, 0, 0, 255, 255, 255, 0, 0, 0,
                  0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0,
                  0, 0, 0, 255, 255, 255, 0, 0, 0, 255, 255, 255, 0, 0, 0,
                  0, 0, 0, 255, 255, 255, 255, 255, 255, 0, 0, 0, 170, 170, 170]],
                dtype=np.uint8),
             np.array([b'B'], dtype=object)]]

        self.generic_dataset_test(dataset, batches, tol=70)

    def test_image_dataset_jpggray_rgbbinary_noproc_smaller(self):
        dataset = spn.ImageDataset(image_files=self.data_path("img_dir1/*-{*}.jpg"),
                                   format=spn.ImageFormat.RGB_BINARY,
                                   num_epochs=1, batch_size=2, shuffle=False,
                                   ratio=1, crop=0, accurate=True,
                                   allow_smaller_final_batch=True)
        batches = [[
            np.array(
                [[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0,     # A
                  0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0,
                  0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
                  0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0,
                  0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                 [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0,   # C
                    0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
                dtype=np.uint8),
            np.array([b'A', b'C'], dtype=object)],
            [np.array(
                [[0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,   # B
                  0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0,
                  0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
                  0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0,
                  0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1]],
                dtype=np.uint8),
             np.array([b'B'], dtype=object)]]

        self.generic_dataset_test(dataset, batches)

    def test_image_dataset_jpggray_rgbfloat_noproc_smaller(self):
        dataset = spn.ImageDataset(image_files=self.data_path("img_dir1/*-{*}.jpg"),
                                   format=spn.ImageFormat.RGB_FLOAT,
                                   num_epochs=1, batch_size=2, shuffle=False,
                                   ratio=1, crop=0, accurate=True,
                                   allow_smaller_final_batch=True)
        batches = [[
            np.array(
                [[0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0.,   # A
                  0., 0., 0., 1., 1., 1., 0., 0., 0., 1., 1., 1., 0., 0., 0.,
                  0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0.,
                  0., 0., 0., 1., 1., 1., 0., 0., 0., 1., 1., 1., 0., 0., 0.,
                  0., 0., 0., 1., 1., 1., 0., 0., 0., 1., 1., 1., 2 / 3, 2 / 3, 2 / 3],
                 [0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 0., 0., 0.,   # C
                    0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                    0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                    0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                    0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 2 / 3, 2 / 3, 2 / 3]],
                dtype=np.float32),
            np.array([b'A', b'C'], dtype=object)],
            [np.array(
                [[0., 0., 0., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0.,   # B
                  0., 0., 0., 1., 1., 1., 0., 0., 0., 1., 1., 1., 0., 0., 0.,
                  0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0.,
                  0., 0., 0., 1., 1., 1., 0., 0., 0., 1., 1., 1., 0., 0., 0.,
                  0., 0., 0., 1., 1., 1., 1., 1., 1., 0., 0., 0., 2 / 3, 2 / 3, 2 / 3]],
                dtype=np.float32),
             np.array([b'B'], dtype=object)]]

        self.generic_dataset_test(dataset, batches, tol=0.3)

    def test_image_dataset_jpgrgb_int_noproc_smaller(self):
        dataset = spn.ImageDataset(image_files=self.data_path("img_dir2/*-{*}.jpg"),
                                   format=spn.ImageFormat.INT,
                                   num_epochs=1, batch_size=2, shuffle=False,
                                   ratio=1, crop=0, accurate=True,
                                   allow_smaller_final_batch=True)
        batches = [
            [np.array([[0, 0, 255, 0, 0,     # A
                        0, 255, 0, 255, 0,
                        0, 255, 255, 255, 0,
                        0, 255, 0, 255, 0,
                        0, 255, 0, 255, 181],
                       [0, 0, 255, 255, 0,   # C
                        0, 255, 0, 0, 0,
                        0, 255, 0, 0, 0,
                        0, 255, 0, 0, 0,
                        0, 0, 255, 255, 181]], dtype=np.uint8),
             np.array([b'A', b'C'], dtype=object)],
            [np.array([[0, 255, 255, 0, 0,   # B
                        0, 255, 0, 255, 0,
                        0, 255, 255, 255, 0,
                        0, 255, 0, 255, 0,
                        0, 255, 255, 0, 181]], dtype=np.uint8),
             np.array([b'B'], dtype=object)]]

        self.generic_dataset_test(dataset, batches, tol=70)

    def test_image_dataset_jpgrgb_binary_noproc_smaller(self):
        dataset = spn.ImageDataset(image_files=self.data_path("img_dir2/*-{*}.jpg"),
                                   format=spn.ImageFormat.BINARY,
                                   num_epochs=1, batch_size=2, shuffle=False,
                                   ratio=1, crop=0, accurate=True,
                                   allow_smaller_final_batch=True)
        batches = [
            [np.array([[0, 0, 1, 0, 0,     # A
                        0, 1, 0, 1, 0,
                        0, 1, 1, 1, 0,
                        0, 1, 0, 1, 0,
                        0, 1, 0, 1, 1],
                       [0, 0, 1, 1, 0,   # C
                        0, 1, 0, 0, 0,
                        0, 1, 0, 0, 0,
                        0, 1, 0, 0, 0,
                        0, 0, 1, 1, 1]], dtype=np.uint8),
             np.array([b'A', b'C'], dtype=object)],
            [np.array([[0, 1, 1, 0, 0,   # B
                        0, 1, 0, 1, 0,
                        0, 1, 1, 1, 0,
                        0, 1, 0, 1, 0,
                        0, 1, 1, 0, 1]], dtype=np.uint8),
             np.array([b'B'], dtype=object)]]

        self.generic_dataset_test(dataset, batches)

    def test_image_dataset_jpgrgb_float_noproc_smaller(self):
        dataset = spn.ImageDataset(image_files=self.data_path("img_dir2/*-{*}.jpg"),
                                   format=spn.ImageFormat.FLOAT,
                                   num_epochs=1, batch_size=2, shuffle=False,
                                   ratio=1, crop=0, accurate=True,
                                   allow_smaller_final_batch=True)
        batches = [
            [np.array([[0., 0., 1., 0., 0.,     # A
                        0., 1., 0., 1., 0.,
                        0., 1., 1., 1., 0.,
                        0., 1., 0., 1., 0.,
                        0., 1., 0., 1., 0.70870972],
                       [0., 0., 1., 1., 0.,   # C
                        0., 1., 0., 0., 0.,
                        0., 1., 0., 0., 0.,
                        0., 1., 0., 0., 0.,
                        0., 0., 1., 1., 0.70870972]], dtype=np.float32),
             np.array([b'A', b'C'], dtype=object)],
            [np.array([[0., 1., 1., 0., 0.,   # B
                        0., 1., 0., 1., 0.,
                        0., 1., 1., 1., 0.,
                        0., 1., 0., 1., 0.,
                        0., 1., 1., 0., 0.70870972]], dtype=np.float32),
             np.array([b'B'], dtype=object)]]

        self.generic_dataset_test(dataset, batches, tol=0.3)

    def test_image_dataset_jpgrgb_rgbint_noproc_smaller(self):
        dataset = spn.ImageDataset(image_files=self.data_path("img_dir2/*-{*}.jpg"),
                                   format=spn.ImageFormat.RGB_INT,
                                   num_epochs=1, batch_size=2, shuffle=False,
                                   ratio=1, crop=0, accurate=True,
                                   allow_smaller_final_batch=True)
        batches = [[
            np.array(
                [[0, 0, 0, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0,     # A
                  0, 0, 0, 255, 0, 0, 0, 0, 0, 255, 0, 0, 0, 0, 0,
                  0, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 0, 0, 0,
                  0, 0, 0, 255, 0, 0, 0, 0, 0, 255, 0, 0, 0, 0, 0,
                  0, 0, 0, 255, 0, 0, 0, 0, 0, 255, 0, 0, 0, 42, 255],
                 [0, 0, 0, 0, 0, 0, 255, 0, 0, 255, 0, 0, 0, 0, 0,   # C
                    0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 255, 0, 0, 255, 0, 0, 0, 42, 255]],
                dtype=np.uint8),
            np.array([b'A', b'C'], dtype=object)],
            [np.array(
                [[0, 0, 0, 255, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0,   # B
                  0, 0, 0, 255, 0, 0, 0, 0, 0, 255, 0, 0, 0, 0, 0,
                  0, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 0, 0, 0,
                  0, 0, 0, 255, 0, 0, 0, 0, 0, 255, 0, 0, 0, 0, 0,
                  0, 0, 0, 255, 0, 0, 255, 0, 0, 0, 0, 0, 0, 42, 255]],
                dtype=np.uint8),
             np.array([b'B'], dtype=object)]]

        self.generic_dataset_test(dataset, batches, tol=70)

    def test_image_dataset_jpgrgb_rgbbinary_noproc_smaller(self):
        dataset = spn.ImageDataset(image_files=self.data_path("img_dir2/*-{*}.jpg"),
                                   format=spn.ImageFormat.RGB_BINARY,
                                   num_epochs=1, batch_size=2, shuffle=False,
                                   ratio=1, crop=0, accurate=True,
                                   allow_smaller_final_batch=True)
        batches = [[
            np.array(
                [[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,     # A
                  0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                  0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0,
                  0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                  0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0,   # C
                    0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1]],
                dtype=np.uint8),
            np.array([b'A', b'C'], dtype=object)],
            [np.array(
                [[0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,   # B
                  0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                  0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0,
                  0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                  0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1]],
                dtype=np.uint8),
             np.array([b'B'], dtype=object)]]

        self.generic_dataset_test(dataset, batches)

    def test_image_dataset_jpgrgb_rgbfloat_noproc_smaller(self):
        dataset = spn.ImageDataset(image_files=self.data_path("img_dir2/*-{*}.jpg"),
                                   format=spn.ImageFormat.RGB_FLOAT,
                                   num_epochs=1, batch_size=2, shuffle=False,
                                   ratio=1, crop=0, accurate=True,
                                   allow_smaller_final_batch=True)
        batches = [[
            np.array(
                [[0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,   # A
                  0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
                  0., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0.,
                  0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
                  0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1 / 6, 1.],
                 [0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0.,   # C
                    0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                    0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                    0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                    0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1 / 6, 1.]],
                dtype=np.float32),
            np.array([b'A', b'C'], dtype=object)],
            [np.array(
                [[0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,   # B
                  0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
                  0., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0.,
                  0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
                  0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1 / 6, 1.]],
                dtype=np.float32),
             np.array([b'B'], dtype=object)]]

        self.generic_dataset_test(dataset, batches, tol=0.3)


if __name__ == '__main__':
    unittest.main()
