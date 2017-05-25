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


class TestData(unittest.TestCase):

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
            return [os.path.join(TestData.data_dir, i) for i in p]
        else:
            return os.path.join(TestData.data_dir, p)

    def generic_dataset_test(self, dataset, correct_batches):
        with self.subTest(dataset=dataset,
                          correct_batches=correct_batches):
            data = dataset.get_data()
            batches = []
            with spn.session() as (sess, run):
                while run():
                    out = sess.run(data)
                    batches.append(out)
            print(batches)
            self.assertEqual(len(batches), len(correct_batches))
            for b, cb in zip(batches, correct_batches):
                if isinstance(b, list):
                    self.assertEqual(len(b), len(cb))
                    for bb, cbcb in zip(b, cb):
                        if cbcb is None:
                            self.assertIs(bb, None)
                        else:
                            np.testing.assert_array_equal(bb, cbcb)
                else:
                    if cb is None:
                        self.assertIs(b, None)
                    else:
                        np.testing.assert_array_equal(b, cb)

    def test_image_dataset_pnggray_int_noproc_smaller(self):
        dataset = spn.ImageDataset(image_files=self.data_path("img_dir1/*-{*}.png"),
                                   format=spn.ImageFormat.INT,
                                   num_epochs=3, batch_size=2, shuffle=False,
                                   ratio=1, crop=0,
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
                                   ratio=1, crop=0,
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
                                   ratio=1, crop=0,
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

    def test_image_dataset_pngrgb_int_noproc_smaller(self):
        dataset = spn.ImageDataset(image_files=self.data_path("img_dir2/*-{*}.png"),
                                   format=spn.ImageFormat.INT,
                                   num_epochs=1, batch_size=2, shuffle=False,
                                   ratio=1, crop=0,
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


if __name__ == '__main__':
    unittest.main()
