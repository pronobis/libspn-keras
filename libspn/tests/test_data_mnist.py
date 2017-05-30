#!/usr/bin/env python3

# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

import tensorflow as tf
import numpy as np
from context import libspn as spn


class TestMnistDataset(tf.test.TestCase):

    def test_mnist_load(self):
        """Loading MNIST dataset"""
        dataset_train = spn.MnistDataset(subset=spn.MnistDataset.Subset.TRAIN,
                                         format=spn.ImageFormat.INT,
                                         num_epochs=1, batch_size=100, shuffle=False,
                                         ratio=1, crop=0, num_threads=1,
                                         allow_smaller_final_batch=True,
                                         classes=None)
        dataset_test = spn.MnistDataset(subset=spn.MnistDataset.Subset.TEST,
                                        format=spn.ImageFormat.INT,
                                        num_epochs=1, batch_size=100, shuffle=False,
                                        ratio=1, crop=0, num_threads=1,
                                        allow_smaller_final_batch=True,
                                        classes=None)
        dataset_all = spn.MnistDataset(subset=spn.MnistDataset.Subset.ALL,
                                       format=spn.ImageFormat.INT,
                                       num_epochs=1, batch_size=100, shuffle=False,
                                       ratio=1, crop=0, num_threads=1,
                                       allow_smaller_final_batch=True,
                                       classes=None)

        dataset_train.load_data()
        self.assertEqual(dataset_train.samples.shape, (60000, 784))
        self.assertIs(dataset_train.samples.dtype.type, np.uint8)
        self.assertEqual(dataset_train.labels.shape, (60000, 1))
        self.assertIs(dataset_train.labels.dtype.type, np.dtype(np.int).type)

        dataset_test.load_data()
        self.assertEqual(dataset_test.samples.shape, (10000, 784))
        self.assertIs(dataset_test.samples.dtype.type, np.uint8)
        self.assertEqual(dataset_test.labels.shape, (10000, 1))
        self.assertIs(dataset_test.labels.dtype.type, np.dtype(np.int).type)

        dataset_all.load_data()
        self.assertEqual(dataset_all.samples.shape, (70000, 784))
        self.assertIs(dataset_all.samples.dtype.type, np.uint8)
        self.assertEqual(dataset_all.labels.shape, (70000, 1))
        self.assertIs(dataset_all.labels.dtype.type, np.dtype(np.int).type)

    def test_mnist_classes(self):
        dataset = spn.MnistDataset(subset=spn.MnistDataset.Subset.TEST,
                                   format=spn.ImageFormat.INT,
                                   num_epochs=1, batch_size=100, shuffle=False,
                                   ratio=1, crop=0, num_threads=1,
                                   allow_smaller_final_batch=True,
                                   classes=[1, 3, 5])
        dataset.load_data()
        self.assertEqual(dataset.samples.shape, (3037, 784))
        self.assertIs(dataset.samples.dtype.type, np.uint8)
        self.assertEqual(dataset.labels.shape, (3037, 1))
        self.assertIs(dataset.labels.dtype.type, np.dtype(np.int).type)
        self.assertEqual(set(dataset.labels.flatten()), {1, 3, 5})

    def generic_dataset_test(self, dataset):
        data = dataset.get_data()
        # Check if size of the sample is set
        self.assertIsNotNone(data[0].shape[1].value)
        # Check values
        with spn.session() as (sess, run):
            # while run():  # Getting 1 image only
            out = sess.run(data)
            return out

    def test_int_noproc(self):
        dataset = spn.MnistDataset(subset=spn.MnistDataset.Subset.TEST,
                                   format=spn.ImageFormat.INT,
                                   num_epochs=1, batch_size=1, shuffle=False,
                                   ratio=1, crop=0, num_threads=1,
                                   allow_smaller_final_batch=True,
                                   classes=None)
        img, label = self.generic_dataset_test(dataset)


if __name__ == '__main__':
    tf.test.main()
