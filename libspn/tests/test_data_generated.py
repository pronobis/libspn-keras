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


class TestDataGenerated(TestCase):

    def test_gaussian_mixture_dataset_without_final(self):
        """Batch generation (without smaller final batch) for
        GaussianMixtureDataset"""
        # Tests: - normalization of weights,
        #        - components with and without labels
        # Note: shuffling is NOT tested
        components = [
            spn.GaussianMixtureDataset.Component(0.301, [1, 1], [[1, 0],
                                                                 [0, 1]]),
            spn.GaussianMixtureDataset.Component(0.2, [2, 2], [[1, 0],
                                                               [0, 1]], 10),
            spn.GaussianMixtureDataset.Component(0.2, [1, 1], [[1, 0],
                                                               [0, 1]])]
        dataset = spn.GaussianMixtureDataset(components=components,
                                             num_samples=100,
                                             num_epochs=2,
                                             batch_size=90,
                                             shuffle=False,
                                             num_threads=1,
                                             allow_smaller_final_batch=False)
        # Get batches
        data = dataset.get_data()
        batches = []
        with spn.session() as (sess, run):
            while run():
                out = sess.run(data)
                batches.append(out)

        # Num of batches
        self.assertEqual(len(batches), 2)

        # Batch size = 90
        batch1 = batches[0]
        batch2 = batches[1]
        self.assertTupleEqual(batch1[0].shape, (90, 2))  # samples
        self.assertTupleEqual(batch2[0].shape, (90, 2))
        self.assertTupleEqual(batch1[1].shape, (90, 1))  # labels
        self.assertTupleEqual(batch2[1].shape, (90, 1))
        self.assertTupleEqual(batch1[2].shape, (90, ))  # likelihoods
        self.assertTupleEqual(batch2[2].shape, (90, ))

        # Data type
        self.assertTrue(np.issubdtype(batch1[0].dtype, np.floating))
        self.assertTrue(np.issubdtype(batch2[0].dtype, np.floating))
        self.assertTrue(np.issubdtype(batch1[1].dtype, np.integer))
        self.assertTrue(np.issubdtype(batch2[1].dtype, np.integer))
        self.assertTrue(np.issubdtype(batch1[2].dtype, np.floating))
        self.assertTrue(np.issubdtype(batch2[2].dtype, np.floating))

        # Are the overlapping parts of the two batches identical?
        np.testing.assert_array_equal(batch1[0][:80], batch2[0][10:])
        np.testing.assert_array_equal(batch1[1][:80], batch2[1][10:])
        np.testing.assert_array_equal(batch1[2][:80], batch2[2][10:])

        # Number of samples per component
        epoch_labels = np.concatenate([batch1[1], batch2[1][:10]])
        self.assertEqual((epoch_labels == 0).sum(), 43)
        self.assertEqual((epoch_labels == 10).sum(), 28)
        self.assertEqual((epoch_labels == 2).sum(), 29)   # Roundoff error

    def test_gaussian_mixture_dataset_with_final(self):
        """Batch generation (with smaller final batch) for
        GaussianMixtureDataset"""
        # Tests: - normalization of weights,
        #        - components with and without labels
        # Note: shuffling is NOT tested
        components = [
            spn.GaussianMixtureDataset.Component(0.301, [1, 1], [[1, 0],
                                                                 [0, 1]]),
            spn.GaussianMixtureDataset.Component(0.2, [2, 2], [[1, 0],
                                                               [0, 1]], 10),
            spn.GaussianMixtureDataset.Component(0.2, [1, 1], [[1, 0],
                                                               [0, 1]])]
        dataset = spn.GaussianMixtureDataset(components=components,
                                             num_samples=100,
                                             num_epochs=2,
                                             batch_size=90,
                                             shuffle=False,
                                             num_threads=1,
                                             allow_smaller_final_batch=True)
        # Get batches
        data = dataset.get_data()
        batches = []
        with spn.session() as (sess, run):
            while run():
                out = sess.run(data)
                batches.append(out)

        # Num of batches
        self.assertEqual(len(batches), 3)

        # Batch size = 90
        batch1 = batches[0]
        batch2 = batches[1]
        batch3 = batches[2]
        self.assertTupleEqual(batch1[0].shape, (90, 2))  # samples
        self.assertTupleEqual(batch2[0].shape, (90, 2))
        self.assertTupleEqual(batch3[0].shape, (20, 2))
        self.assertTupleEqual(batch1[1].shape, (90, 1))  # labels
        self.assertTupleEqual(batch2[1].shape, (90, 1))
        self.assertTupleEqual(batch3[1].shape, (20, 1))
        self.assertTupleEqual(batch1[2].shape, (90, ))  # likelihoods
        self.assertTupleEqual(batch2[2].shape, (90, ))
        self.assertTupleEqual(batch3[2].shape, (20, ))

        # Data type
        self.assertTrue(np.issubdtype(batch1[0].dtype, np.floating))
        self.assertTrue(np.issubdtype(batch2[0].dtype, np.floating))
        self.assertTrue(np.issubdtype(batch3[0].dtype, np.floating))
        self.assertTrue(np.issubdtype(batch1[1].dtype, np.integer))
        self.assertTrue(np.issubdtype(batch2[1].dtype, np.integer))
        self.assertTrue(np.issubdtype(batch3[1].dtype, np.integer))
        self.assertTrue(np.issubdtype(batch1[2].dtype, np.floating))
        self.assertTrue(np.issubdtype(batch2[2].dtype, np.floating))
        self.assertTrue(np.issubdtype(batch3[2].dtype, np.floating))

        # Are the overlapping parts of the batches identical?
        np.testing.assert_array_equal(batch1[0][:80], batch2[0][10:])
        np.testing.assert_array_equal(np.concatenate([batch1[0][80:],
                                                      batch2[0][:10]]),
                                      batch3[0])
        np.testing.assert_array_equal(batch1[1][:80], batch2[1][10:])
        np.testing.assert_array_equal(np.concatenate([batch1[1][80:],
                                                      batch2[1][:10]]),
                                      batch3[1])
        np.testing.assert_array_equal(batch1[2][:80], batch2[2][10:])
        np.testing.assert_array_equal(np.concatenate([batch1[2][80:],
                                                      batch2[2][:10]]),
                                      batch3[2])

        # Number of samples per component
        epoch_labels = np.concatenate([batch1[1], batch2[1][:10]])
        self.assertEqual((epoch_labels == 0).sum(), 43)
        self.assertEqual((epoch_labels == 10).sum(), 28)
        self.assertEqual((epoch_labels == 2).sum(), 29)  # Roundoff error

    def test_discrete_gaussian_mixture_dataset_with_final(self):
        """Batch generation (with smaller final batch) for
        GaussianMixtureDataset with digitization"""
        # Tests: - normalization of weights,
        #        - components with and without labels
        # Note: shuffling is NOT tested
        components = [
            spn.GaussianMixtureDataset.Component(0.301, [1, 1], [[1, 0],
                                                                 [0, 1]]),
            spn.GaussianMixtureDataset.Component(0.2, [2, 2], [[1, 0],
                                                               [0, 1]], 10),
            spn.GaussianMixtureDataset.Component(0.2, [1, 1], [[1, 0],
                                                               [0, 1]])]
        dataset = spn.GaussianMixtureDataset(components=components,
                                             num_samples=100,
                                             num_epochs=2,
                                             batch_size=90,
                                             shuffle=False,
                                             num_threads=1,
                                             allow_smaller_final_batch=True,
                                             num_vals=10)
        # Get batches
        data = dataset.get_data()
        batches = []
        with spn.session() as (sess, run):
            while run():
                out = sess.run(data)
                batches.append(out)

        # Num of batches
        self.assertEqual(len(batches), 3)

        # Batch size = 90
        batch1 = batches[0]
        batch2 = batches[1]
        batch3 = batches[2]
        self.assertTupleEqual(batch1[0].shape, (90, 2))  # samples
        self.assertTupleEqual(batch2[0].shape, (90, 2))
        self.assertTupleEqual(batch3[0].shape, (20, 2))
        self.assertTupleEqual(batch1[1].shape, (90, 1))  # labels
        self.assertTupleEqual(batch2[1].shape, (90, 1))
        self.assertTupleEqual(batch3[1].shape, (20, 1))
        self.assertTupleEqual(batch1[2].shape, (90, ))  # likelihoods
        self.assertTupleEqual(batch2[2].shape, (90, ))
        self.assertTupleEqual(batch3[2].shape, (20, ))

        # Data type
        self.assertTrue(np.issubdtype(batch1[0].dtype, np.integer))
        self.assertTrue(np.issubdtype(batch2[0].dtype, np.integer))
        self.assertTrue(np.issubdtype(batch3[0].dtype, np.integer))
        self.assertTrue(np.issubdtype(batch1[1].dtype, np.integer))
        self.assertTrue(np.issubdtype(batch2[1].dtype, np.integer))
        self.assertTrue(np.issubdtype(batch3[1].dtype, np.integer))
        self.assertTrue(np.issubdtype(batch1[2].dtype, np.floating))
        self.assertTrue(np.issubdtype(batch2[2].dtype, np.floating))
        self.assertTrue(np.issubdtype(batch3[2].dtype, np.floating))

        # Are the overlapping parts of the batches identical?
        np.testing.assert_array_equal(batch1[0][:80], batch2[0][10:])
        np.testing.assert_array_equal(np.concatenate([batch1[0][80:],
                                                      batch2[0][:10]]),
                                      batch3[0])
        np.testing.assert_array_equal(batch1[1][:80], batch2[1][10:])
        np.testing.assert_array_equal(np.concatenate([batch1[1][80:],
                                                      batch2[1][:10]]),
                                      batch3[1])
        np.testing.assert_array_equal(batch1[2][:80], batch2[2][10:])
        np.testing.assert_array_equal(np.concatenate([batch1[2][80:],
                                                      batch2[2][:10]]),
                                      batch3[2])

        # Number of samples per component
        epoch_labels = np.concatenate([batch1[1], batch2[1][:10]])
        self.assertEqual((epoch_labels == 0).sum(), 43)
        self.assertEqual((epoch_labels == 10).sum(), 28)
        self.assertEqual((epoch_labels == 2).sum(), 29)  # Roundoff error

        # Are values within range?
        epoch_samples = np.concatenate([batch1[0], batch2[0][:10]])
        self.assertEqual(epoch_samples.min(), 0)
        self.assertEqual(epoch_samples.max(), 9)

    def test_read_all_int_grid_dataset(self):
        dataset = spn.IntGridDataset(num_dims=2,
                                     num_vals=3,
                                     num_epochs=2,
                                     batch_size=4,
                                     shuffle=False,
                                     num_threads=1,
                                     allow_smaller_final_batch=True)
        data = dataset.read_all()

        np.testing.assert_array_equal(data, np.array([[0, 0],
                                                      [0, 1],
                                                      [0, 2],
                                                      [1, 0],
                                                      [1, 1],
                                                      [1, 2],
                                                      [2, 0],
                                                      [2, 1],
                                                      [2, 2],
                                                      [0, 0],
                                                      [0, 1],
                                                      [0, 2],
                                                      [1, 0],
                                                      [1, 1],
                                                      [1, 2],
                                                      [2, 0],
                                                      [2, 1],
                                                      [2, 2]], dtype=np.int32))


if __name__ == '__main__':
    tf.test.main()
