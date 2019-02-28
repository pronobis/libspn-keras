#!/usr/bin/env python3

from context import libspn as spn
from test import TestCase
import tensorflow as tf
import numpy as np


class TestCSVFileDataset(TestCase):

    def generic_dataset_test(self, dataset, correct_batches, tol=0.0):
        data = dataset.get_data()
        # Check if sample size is set
        if type(data) is tf.Tensor:
            self.assertIsNotNone(data.shape[1].value)
        else:
            for d in data:
                self.assertIsNotNone(d.shape[1].value)
        # Check values
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
                        if (np.issubdtype(bb.dtype, np.floating) or
                                np.issubdtype(bb.dtype, np.integer)):
                            np.testing.assert_allclose(bb, cbcb, atol=tol)
                        else:
                            np.testing.assert_equal(bb, cbcb)
            else:
                if cb is None:
                    self.assertIs(b, None)
                else:
                    self.assertEqual(b.dtype, cb.dtype)
                    if (np.issubdtype(b.dtype, np.floating) or
                            np.issubdtype(b.dtype, np.integer)):
                        np.testing.assert_allclose(b, cb, atol=tol)
                    else:
                        np.testing.assert_equal(b, cb)

    def test_unlabeled_csv_file_dataset_without_final_batch(self):
        """Batch generation (without smaller final batch) for CSV file
        without labels"""
        # Note: shuffling is NOT tested
        dataset = spn.CSVFileDataset(self.data_path(["data_int1.csv",
                                                     "data_int2.csv"]),
                                     num_vals=[255] * 5,
                                     defaults=[[101], [102], [103], [104], [105]],
                                     num_epochs=2,
                                     batch_size=3,
                                     shuffle=False,
                                     num_labels=0,
                                     min_after_dequeue=1000,
                                     num_threads=1,
                                     allow_smaller_final_batch=False)
        batches = [np.array([[1, 2, 3, 4, 5],
                             [6, 102, 8, 9, 10],
                             [11, 12, 103, 14, 15]], dtype=np.int32),
                   np.array([[16, 102, 18, 19, 20],
                             [21, 22, 103, 24, 25],
                             [26, 27, 28, 104, 30]], dtype=np.int32),
                   np.array([[31, 32, 33, 104, 35],
                             [36, 37, 38, 104, 40],
                             [41, 42, 43, 104, 45]], dtype=np.int32),
                   np.array([[46, 47, 48, 104, 50],
                             [1, 2, 3, 4, 5],
                             [6, 102, 8, 9, 10]], dtype=np.int32),
                   np.array([[11, 12, 103, 14, 15],
                             [16, 102, 18, 19, 20],
                             [21, 22, 103, 24, 25]], dtype=np.int32),
                   np.array([[26, 27, 28, 104, 30],
                             [31, 32, 33, 104, 35],
                             [36, 37, 38, 104, 40]], dtype=np.int32)]
        self.generic_dataset_test(dataset, batches)

    def test_unlabeled_csv_file_dataset_with_final_batch(self):
        """Batch generation (without smaller final batch) for CSV file
        with labels"""
        # Note: shuffling is NOT tested
        dataset = spn.CSVFileDataset(self.data_path(["data_int1.csv",
                                                     "data_int2.csv"]),
                                     num_vals=[255] * 5,
                                     defaults=[[101], [102], [103], [104], [105]],
                                     num_epochs=2,
                                     batch_size=3,
                                     shuffle=False,
                                     num_labels=0,
                                     min_after_dequeue=1000,
                                     num_threads=1,
                                     allow_smaller_final_batch=True)
        batches = [np.array([[1, 2, 3, 4, 5],
                             [6, 102, 8, 9, 10],
                             [11, 12, 103, 14, 15]], dtype=np.int32),
                   np.array([[16, 102, 18, 19, 20],
                             [21, 22, 103, 24, 25],
                             [26, 27, 28, 104, 30]], dtype=np.int32),
                   np.array([[31, 32, 33, 104, 35],
                             [36, 37, 38, 104, 40],
                             [41, 42, 43, 104, 45]], dtype=np.int32),
                   np.array([[46, 47, 48, 104, 50],
                             [1, 2, 3, 4, 5],
                             [6, 102, 8, 9, 10]], dtype=np.int32),
                   np.array([[11, 12, 103, 14, 15],
                             [16, 102, 18, 19, 20],
                             [21, 22, 103, 24, 25]], dtype=np.int32),
                   np.array([[26, 27, 28, 104, 30],
                             [31, 32, 33, 104, 35],
                             [36, 37, 38, 104, 40]], dtype=np.int32),
                   np.array([[41, 42, 43, 104, 45],
                             [46, 47, 48, 104, 50]], dtype=np.int32)]
        self.generic_dataset_test(dataset, batches)

    def test_labeled_csv_file_dataset_int(self):
        """Batch generation for CSV file with integer data and 2 labels"""
        # Note: shuffling is NOT tested
        dataset = spn.CSVFileDataset(self.data_path(["data_int1.csv",
                                                     "data_int2.csv"]),
                                     num_vals=[255] * 3,
                                     defaults=[[101], [102], [103], [104], [105]],
                                     num_epochs=2,
                                     batch_size=3,
                                     shuffle=False,
                                     num_labels=2,
                                     min_after_dequeue=1000,
                                     num_threads=1,
                                     allow_smaller_final_batch=True)
        batches = [[np.array([[1, 2],
                              [6, 102],
                              [11, 12]], dtype=np.int32),
                    np.array([[3, 4, 5],
                              [8, 9, 10],
                              [103, 14, 15]], dtype=np.int32)],
                   [np.array([[16, 102],
                              [21, 22],
                              [26, 27]], dtype=np.int32),
                    np.array([[18, 19, 20],
                              [103, 24, 25],
                              [28, 104, 30]], dtype=np.int32)],
                   [np.array([[31, 32],
                              [36, 37],
                              [41, 42]], dtype=np.int32),
                    np.array([[33, 104, 35],
                              [38, 104, 40],
                              [43, 104, 45]], dtype=np.int32)],
                   [np.array([[46, 47],
                              [1, 2],
                              [6, 102]], dtype=np.int32),
                    np.array([[48, 104, 50],
                              [3, 4, 5],
                              [8, 9, 10]], dtype=np.int32)],
                   [np.array([[11, 12],
                              [16, 102],
                              [21, 22]], dtype=np.int32),
                    np.array([[103, 14, 15],
                              [18, 19, 20],
                              [103, 24, 25]], dtype=np.int32)],
                   [np.array([[26, 27],
                              [31, 32],
                              [36, 37]], dtype=np.int32),
                    np.array([[28, 104, 30],
                              [33, 104, 35],
                              [38, 104, 40]], dtype=np.int32)],
                   [np.array([[41, 42],
                              [46, 47]], dtype=np.int32),
                    np.array([[43, 104, 45],
                              [48, 104, 50]], dtype=np.int32)]]
        # Since we changed the order of data in CSVFileDataset,
        # we also change the order in batches
        for b in batches:
            b[1], b[0] = b[0], b[1]

        self.generic_dataset_test(dataset, batches)

    def test_labeled_csv_file_dataset_int_onelabel(self):
        """Batch generation for CSV file with integer data and 1 label"""
        # Note: shuffling is NOT tested
        dataset = spn.CSVFileDataset(self.data_path(["data_int1.csv",
                                                     "data_int2.csv"]),
                                     num_vals=[255] * 4,
                                     defaults=[[101], [102], [103], [104], [105]],
                                     num_epochs=2,
                                     batch_size=3,
                                     shuffle=False,
                                     num_labels=1,
                                     min_after_dequeue=1000,
                                     num_threads=1,
                                     allow_smaller_final_batch=True)
        batches = [[np.array([[1],
                              [6],
                              [11]], dtype=np.int32),
                    np.array([[2, 3, 4, 5],
                              [102, 8, 9, 10],
                              [12, 103, 14, 15]], dtype=np.int32)],
                   [np.array([[16],
                              [21],
                              [26]], dtype=np.int32),
                    np.array([[102, 18, 19, 20],
                              [22, 103, 24, 25],
                              [27, 28, 104, 30]], dtype=np.int32)],
                   [np.array([[31],
                              [36],
                              [41]], dtype=np.int32),
                    np.array([[32, 33, 104, 35],
                              [37, 38, 104, 40],
                              [42, 43, 104, 45]], dtype=np.int32)],
                   [np.array([[46],
                              [1],
                              [6]], dtype=np.int32),
                    np.array([[47, 48, 104, 50],
                              [2, 3, 4, 5],
                              [102, 8, 9, 10]], dtype=np.int32)],
                   [np.array([[11],
                              [16],
                              [21]], dtype=np.int32),
                    np.array([[12, 103, 14, 15],
                              [102, 18, 19, 20],
                              [22, 103, 24, 25]], dtype=np.int32)],
                   [np.array([[26],
                              [31],
                              [36]], dtype=np.int32),
                    np.array([[27, 28, 104, 30],
                              [32, 33, 104, 35],
                              [37, 38, 104, 40]], dtype=np.int32)],
                   [np.array([[41],
                              [46]], dtype=np.int32),
                    np.array([[42, 43, 104, 45],
                              [47, 48, 104, 50]], dtype=np.int32)]]
        # Since we changed the order of data in CSVFileDataset,
        # we also change the order in batches
        for b in batches:
            b[1], b[0] = b[0], b[1]

        self.generic_dataset_test(dataset, batches)

    def test_labeled_csv_file_dataset_float(self):
        """Batch generation for CSV file with float data and 2 labels"""
        # Note: shuffling is NOT tested
        dataset = spn.CSVFileDataset(self.data_path("data_mix.csv"),
                                     num_vals=[None] * 3,
                                     defaults=[[101.0], [102.0], [103.0],
                                               [104.0], [105.0]],
                                     num_epochs=2,
                                     batch_size=3,
                                     shuffle=False,
                                     num_labels=2,
                                     min_after_dequeue=1000,
                                     num_threads=1,
                                     allow_smaller_final_batch=True)
        batches = [[np.array([[1., 2.],
                              [6., 102.],
                              [11., 12.]], dtype=np.float32),
                    np.array([[3., 4., 5.],
                              [8., 104., 10.],
                              [103., 104., 15.]], dtype=np.float32)],
                   [np.array([[16., 102.],
                              [21., 22.],
                              [1., 2.]], dtype=np.float32),
                    np.array([[18., 19., 20.],
                              [103., 24., 25.],
                              [3., 4., 5.]], dtype=np.float32)],
                   [np.array([[6., 102.],
                              [11., 12.],
                              [16., 102.]], dtype=np.float32),
                    np.array([[8., 104., 10.],
                              [103., 104., 15.],
                              [18., 19., 20.]], dtype=np.float32)],
                   [np.array([[21., 22.]], dtype=np.float32),
                    np.array([[103., 24., 25.]], dtype=np.float32)]]
        # Since we changed the order of data in CSVFileDataset,
        # we also change the order in batches
        for b in batches:
            b[1], b[0] = b[0], b[1]

        self.generic_dataset_test(dataset, batches)

    def test_custom_csv_file_dataset(self):
        """Batch generation for CSV file with custom data"""
        class CustomCSVFileDataset(spn.CSVFileDataset):
            """Our custom dataset."""

            def process_data(self, data):
                return [tf.stack(data[0:1]), tf.stack(data[1:3]), tf.stack(data[3:])]

        # Note: shuffling is NOT tested
        dataset = CustomCSVFileDataset(self.data_path("data_mix.csv"),
                                       num_vals=[255, None, None],
                                       defaults=[[101.0], [102], [103],
                                                 [104.0], [105.0]],
                                       num_epochs=2,
                                       batch_size=3,
                                       shuffle=False,
                                       num_labels=2,
                                       min_after_dequeue=1000,
                                       num_threads=1,
                                       allow_smaller_final_batch=True)
        batches = [[np.array([[1.], [6.], [11.]], dtype=np.float32),
                    np.array([[2, 3],
                              [102, 8],
                              [12, 103]], dtype=np.int32),
                    np.array([[4., 5.],
                              [104., 10.],
                              [104., 15.]], dtype=np.float32)],
                   [np.array([[16.], [21.], [1.]], dtype=np.float32),
                    np.array([[102, 18],
                              [22, 103],
                              [2, 3]], dtype=np.int32),
                    np.array([[19., 20.],
                              [24., 25.],
                              [4., 5.]], dtype=np.float32)],
                   [np.array([[6.], [11.], [16.]], dtype=np.float32),
                    np.array([[102, 8],
                              [12, 103],
                              [102, 18]], dtype=np.int32),
                    np.array([[104., 10.],
                              [104., 15.],
                              [19., 20.]], dtype=np.float32)],
                   [np.array([[21.]], dtype=np.float32),
                    np.array([[22, 103]], dtype=np.int32),
                    np.array([[24., 25.]], dtype=np.float32)]]
        self.generic_dataset_test(dataset, batches)

    def test_read_all_labeled_csv_file_dataset(self):
        """Test read_all for CSV file with 2 labels."""
        dataset = spn.CSVFileDataset(self.data_path(["data_int1.csv",
                                                     "data_int2.csv"]),
                                     num_vals=[255] * 3,
                                     defaults=[[101], [102], [103], [104], [105]],
                                     num_epochs=2,
                                     batch_size=3,
                                     shuffle=False,
                                     num_labels=2,
                                     min_after_dequeue=1000,
                                     num_threads=1,
                                     allow_smaller_final_batch=True)
        data = dataset.read_all()
        self.assertEqual(len(data), 2)
        np.testing.assert_array_equal(data[0],
                                      np.array([[3, 4, 5],
                                                [8, 9, 10],
                                                [103, 14, 15],
                                                [18, 19, 20],
                                                [103, 24, 25],
                                                [28, 104, 30],
                                                [33, 104, 35],
                                                [38, 104, 40],
                                                [43, 104, 45],
                                                [48, 104, 50],
                                                [3, 4, 5],
                                                [8, 9, 10],
                                                [103, 14, 15],
                                                [18, 19, 20],
                                                [103, 24, 25],
                                                [28, 104, 30],
                                                [33, 104, 35],
                                                [38, 104, 40],
                                                [43, 104, 45],
                                                [48, 104, 50]], dtype=np.int32))
        np.testing.assert_array_equal(data[1],
                                      np.array([[1, 2],
                                                [6, 102],
                                                [11, 12],
                                                [16, 102],
                                                [21, 22],
                                                [26, 27],
                                                [31, 32],
                                                [36, 37],
                                                [41, 42],
                                                [46, 47],
                                                [1, 2],
                                                [6, 102],
                                                [11, 12],
                                                [16, 102],
                                                [21, 22],
                                                [26, 27],
                                                [31, 32],
                                                [36, 37],
                                                [41, 42],
                                                [46, 47]], dtype=np.int32))


if __name__ == '__main__':
    tf.test.main()
