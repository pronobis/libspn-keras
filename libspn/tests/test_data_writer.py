#!/usr/bin/env python3

# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

import os
import tensorflow as tf
import numpy as np
from context import libspn as spn


class TestDataWriter(tf.test.TestCase):

    @classmethod
    def setUpClass(cls):
        super(TestDataWriter, cls).setUpClass()
        cls.data_dir = os.path.realpath(os.path.join(os.getcwd(),
                                                     os.path.dirname(__file__),
                                                     "data"))

    @staticmethod
    def data_path(p):
        if isinstance(p, list):
            return [os.path.join(TestDataWriter.data_dir, i) for i in p]
        else:
            return os.path.join(TestDataWriter.data_dir, p)

    def test_write_all_single_tensor(self):
        path = self.data_path("out_test_write_all_single_tensor.csv")

        # Read&write
        dataset = spn.CSVFileDataset(self.data_path("data_int1.csv"),
                                     num_epochs=2,
                                     batch_size=4,
                                     shuffle=False,
                                     num_labels=0,
                                     defaults=[[101], [102], [103], [104], [105]],
                                     min_after_dequeue=1000,
                                     num_threads=1,
                                     allow_smaller_final_batch=True)
        writer = spn.CSVDataWriter(path)
        data1 = dataset.read_all()
        dataset.write_all(writer)

        # Read again
        dataset = spn.CSVFileDataset(path,
                                     num_epochs=1,
                                     batch_size=4,
                                     shuffle=False,
                                     num_labels=0,
                                     defaults=[[201], [202], [203], [204], [205]],
                                     min_after_dequeue=1000,
                                     num_threads=1,
                                     allow_smaller_final_batch=True)
        data2 = dataset.read_all()

        # Compare
        np.testing.assert_array_equal(data1, data2)

    def test_write_all_tensor_list(self):
        path = self.data_path("out_test_write_all_tensor_list.csv")

        # Read&write
        dataset = spn.CSVFileDataset(self.data_path("data_int1.csv"),
                                     num_epochs=2,
                                     batch_size=4,
                                     shuffle=False,
                                     num_labels=2,
                                     defaults=[[101], [102], [103.0],
                                               [104.0], [105.0]],
                                     min_after_dequeue=1000,
                                     num_threads=1,
                                     allow_smaller_final_batch=True)
        writer = spn.CSVDataWriter(path)
        data1 = dataset.read_all()
        dataset.write_all(writer)

        # Read again
        dataset = spn.CSVFileDataset(path,
                                     num_epochs=1,
                                     batch_size=4,
                                     shuffle=False,
                                     num_labels=2,
                                     defaults=[[201], [202], [203.0],
                                               [204.0], [205.0]],
                                     min_after_dequeue=1000,
                                     num_threads=1,
                                     allow_smaller_final_batch=True)
        data2 = dataset.read_all()

        # Compare
        np.testing.assert_array_equal(data1[0], data2[0])
        np.testing.assert_array_almost_equal(data1[1], data2[1])

    def test_csv_data_writer(self):
        # Write
        path = self.data_path("out_test_csv_data_writer.csv")
        writer = spn.CSVDataWriter(path)

        arr1 = np.array([1, 2, 3, 4])
        arr2 = np.array([[1 / 1, 1 / 2],
                         [1 / 3, 1 / 4],
                         [1 / 5, 1 / 6],
                         [1 / 7, 1 / 8]])
        writer.write(arr1, arr2)
        writer.write(arr1, arr2)

        # Read
        dataset = spn.CSVFileDataset(path,
                                     num_epochs=1,
                                     batch_size=10,
                                     shuffle=False,
                                     num_labels=1,
                                     defaults=[[1], [1.0], [1.0]],
                                     min_after_dequeue=1000,
                                     num_threads=1,
                                     allow_smaller_final_batch=True)
        data = dataset.read_all()

        # Compare
        np.testing.assert_array_equal(np.concatenate((arr1, arr1)),
                                      data[0].flatten())
        np.testing.assert_array_almost_equal(np.concatenate((arr2, arr2)),
                                             data[1])


if __name__ == '__main__':
    tf.test.main()
