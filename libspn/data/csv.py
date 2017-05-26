# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

import tensorflow as tf
from libspn.data.dataset import Dataset
from libspn.data.file import FileDataset
from libspn import utils


class CSVFileDataset(FileDataset):
    """
    A dataset read from a CSV file. The file can contain labels, which will be
    returned in a separate tensor. The labels should be stored in the first
    ``num_labels`` columns of the CSV file.

    If ``num_labels>0``, the data is returned as a tuple of tensors ``(samples,
    labels)``, where ``labels`` is a tensor of shape ``[batch_size,
    num_labels]``, containing the first ``num_labels`` columns and ``samples``
    is a tensor ``[batch_size, ?]`` containing the data samples. If
    ``num_labels==0``, the data is returned as a single tensor ``samples``.

    This dataset can be overridden to customize the way the data is processed
    grouped and cast. For instance, to divide the batch into three tensors, with
    different dtypes in different columns, define custom dataset::

        class CustomCSVFileDataset(spn.CSVFileDataset):

            def process_data(self, data):
                return [data[0], tf.stack(data[1:3]), tf.stack(data[3:])]

    and then, give defaults of different type::

        dataset = CustomCSVFileDataset(...,
                                       defaults=[[1.0], [1], [1], [1.0], [1.0]])

    Args:
        files (str or list): A string containing a path to a file or a glob
                             matching multiple files, or a list of paths to
                             multiple files. When glob is used, the files will
                             be sorted, unless ``shuffle`` is set to ``True``.
        num_epochs (int): Number of epochs of produced data.
        batch_size (int): Size of a single batch.
        shuffle (bool): Shuffle data within each epoch.
        num_labels (int): The number of columns considered labels. If set to
                          ``0``, no labels are returned.
        defaults (list of Tensor): A list of tensors, one tensor per column of
                                   the input record, with a default value for
                                   that column.
        min_after_dequeue (int): Min number of elements in the data queue after
                                 each dequeue. This is the minimum number of
                                 elements from which the shuffled batch will
                                 be drawn. Relevant only if ``shuffle``
                                 is ``True``.
        num_threads (int): Number of threads enqueuing the data queue. If
                           larger than ``1``, the performance will be better,
                           but examples might not be in order even if
                           ``shuffle`` is ``False``. If ``shuffle`` is ``True``,
                           this might lead to examples repeating in the same
                           batch.
        allow_smaller_final_batch(bool): If ``False``, the last batch will be
                                         omitted if it has less elements than
                                         ``batch_size``.
        seed (int): Optional. Seed used when shuffling.
    """

    def __init__(self, files, num_epochs, batch_size, shuffle,
                 num_labels=0, defaults=None,
                 min_after_dequeue=None, num_threads=1,
                 allow_smaller_final_batch=False, seed=None):
        super().__init__(files=files, num_epochs=num_epochs, batch_size=batch_size,
                         shuffle=shuffle, shuffle_batch=shuffle,
                         min_after_dequeue=min_after_dequeue,
                         num_threads=num_threads,
                         allow_smaller_final_batch=allow_smaller_final_batch,
                         seed=seed)
        self._num_labels = num_labels
        self._defaults = defaults

    @utils.docinherit(Dataset)
    def generate_data(self):
        file_queue = self._get_file_queue()
        reader = tf.TextLineReader()
        key, value = reader.read(file_queue)
        return tf.decode_csv(value, record_defaults=self._defaults)

    @utils.docinherit(Dataset)
    def process_data(self, data):
        if self._num_labels > 0:
            return [tf.stack(data[self._num_labels:]),
                    tf.stack(data[0:self._num_labels])]
        else:
            return [tf.stack(data)]
