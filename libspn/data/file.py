# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

import tensorflow as tf
from libspn.data.dataset import Dataset
from libspn import utils
import os
import glob
import fnmatch
import re


class FileDataset(Dataset):
    """
    An abstract class for a dataset stored in a file.

    Args:
        files (str or list of str): A string containing a path to a file or a
              glob matching multiple files, or a list of paths to multiple
              files. When glob is used, the files will be sorted, unless
              ``shuffle`` is set to ``True``. If a part of a path is wrapped in
              curly braces, it will be extracted as a label for the file. This
              works even for a glob, e.g. ``dir/{*}.jpg`` will use the filename
              without the extension as the label.
        num_epochs (int): Number of epochs of produced data.
        batch_size (int): Size of a single batch.
        shuffle (bool): Shuffle data within each epoch.
        shuffle_batch (bool): Shuffle data when generating batches.
        min_after_dequeue (int): Min number of elements in the data queue after
                                 each dequeue. This is the minimum number of
                                 elements from which the shuffled batch will
                                 be drawn. Relevant only if ``shuffle_batch``
                                 is ``True``.
        num_threads (int): Number of threads enqueuing the data queue. If
                           larger than ``1``, the performance will be better,
                           but examples might not be in order even if
                           ``shuffle_batch`` is ``False``. If ``shuffle_batch``
                           is ``True``, this might lead to examples repeating in
                           the same batch.
        allow_smaller_final_batch(bool): If ``False``, the last batch will be
                                         omitted if it has less elements than
                                         ``batch_size``.
        seed (int): Optional. Seed used when shuffling.
    """

    def __init__(self, files, num_epochs, batch_size, shuffle,
                 shuffle_batch, min_after_dequeue=None, num_threads=1,
                 allow_smaller_final_batch=False, seed=None):
        if not isinstance(files, str) and not isinstance(files, list):
            raise ValueError("file_name is neither a string or a list of strings")
        super().__init__(num_epochs=num_epochs, batch_size=batch_size,
                         shuffle=shuffle, shuffle_batch=shuffle_batch,
                         min_after_dequeue=min_after_dequeue,
                         num_threads=num_threads,
                         allow_smaller_final_batch=allow_smaller_final_batch,
                         seed=seed)
        self.__files = files

    def _get_file_label_tensor(self):
        """
        Serve filenames and labels for multiple epochs as a pair of tensors.
        Internally, this is implemented using a single queue, from which both a
        filename and a label are dequeued. The filename can be fed to a function
        that reads the file, but it cannot be used with a `Reader` which
        requires a queue.

        Returns:
            tuple: A 2-element tuple containing data taken from the queue:

            - filename (Tensor): A tensor containing the filename.
            - label (Tensor): A tensor containing the label.
        """
        files, labels = FileDataset._get_files_labels(self.__files)
        # Since files is a variable holding all the files, and since
        # input_producer shuffles all input data before passing it to
        # the queue, all files will always be shuffled independently
        # of the capacity parameter of string_input_producer
        return tf.train.slice_input_producer([files, labels],
                                             num_epochs=self._num_epochs,
                                             shuffle=self._shuffle,
                                             seed=self._seed)

    def _get_file_queue(self):
        """
        Build a queue serving file names for multiple epochs. This method can be
        used in combination with a `Reader` to read multiple data samples from
        each file. In such case, the labels are assumed to be stored inside the
        file and extracting them is the responsibility of the reader.

        Returns:
            FIFOQueue: A queue serving file names.
        """
        files, _ = FileDataset._get_files_labels(self.__files)
        # Since fnames is a variable holding all the files, and since
        # input_producer shuffles all input data before passing it to
        # the queue, all files will always be shuffled independently
        # of the capacity parameter of string_input_producer
        return tf.train.string_input_producer(files,
                                              num_epochs=self._num_epochs,
                                              shuffle=self._shuffle,
                                              seed=self._seed)

    @staticmethod
    def get_files_labels(files):
        """
        Convert the file specification to a list of files and labels. The files
        can be specified using a string containing a path to a file or a glob
        matching multiple files, or a list of paths to multiple files. When glob
        is used, the files will be returned sorted. If a part of a path is
        wrapped in curly braces, it will be extracted as a label for the file.
        This works even for a glob, e.g. ``dir/{*}.jpg`` will use the filename
        without the extension as the label.

        Args:
            files (str or list of str): File specification.

        Returns:
            tuple: A 2-element tuple containing:

            - files (list): A list of file paths matching the specification.
            - labels (list): A list of labels discovered for the files. If
              label has not been indicated for a file, the list contains an
              empty string for that file.
        """
        # Convert single path to a list
        if isinstance(files, str):
            files = [files]
        all_files = []
        all_labels = []
        for f in files:
            # Get files matching a glob
            f_clean = f.replace('{', '').replace('}', '')
            f_files = sorted(glob.glob(os.path.expanduser(f_clean)))
            # Get regexp for extracting labels
            f_re = fnmatch.translate(f)
            f_re = f_re.replace('\{', '(?P<label>').replace('\}', ')')
            f_re = re.compile(f_re)
            # Extract labels
            f_labels = [None] * len(f_files)
            for i, j in enumerate(f_files):
                try:
                    f_labels[i] = re.search(f_re, j).group('label')
                except IndexError:
                    f_labels[i] = ''
            # Append to all
            all_files += f_files
            all_labels += f_labels

        return all_files, all_labels


class CSVFileDataset(FileDataset):
    """
    A dataset read from a CSV file. The file can contain labels, which will be
    returned in a separate tensor. The labels should be stored in the first
    ``num_labels`` columns of the CSV file.

    If ``num_labels>0``, the data is returned as a tuple of tensors ``(labels,
    samples)``, where ``labels`` is a tensor of shape ``[batch_size,
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
            return [tf.stack(data[0:self._num_labels]),
                    tf.stack(data[self._num_labels:])]
        else:
            return [tf.stack(data)]
