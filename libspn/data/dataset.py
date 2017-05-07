# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

from abc import ABC, abstractmethod
import tensorflow as tf
import numpy as np
from libspn.session import session


class Dataset(ABC):
    """An abstract class defining the interface of a dataset.

    Args:
        num_epochs (int): Number of training epochs for which data is produced.
        batch_size (int): Size of a single batch.
        shuffle (bool): Shuffle data within an epoch.
        num_threads (int): Number of threads enqueuing the example queue. If
                           larger than ``1``, the performance will be better,
                           but examples might not be in order even if ``shuffle``
                           is ``False``. If ``shuffle`` is ``True``, this might
                           lead to examples repeating in the same batch.
        min_after_dequeue (int): Min number of elements in the queue after each
                                 dequeue. This is the minimum number of elements
                                 from which the shuffled batch will be drawn.
                                 Relevant only if ``shuffle`` is ``True``.
        allow_smaller_final_batch(bool): If ``False``, the last batch will be
                                         omitted if it has less elements than
                                         ``batch_size``.
    """

    def __init__(self, num_epochs, batch_size, shuffle, min_after_dequeue=1000,
                 num_threads=1, allow_smaller_final_batch=False):
        self._num_epochs = num_epochs
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._name_scope = None
        self.__min_after_dequeue = min_after_dequeue
        self.__num_threads = num_threads
        self.__allow_smaller_final_batch = allow_smaller_final_batch

    def get_data(self):
        """Get an operation obtaining batches of data from the dataset.

        Returns:
            A tensor, a list of tensors or a dictionary of tensors with the
            batch data.
        """
        with tf.name_scope("Dataset") as self._name_scope:
            raw_data = self.generate_data()
            proc_data = self.process_data(raw_data)
            return self.batch_data(proc_data)

    @abstractmethod
    def generate_data(self):
        """Assemble a TF operation generating the next data sample.

        Returns:
            A tensor, a list of tensors or a dictionary of tensors with a single
            data sample.
        """
        pass

    @abstractmethod
    def process_data(self, data):
        """Assemble a TF operation processing a data sample.

        Args:
            data: A tensor, a list of tensors or a dictionary of tensors with
                  a single data sample.

        Returns:
            A tensor, a list of tensors or a dictionary of tensors with a single
            data sample.
        """
        pass

    def batch_data(self, data):
        """Assemble a TF operation producing batches of data samples.

        Args:
            data: A tensor, a list of tensors or a dictionary of tensors with
                  a single data sample.

        Returns:
            A tensor, a list of tensors or a dictionary of tensors with a
            batch of data.
        """
        if self._shuffle:
            batch = tf.train.shuffle_batch(
                data, batch_size=self._batch_size,
                num_threads=self.__num_threads,
                capacity=(self.__min_after_dequeue +
                          (self.__num_threads + 1) * self._batch_size),
                min_after_dequeue=self.__min_after_dequeue,
                allow_smaller_final_batch=self.__allow_smaller_final_batch)
        else:
            batch = tf.train.batch(
                data, batch_size=self._batch_size,
                num_threads=self.__num_threads,
                capacity=(self.__num_threads + 1) * self._batch_size,
                allow_smaller_final_batch=self.__allow_smaller_final_batch)
        return batch

    def read_all(self):
        """Read all data (all batches and epochs) from the dataset into numpy
        arrays.

        Returns:
            An array, a list of arrays or a dictionary of arrays with all the
            data in the dataset.
        """
        # Read all batches in internal graph
        batches = []
        with tf.Graph().as_default():
            data = self.get_data()
            with session() as (sess, run):
                while run():
                    out = sess.run(data)
                    batches.append(out)
        # Concatenate
        if isinstance(batches[0], list):
            return [np.concatenate([b[key] for b in batches])
                    for key in range(len(batches[0]))]
        elif isinstance(batches[0], dict):
            return {key: np.concatenate([b[key] for b in batches])
                    for key in batches[0].keys()}
        else:
            return np.concatenate(batches)

    def write_all(self, writer):
        """Write all data (all batches and epochs) from the dataset using the
        given writer. Each batch is written using a separate ``write()`` call on
        the writer. Therefore, even dataset that do not fit in memory can be
        written this way.

        Args:
            writer (DataWriter): The data writer to use.
        """
        with tf.Graph().as_default():
            data = self.get_data()
            with session() as (sess, run):
                while run():
                    out = sess.run(data)
                    writer.write(out)
