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
from libspn.log import get_logger


class Dataset(ABC):
    """An abstract class defining the interface of a dataset.

    Args:
        num_vars (int): Number of variables in each data sample.
        num_vals (int or list of int): Number of values of each variable. Can be
            a single value or a list of values for each of ``num_vars``. Use
            ``None``, to indicate that a variable is continuous, in the range
            ``[0, 1]``.
        num_labels (int): Number of labels for each data sample.
        num_epochs (int): Number of epochs of produced data.
        batch_size (int): Size of a single batch.
        shuffle (bool): Shuffle data within each epoch.
        shuffle_batch (bool): Shuffle data when generating batches.
        min_after_dequeue (int): Min number of elements in the data queue after
                                 each dequeue. This is the minimum number of
                                 elements from which the shuffled batch will
                                 be drawn. Relevant only and must be set if
                                 ``shuffle_batch`` is ``True``.
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

    __logger = get_logger()
    __info = __logger.info

    def __init__(self, num_vars, num_vals, num_labels,
                 num_epochs, batch_size, shuffle, shuffle_batch,
                 min_after_dequeue=None, num_threads=1,
                 allow_smaller_final_batch=False, seed=None):
        if not isinstance(num_vars, int) or num_vars < 1:
            raise ValueError("num_vars must be a positive integer")
        self._num_vars = num_vars
        if isinstance(num_vals, list):
            if len(num_vals) != num_vars:
                raise ValueError("num_vals must have num_vars elements")
            if any((i is not None) and (not isinstance(i, int) or i < 1)
                   for i in num_vals):
                raise ValueError("num_vals values must be a positive integers or None")
            self._num_vals = num_vals
        else:
            if ((num_vals is not None) and (not isinstance(num_vals, int) or
                                            num_vals < 1)):
                raise ValueError("num_vals must be a positive integer or None")
            self._num_vals = [num_vals] * num_vars
        if not isinstance(num_labels, int) or num_labels < 0:
            raise ValueError("num_labels must be an integer >= 0")
        self._num_labels = num_labels
        if not isinstance(num_epochs, int) or num_epochs < 1:
            raise ValueError("num_epochs must be a positive integer")
        self._num_epochs = num_epochs
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self._batch_size = batch_size
        if shuffle_batch and not shuffle:
            raise RuntimeError("Batch shuffling should not be enabled "
                               "when shuffle is False.")
        if shuffle_batch and min_after_dequeue is None:
            raise RuntimeError("min_after_dequeue must be set if batch "
                               "shuffling is enabled.")
        self._shuffle = shuffle
        self._shuffle_batch = shuffle_batch
        self._min_after_dequeue = min_after_dequeue
        self._num_threads = num_threads
        self._allow_smaller_final_batch = allow_smaller_final_batch
        if seed is not None and (not isinstance(seed, int) or seed < 1):
            raise ValueError("seed must be None or a positive integer")
        self._seed = seed
        self._name_scope = None

    @property
    def num_vars(self):
        """int: Number of variables in each data sample."""
        return self._num_vars

    @property
    def num_vals(self):
        """list of int: Number of values of each variable.

        Value ``None`` indicates that a variable is continuous, in the range
        ``[0, 1]``.
        """
        return self._num_vals

    @property
    def num_labels(self):
        """int: Number of labels for each data sample."""
        return self._num_labels

    @property
    def num_epochs(self):
        """int: Number of epochs of produced data."""
        return self._num_epochs

    @property
    def batch_size(self):
        """int: Size of a single batch."""
        return self._batch_size

    @property
    def shuffle(self):
        """bool: ``True`` if provided data is shuffled."""
        return self._shuffle

    @property
    def seed(self):
        """int: Seed used when shuffling."""
        return self._seed

    @property
    def allow_smaller_final_batch(self):
        """bool: If ``False``, the last batch is omitted if it has less
        elements than ``batch_size``."""
        return self._allow_smaller_final_batch

    def get_data(self):
        """Get an operation obtaining batches of data from the dataset.

        Returns:
            A tensor or a list of tensors with the batch data.
        """
        self.__info("Building dataset operations")
        with tf.name_scope("Dataset") as self._name_scope:
            raw_data = self.generate_data()
            proc_data = self.process_data(raw_data)
            return self.batch_data(proc_data)

    @abstractmethod
    def generate_data(self):
        """Assemble a TF operation generating the next data sample.

        Returns:
            A list of tensors with a single data sample.
        """
        pass

    @abstractmethod
    def process_data(self, data):
        """Assemble a TF operation processing a data sample.

        Args:
            data: A list of tensors with a single data sample.

        Returns:
            A list of tensors with a single data sample.
        """
        pass

    def batch_data(self, data):
        """Assemble a TF operation producing batches of data samples.

        Args:
            data: A list of tensors or a dictionary of tensors with
                  a single data sample. If the list of tensors contains
                  only one element, this function returns a tensor.
                  Otherwise, it returns a list of dictionary of tensors.

        Returns:
            A tensor, a list of tensors or a dictionary of tensors with a
            batch of data.
        """
        if self._shuffle_batch:
            # If len(data) is 1, batch will be a tensor
            # If len(data) > 0, batch will be a list of tensors
            batch = tf.train.shuffle_batch(
                data, batch_size=self._batch_size,
                num_threads=self._num_threads,
                seed=self._seed,
                capacity=(self._min_after_dequeue +
                          (self._num_threads + 1) * self._batch_size),
                min_after_dequeue=self._min_after_dequeue,
                allow_smaller_final_batch=self._allow_smaller_final_batch)
        else:
            # If len(data) is 1, batch will be a tensor
            # If len(data) > 0, batch will be a list of tensors
            batch = tf.train.batch(
                data, batch_size=self._batch_size,
                num_threads=self._num_threads,
                capacity=(self._num_threads + 1) * self._batch_size,
                allow_smaller_final_batch=self._allow_smaller_final_batch)
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
        self.__info("Writing all data from %s to %s" %
                    (type(self).__name__, type(writer).__name__))
        with tf.Graph().as_default():
            data = self.get_data()
            with session() as (sess, run):
                i = 0
                while run():
                    out = sess.run(data)
                    i += 1
                    self.__info("Writing batch %d" % i)
                    if not isinstance(out, list):  # Convert to list
                        out = [out]
                    writer.write(*out)
