import tensorflow as tf
from libspn.data.dataset import Dataset
import os
import glob
import fnmatch
import re


class FileDataset(Dataset):
    """An abstract class for a dataset stored in a file.

    Args:
        files (str or list of str): A string containing a path to a file or a
              glob matching multiple files, or a list of paths to multiple
              files. When glob is used, the files will be sorted, unless
              ``shuffle`` is set to ``True``. If a part of a path is wrapped in
              curly braces, it will be extracted as a label for the file. This
              works even for a glob, e.g. ``dir/{*}.jpg`` will use the filename
              without the extension as the label. For files without a label
              specification, the returned label is an empty string.
        num_vars (int): Number of variables in each data sample.
        num_vals (int or list of int): Number of values of each variable. Can be
            a single value or a list of values, one for each of ``num_vars``
            variables. Use ``None``, to indicate that a variable is continuous,
            in the range ``[0, 1]``.
        num_labels (int): Number of labels for each data sample.
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
        classes (list of int): Optional. If specified, only files with labels
                               listed here will be provided.
        seed (int): Optional. Seed used when shuffling.
    """

    def __init__(self, files, num_vars, num_vals, num_labels,
                 num_epochs, batch_size, shuffle,
                 shuffle_batch, min_after_dequeue=None, num_threads=1,
                 allow_smaller_final_batch=False, classes=None, seed=None):
        if not isinstance(files, str) and not isinstance(files, list):
            raise ValueError("file_name is neither a string nor a list of strings")
        super().__init__(num_vars=num_vars, num_vals=num_vals,
                         num_labels=num_labels,
                         num_epochs=num_epochs, batch_size=batch_size,
                         shuffle=shuffle, shuffle_batch=shuffle_batch,
                         min_after_dequeue=min_after_dequeue,
                         num_threads=num_threads,
                         allow_smaller_final_batch=allow_smaller_final_batch,
                         seed=seed)
        self._files = files
        if classes is not None:
            if not isinstance(classes, list):
                raise ValueError("classes must be a list")
            try:
                classes = [str(c) for c in classes]
            except ValueError:
                raise ValueError('classes must be convertible to string')
            if len(set(classes)) != len(classes):
                raise ValueError('classes must contain unique elements')
        self._classes = classes

    @property
    def classes(self):
        """list of str: List of labels of classes provided by the dataset.

        If `None`, all classes are provided."""
        return self._classes

    def _get_file_label_tensors(self):
        """Serve filenames and labels for multiple epochs as a pair of tensors.

        Internally, this is implemented using a single queue, from which both a
        filename and a label are dequeued. The filename can be fed to a function
        that reads the file, but it cannot be used with a `Reader` which
        requires a queue.

        Returns:
            tuple: A 2-element tuple containing data taken from the queue:

            - filename (Tensor): A tensor containing the filename.
            - label (Tensor): A tensor containing the label.
        """
        files, labels = self._get_files_labels(self._files, self._classes)
        # Since files is a variable holding all the files, and since
        # input_producer shuffles all input data before passing it to
        # the queue, all files will always be shuffled independently
        # of the capacity parameter of string_input_producer
        return tf.train.slice_input_producer([files, labels],
                                             num_epochs=self._num_epochs,
                                             shuffle=self._shuffle,
                                             seed=self._seed)

    def _get_file_queue(self):
        """Build a queue serving file names for multiple epochs.

        This method can be used in combination with a `Reader` to read multiple
        data samples from each file. In such case, the labels are assumed to be
        stored inside the file and extracting them is the responsibility of the
        reader.

        Returns:
            FIFOQueue: A queue serving file names.
        """
        files, _ = self._get_files_labels(self._files, self._classes)
        # Since fnames is a variable holding all the files, and since
        # input_producer shuffles all input data before passing it to
        # the queue, all files will always be shuffled independently
        # of the capacity parameter of string_input_producer
        return tf.train.string_input_producer(files,
                                              num_epochs=self._num_epochs,
                                              shuffle=self._shuffle,
                                              seed=self._seed)

    @staticmethod
    def _get_files_labels(files, classes=None):
        """Convert the file specification to a list of files and labels.

        The files can be specified using a string containing a path to a file or
        a glob matching multiple files, or a list of paths to multiple files.
        When glob is used, the files will be returned sorted. If a part of a
        path is wrapped in curly braces, it will be extracted as a label for the
        file. This works even for a glob, e.g. ``dir/{*}.jpg`` will use the
        filename without the extension as the label. If no label is specified
        for a file, the returned label is an empty string.

        Args:
            files (str or list of str): File specification.
            classes (list of int): Optional. If specified, only files with labels
                                   listed here will be provided.

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
            f = os.path.expanduser(f)
            # Get files matching a glob
            f_clean = f.replace('{', '').replace('}', '')
            f_files = sorted(glob.glob(f_clean))
            # Get regexp for extracting labels
            f_re = fnmatch.translate(f)
            f_re = f_re.replace('\{', '(?P<label>').replace('\}', ')')
            f_re = re.compile(f_re)
            # Extract labels
            for i, j in enumerate(f_files):
                try:
                    label = re.search(f_re, j).group('label')
                except IndexError:
                    label = ''
                if classes is None or label in classes:
                    all_files.append(j)
                    all_labels.append(label)

        return all_files, all_labels
