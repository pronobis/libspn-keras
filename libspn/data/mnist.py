# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------


from libspn.data.dataset import Dataset
from libspn.data.image import ImageShape, ImageFormat
from libspn import conf
from libspn import utils
import numpy as np
import tensorflow as tf
import os
from enum import Enum
import scipy


class MnistDataset(Dataset):
    """A dataset providing MNIST data with various types of processing applied.

    The data is returned as a tuple of tensors ``(samples, labels)``, where
    ``samples`` has shape ``[batch_size, width*height]`` and contains
    flattened image data, and ``labels`` has shape ``[batch_size, 1]`` and
    contains integer labels representing the digits in the images.

    Args:
        data_dir (str): Path to the folder containing the MNIST dataset.
        subset (Subset): Determines what data to provide.
        format (ImageFormat): Image format.
        num_epochs (int): Number of epochs of produced data.
        batch_size (int): Size of a single batch.
        shuffle (bool): Shuffle data within each epoch.
        ratio (int): Downsample by the given ratio.
        crop (int): Crop that many border pixels (after downsampling).
        num_threads (int): Number of threads enqueuing the data queue. If
                           larger than ``1``, the performance will be better,
                           but examples might not be in order even if
                           ``shuffle`` is ``False``.
        allow_smaller_final_batch(bool): If ``False``, the last batch will be
                                         omitted if it has less elements than
                                         ``batch_size``.
        classes (set of int): Optional. If specified, only the listed classes
                               will be provided.
        seed (int): Optional. Seed used when shuffling.
    """

    class Subset(Enum):
        """Specifies what data is provided."""

        ALL = 0
        """Provide all data as one dataset combined of training and test samples."""

        TRAIN = 1
        """Provide only training samples"""

        TEST = 2
        """Provide only test samples."""

    def __init__(self, data_dir, subset, format, num_epochs, batch_size,
                 shuffle, ratio=1, crop=0, num_threads=1,
                 allow_smaller_final_batch=False, classes=None, seed=None):
        super().__init__(num_epochs=num_epochs, batch_size=batch_size,
                         shuffle=shuffle,
                         # We shuffle the samples in this class
                         # so batch shuffling is not needed
                         shuffle_batch=False, min_after_dequeue=None,
                         num_threads=num_threads,
                         allow_smaller_final_batch=allow_smaller_final_batch,
                         seed=seed)
        self._orig_width = 28
        self._orig_height = 28
        if not isinstance(data_dir, str):
            raise ValueError("data_dir must be a string")
        data_dir = os.path.expanduser(data_dir)
        if not os.path.isdir(data_dir):
            raise RuntimeError("data_dir must point to an existing directory")
        self._data_dir = data_dir
        if subset not in MnistDataset.Subset:
            raise ValueError("Incorrect subset: %s" % subset)
        self._subset = subset
        if format not in {ImageFormat.FLOAT, ImageFormat.INT, ImageFormat.BINARY}:
            raise ValueError("Incorrect format: %s, "
                             "only FLOAT, INT and BINARY are accepted" % format)
        self._format = format
        if not isinstance(ratio, int):
            raise ValueError("ratio must be an integer")
        if ratio not in {1, 2, 4}:
            raise ValueError("ratio must be one of {1, 2, 4}")
        self._ratio = ratio
        self._width = self._orig_width // ratio
        self._height = self._orig_height // ratio
        if not isinstance(crop, int):
            raise ValueError("crop must be an integer")
        if crop < 0 or crop > (self._width // 2) or crop > (self._height // 2):
            raise ValueError("invalid value of crop")
        self._crop = crop
        self._width -= 2 * crop
        self._height -= 2 * crop
        if classes is not None and not isinstance(classes, set):
            raise ValueError("classes must be a set")
        if (classes is not None and
                not all(isinstance(i, int) and i >= 0 and i <= 9 for i in classes)):
            raise ValueError("Elements of classes must be integers in "
                             "interval [0, 9]")
        self._classes = classes
        self._samples = None
        self._labels = None
        self._channels = 1

    @property
    def orig_height(self):
        """int: Height of the original images."""
        return self._orig_height

    @property
    def orig_width(self):
        """int: Width of the original images."""
        return self._orig_width

    @property
    def format(self):
        """Image format."""
        return self._format

    @property
    def ratio(self):
        """int: Original images are downsampled this number of times."""
        return self._ratio

    @property
    def crop(self):
        """int: That many border pixels are cropped."""
        return self._crop

    @property
    def classes(self):
        """list of int: List of classes provided by the dataset."""
        if self._classes is not None:
            return self._classes
        else:
            return list(range(10))

    @property
    def samples(self):
        """array: Array of all data samples."""
        return self._samples

    @property
    def labels(self):
        """array: Array of all data labels."""
        return self._labels

    @property
    def shape(self):
        """Shape of the image data samples."""
        return ImageShape(self._height, self._width, self._channels)

    @utils.docinherit(Dataset)
    def generate_data(self):
        self.load_data()
        # Add input producer that serves the loaded samples
        # All data is shuffled independently of the capacity parameter
        producer = tf.train.slice_input_producer([self._samples, self._labels],
                                                 num_epochs=self._num_epochs,
                                                 # Shuffle data
                                                 shuffle=self._shuffle,
                                                 seed=self._seed)
        return producer

    @utils.docinherit(Dataset)
    def process_data(self, data):
        # Everything is processed before entering the producer
        return data

    def load_data(self):
        """Load all data from MNIST data files."""
        # Load data
        if (self._subset == MnistDataset.Subset.ALL or
                self._subset == MnistDataset.Subset.TRAIN):
            loaded = np.fromfile(os.path.join(self._data_dir, 'train-images-idx3-ubyte'),
                                 dtype=np.uint8)
            train_x = loaded[16:].reshape(
                (60000, self._orig_height, self._orig_width))
            loaded = np.fromfile(os.path.join(self._data_dir, 'train-labels-idx1-ubyte'),
                                 dtype=np.uint8)
            train_y = loaded[8:].reshape((60000)).astype(np.int)

        if (self._subset == MnistDataset.Subset.ALL or
                self._subset == MnistDataset.Subset.TEST):
            loaded = np.fromfile(os.path.join(self._data_dir, 't10k-images-idx3-ubyte'),
                                 dtype=np.uint8)
            test_x = loaded[16:].reshape((
                10000, self._orig_height, self._orig_width))
            loaded = np.fromfile(os.path.join(self._data_dir, 't10k-labels-idx1-ubyte'),
                                 dtype=np.uint8)
            test_y = loaded[8:].reshape((10000)).astype(np.int)

        # Collect
        if self._subset == MnistDataset.Subset.TRAIN:
            samples = train_x
            labels = train_y
        elif self._subset == MnistDataset.Subset.TEST:
            for i in range(test_x.shape[0]):
                test_x
            samples = test_x
            labels = test_y
        elif self._subset == MnistDataset.Subset.ALL:
            samples = np.concatenate([train_x, test_x])
            labels = np.concatenate([train_y, test_y])

        # Filter classes
        if self._classes is not None:
            chosen = np.in1d(labels, list(self._classes))
            samples = samples[chosen]
            self._labels = labels[chosen]

        # Process data (input samples are HxW uint8, and well normalized (0-254/255))
        # - convert to float for accuracy, use float32, since that's what scipy wants
        samples = samples.astype(np.float32) / 255.0
        # - downsample
        if self._ratio > 1:
            num_samples = samples.shape[0]
            samples_resized = [None] * num_samples
            for i in range(num_samples):
                samples_resized[i] = scipy.misc.imresize(
                    samples[i], 1.0 / self._ratio,
                    # bicubic looks best after normalization, sharper than bilinear/lanczos
                    interp='bicubic',
                    mode='F')  # Operate on float32 images
            samples = np.array(samples_resized)
        # - crop (samples are float32 HxW)
        if self._crop > 0:
            samples = samples[:, self._crop:-self._crop, self._crop:-self._crop]
        # - flatten
        samples = samples.reshape(samples.shape[0], -1)
        # - normalize (resized image is likely not normalized)
        samples -= np.amin(samples, axis=1, keepdims=True)
        samples /= np.amax(samples, axis=1, keepdims=True)
        # - convert to format (samples are float32 [0, 1] flattened)
        if self._format == ImageFormat.FLOAT:
            # Already float [0,1]. Ensure the dtype is spn float dtype
            self._samples = samples.astype(conf.dtype.as_numpy_dtype())
        elif self._format == ImageFormat.INT:
            self._samples = np.rint(samples * 255.0).astype(np.uint8)
        elif self._format == ImageFormat.BINARY:
            self._samples = (samples > 0.5).astype(np.uint8)
