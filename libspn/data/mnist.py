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
from libspn.log import get_logger
import numpy as np
import tensorflow as tf
import os
from enum import Enum
import scipy
import gzip


def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


class MNISTDataset(Dataset):
    """A dataset providing MNIST data with various types of processing applied.

    The data is returned as a tuple of tensors ``(samples, labels)``, where
    ``samples`` has shape ``[batch_size, width*height]`` and contains
    flattened image data, and ``labels`` has shape ``[batch_size, 1]`` and
    contains integer labels representing the digits in the images.

    Args:
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
        classes (list of int): Optional. If specified, only the listed classes
                               will be provided.
        seed (int): Optional. Seed used when shuffling.
    """

    __logger = get_logger()
    __info = __logger.info
    __debug1 = __logger.debug1

    class Subset(Enum):
        """Specifies what data is provided."""

        ALL = 0
        """Provide all data as one dataset combined of training and test samples."""

        TRAIN = 1
        """Provide only training samples"""

        TEST = 2
        """Provide only test samples."""

    def __init__(self, subset, format, num_epochs, batch_size,
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
        if subset not in MNISTDataset.Subset:
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
        if classes is not None:
            if not isinstance(classes, list):
                raise ValueError("classes must be a list")
            try:
                classes = [int(c) for c in classes]
            except ValueError:
                raise ValueError('classes must be convertible to int')
            if not all(i >= 0 and i <= 9 for i in classes):
                raise ValueError("elements of classes must be digits in the "
                                 "interval [0, 9]")
            if len(set(classes)) != len(classes):
                raise ValueError('classes must contain unique elements')
        self._classes = classes
        self._samples = None
        self._labels = None
        self._num_channels = 1
        # Get data dir
        self._data_dir = os.path.realpath(os.path.join(
            os.getcwd(), os.path.dirname(__file__),
            os.pardir, os.pardir, 'data', 'mnist'))

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
        return ImageShape(self._height, self._width, self._num_channels)

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
        if (self._subset == MNISTDataset.Subset.ALL or
                self._subset == MNISTDataset.Subset.TRAIN):
            self.__info("Loading MNIST training data")
            train_x = self._load_images('train-images-idx3-ubyte.gz')
            train_y = self._load_labels('train-labels-idx1-ubyte.gz')

        if (self._subset == MNISTDataset.Subset.ALL or
                self._subset == MNISTDataset.Subset.TEST):
            self.__info("Loading MNIST test data")
            test_x = self._load_images('t10k-images-idx3-ubyte.gz')
            test_y = self._load_labels('t10k-labels-idx1-ubyte.gz')

        # Collect
        if self._subset == MNISTDataset.Subset.TRAIN:
            samples = train_x
            labels = train_y
        elif self._subset == MNISTDataset.Subset.TEST:
            for i in range(test_x.shape[0]):
                test_x
            samples = test_x
            labels = test_y
        elif self._subset == MNISTDataset.Subset.ALL:
            samples = np.concatenate([train_x, test_x])
            labels = np.concatenate([train_y, test_y])

        # Filter classes
        if self._classes is None:
            self._labels = labels
        else:
            self.__debug1("Selecting classes %s" % self._classes)
            chosen = np.in1d(labels, self._classes)
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

    def _load_images(self, filename):
        """Extract MNIST images from a file. Stolen from TensorFlow.

        Args:
            filename (str): Filename of the labels file to load.

        Returns:
            array: A 3D uint8 numpy array [num, height, width].

        Raises:
            ValueError: If the bytestream does not start with 2051.
        """
        with gzip.GzipFile(os.path.join(self._data_dir, filename)) as bytestream:
            magic = _read32(bytestream)
            if magic != 2051:
                raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                                 (magic, filename))
            num_images = _read32(bytestream)
            rows = _read32(bytestream)
            cols = _read32(bytestream)
            buf = bytestream.read(rows * cols * num_images)
            data = np.frombuffer(buf, dtype=np.uint8)
            data = data.reshape(num_images, rows, cols)
            return data

    def _load_labels(self, filename):
        """Extract MNIST labels from a file. Stolen from TensorFlow.

        Args:
            filename (str): Filename of the labels file to load.

        Returns:
            array: a 2D int numpy array [num, 1].

        Raises:
            ValueError: If the bystream doesn't start with 2049.
        """
        with gzip.GzipFile(os.path.join(self._data_dir, filename)) as bytestream:
            magic = _read32(bytestream)
            if magic != 2049:
                raise ValueError('Invalid magic number %d in MNIST label file: %s' %
                                 (magic, filename))
            num_items = _read32(bytestream)
            buf = bytestream.read(num_items)
            labels = np.frombuffer(buf, dtype=np.uint8)
            return labels.reshape((-1, 1)).astype(np.int)
