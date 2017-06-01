# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

from libspn.data.dataset import Dataset
from libspn.data.image import ImageShape, ImageFormat


class CIFAR10Dataset(Dataset):
    """A dataset providing CIFAR-10 data with various types of processing applied.

    The data is returned as a tuple of tensors ``(samples, labels)``, where
    ``samples`` has shape ``[batch_size, width*height*num_channels]`` and
    contains flattened image data, and ``labels`` has shape ``[batch_size, 1]``
    and contains image labels.

    Args:
        data_dir (str): Path to the directory with the data.
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

    labels = {'airplane',
              'automobile',
              'bird',
              'cat',
              'deer',
              'dog',
              'frog',
              'horse',
              'ship',
              'truck'}

    def __init__(self, data_dir, format, num_epochs, batch_size,
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
        self._data_dir = data_dir
        self._orig_width = 32
        self._orig_height = 32
        self._orig_channels = 3
        self._format = format
        self._num_channels = format.num_channels
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
            if len(set(classes)) != len(classes):
                raise ValueError('classes must contain unique elements')
            if any(i not in CIFAR10Dataset.labels for i in classes):
                raise ValueError('classes must be one of %s' %
                                 CIFAR10Dataset.labels)
        self._classes = classes
        self._samples = None
        self._labels = None

    @property
    def orig_height(self):
        """int: Height of the original images."""
        return self._orig_height

    @property
    def orig_width(self):
        """int: Width of the original images."""
        return self._orig_width

    @property
    def orig_num_channels(self):
        """int: Number of channels of the original images."""
        return self._orig_num_channels

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
    def shape(self):
        """Shape of the image data samples."""
        return ImageShape(self._height, self._width, self._num_channels)
