# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

from libspn.data.dataset import Dataset
from libspn.data.image import ImageDatasetBase, ImageShape, ImageFormat
from libspn import utils
import tensorflow as tf
import os


class CIFAR10Dataset(ImageDatasetBase):
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
        min_after_dequeue (int): Min number of elements in the data queue after
                                 each dequeue. This is the minimum number of
                                 elements from which the shuffled batch will be
                                 drawn. Relevant only if ``shuffle`` is ``True``.
        num_threads (int): Number of threads enqueuing the data queue. If
                           larger than ``1``, the performance will be better,
                           but examples might not be in order even if
                           ``shuffle`` is ``False``. If ``shuffle`` is ``True``,
                           this might lead to examples repeating in the same
                           batch.
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
                 shuffle, ratio=1, crop=0, min_after_dequeue=None,
                 num_threads=1, allow_smaller_final_batch=False,
                 classes=None, seed=None):
        super().__init__(os.path.join(data_dir, 'data_batch*.bin'),
                         orig_height=32, orig_width=32, orig_num_channels=3,
                         format=format, num_epochs=num_epochs,
                         batch_size=batch_size, shuffle=shuffle,
                         shuffle_batch=shuffle,
                         min_after_dequeue=min_after_dequeue,
                         num_threads=num_threads,
                         allow_smaller_final_batch=allow_smaller_final_batch,
                         classes=classes, seed=seed)
        if classes is not None:
            if any(i not in CIFAR10Dataset.labels for i in classes):
                raise ValueError('classes must be one of %s' %
                                 CIFAR10Dataset.labels)

    @utils.docinherit(Dataset)
    def generate_data(self):
        pass
