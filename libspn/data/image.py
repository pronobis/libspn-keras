# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

import tensorflow as tf
from libspn.data.file import FileDataset
from libspn import conf
from libspn import utils
from enum import Enum
from collections import namedtuple
import scipy


ImageShape = namedtuple("ImageShape", ["height", "width", "channels"])


class ImageFormat(Enum):
    """Specifies the format of the image data. The convention employed in
    TensorFlow is used, where float images use the interval [0, 1] and
    integer images use the interval [0, MAX_VAL]."""

    FLOAT = 0
    """Image data is 1 channel (intensity), of type spn.conf.dtype, with values
    in the interval [0, 1]."""

    INT = 1
    """Image data is 1 channel (intensity), of type uint8 with values in the
    interval [0, 255]."""

    BINARY = 2
    """Image data is 1 channel (intensity), of type uint8 with binary values
    {0,1}."""

    RGB_FLOAT = 3
    """Image data is 3 channels (rgb), of type spn.conf.dtype, with values in
    the interval [0, 1]."""

    RGB_INT = 4
    """Image data is 3 channels (rgb), of type uint8 with values in the
    interval [0, 255]."""

    RGB_BINARY = 5
    """Image data is 3 channels (rgb), of type uint8 with binary values {0,1}."""

    @property
    def num_vals(self):
        """int: Number of possible values for discretized formats.  Returns
        ``None`` for continuous values.
        """
        return (2 if self in {ImageFormat.BINARY or ImageFormat.RGB_BINARY}
                else 255 if self in {ImageFormat.INT or ImageFormat.RGB_INT}
                else None)


class ImageDataset(FileDataset):
    """A dataset serving images loaded from image files.

    The data is returned as a tuple of tensors ``(samples, labels)``, where
    ``samples`` has shape ``[batch_size, width*height*num_channels]`` and
    contains flattened image data, and ``labels`` has shape ``[batch_size, 1]``
    and contains image labels if the file specification specifies labels. For
    files without a label specification, the returned label is an empty string.

    Args:
        image_files (str or list of str): A string containing a path to a file
              or a glob matching multiple files, or a list of paths to multiple
              files. When glob is used, the files will be sorted, unless
              ``shuffle`` is set to ``True``. If a part of a path is wrapped in
              curly braces, it will be extracted as a label for the file. This
              works even for a glob, e.g. ``dir/{*}.jpg`` will use the filename
              without the extension as the label.
        format (ImageFormat): Image format.
        num_epochs (int): Number of epochs of produced data.
        batch_size (int): Size of a single batch.
        shuffle (bool): Shuffle data within each epoch.
        ratio (int): Downsample by the given ratio.
        crop (int): Crop that many border pixels (after downsampling).
        num_threads (int): Number of threads enqueuing the example queue. If
                           larger than ``1``, the performance will be better,
                           but examples might not be in order even if ``shuffle``
                           is ``False``.
        allow_smaller_final_batch(bool): If ``False``, the last batch will be
                                         omitted if it has less elements than
                                         ``batch_size``.
        seed (int): Optional. Seed used when shuffling.
    """

    def __init__(self, image_files, format, num_epochs, batch_size, shuffle,
                 ratio=1, crop=0, num_threads=1, allow_smaller_final_batch=False,
                 seed=None):
        super().__init__(image_files, num_epochs=num_epochs,
                         batch_size=batch_size, shuffle=shuffle,
                         # Since each image is in a separate file, and we
                         # shuffle all files, we do not need batch shuffling
                         shuffle_batch=False, min_after_dequeue=None,
                         num_threads=num_threads,
                         allow_smaller_final_batch=allow_smaller_final_batch,
                         seed=seed)
        self._guess_orig_shape(image_files)
        if format not in ImageFormat:
            raise ValueError("Incorrect format: %s" % format)
        self._format = format
        self._channels = (3 if format in {ImageFormat.RGB_FLOAT,
                                          ImageFormat.RGB_INT,
                                          ImageFormat.RGB_BINARY}
                          else 1)
        if not isinstance(ratio, int):
            raise ValueError("ratio must be an integer")
        if self._orig_width % ratio or self._orig_height % ratio:
            raise ValueError("orig_height '%s' and orig_width '%s' must be "
                             "dividable by ratio '%s'" % (
                                 self._orig_height, self._orig_width, ratio))
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
    def orig_channels(self):
        """int: Number of channels of the original images."""
        return self._orig_channels

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
        return ImageShape(self._height, self._width, self._channels)

    @utils.docinherit(FileDataset)
    def generate_data(self):
        file_queue = self._get_file_queue()
        reader = tf.WholeFileReader()
        fname, value = reader.read(file_queue)
        # Decode image
        image = tf.image.decode_image(value)
        # Since decode_jpeg does not set the image shape, we need to set it manually
        # https://github.com/tensorflow/tensorflow/issues/521
        # https://stackoverflow.com/questions/34746777/why-do-i-get-valueerror-image-must-be-fully-defined-when-transforming-im
        image.set_shape((self._orig_height, self._orig_width, self._orig_channels))
        # Parse label
        label = tf.string_split([fname], delimiter=os.sep)
        label = tf.sparse_tensor_to_dense(label, default_value='str')[0, -1]
        label = tf.string_split([label], delimiter='.')
        label = tf.sparse_tensor_to_dense(label, default_value='str')[0, 0]
        return image, label

    @utils.docinherit(FileDataset)
    def process_data(self, data):
        image = data[0]  # Type uint8, HxWxC
        label = data[1]
        # Convert to float [0, 1] for which all processing behaves consistently
        # This does not normalize the data
        image = tf.image.convert_image_dtype(image, dtype=conf.dtype)
        # Downsample
        if self._ratio > 1:
            image = tf.image.resize_images(image,
                                           size=(self._orig_height // self._ratio,
                                                 self._orig_width // self._ratio),
                                           # Area seems to be doing best for downsampling
                                           method=tf.image.ResizeMethod.AREA,
                                           align_corners=False)
        # Crop
        if self._crop > 0:
            image = tf.image.crop_to_bounding_box(image, self._crop, self._crop,
                                                  self._height, self._width)
        # Convert color spaces
        if (self._format in {ImageFormat.FLOAT, ImageFormat.INT,
                             ImageFormat.BINARY}
                and self._orig_channels > 1):
            image = tf.image.rgb_to_grayscale(image)
        elif (self._format in {ImageFormat.RGB_FLOAT, ImageFormat.RGB_INT,
                               ImageFormat.RGB_BINARY}
              and self._orig_channels == 1):
            image = tf.image.grayscale_to_rgb(image)
        # Normalize
        image -= tf.reduce_min(image)
        image /= tf.reduce_max(image)
        # Convert to format and normalize
        if self._format in {ImageFormat.FLOAT, ImageFormat.RGB_FLOAT}:
            pass  # Already 1 channel float
        elif self._format in {ImageFormat.INT, ImageFormat.RGB_INT}:
            image = tf.image.convert_image_dtype(image, dtype=tf.uint8)
        elif self._format in {ImageFormat.BINARY, ImageFormat.RGB_BINARY}:
            image = tf.cast(tf.greater(image, 0.5), dtype=tf.uint8)
        # Flatten
        image = tf.reshape(image, [-1])
        return [image, label]

    def _guess_orig_shape(self, image_files):
        """A trick to guess original image shape from one of the given images
        since TensorFlow does not set static image shape."""
        try:
            if isinstance(image_files, str):
                fname = glob.glob(os.path.expanduser(image_files))[0]
            else:
                fname = image_files[0]
        except IndexError:
            raise RuntimeError("Cannot guess original image shape since"
                               " a representative file is not found")
        img = scipy.misc.imread(fname)
        self._orig_height = img.shape[0]
        self._orig_width = img.shape[1]
        if len(img.shape) == 3:
            self._orig_channels = img.shape[2]
        else:
            self._orig_channels = 1
