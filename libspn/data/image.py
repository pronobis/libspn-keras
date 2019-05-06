import tensorflow as tf
from libspn.data.file import FileDataset
from libspn import conf
from libspn import utils
from enum import Enum
from collections import namedtuple
import scipy
import PIL


ImageShape = namedtuple("ImageShape", ["height", "width", "channels"])


class ImageFormat(Enum):
    """Specifies the format of the image data.

    The convention employed in TensorFlow is used, where float images use the
    interval ``[0, 1]`` and integer images use the interval ``[0, MAX_VAL]``.
    """

    FLOAT = 0
    """Image data is 1 channel (intensity), of type spn.conf.dtype, with values
    in the interval ``[0, 1]``."""

    INT = 1
    """Image data is 1 channel (intensity), of type uint8 with values in the
    interval ``[0, 255]``."""

    BINARY = 2
    """Image data is 1 channel (intensity), of type uint8 with binary values ``{0,1}``."""

    RGB_FLOAT = 3
    """Image data is 3 channels (rgb), of type spn.conf.dtype, with values in
    the interval ``[0, 1]``."""

    RGB_INT = 4
    """Image data is 3 channels (rgb), of type uint8 with values in the
    interval ``[0, 255]``."""

    RGB_BINARY = 5
    """Image data is 3 channels (rgb), of type uint8 with binary values ``{0,1}``."""

    @property
    def num_vals(self):
        """int: Number of possible values for discretized formats.

        Returns ``None`` for continuous values.
        """
        return (2 if self in {ImageFormat.BINARY, ImageFormat.RGB_BINARY}
                else 255 if self in {ImageFormat.INT, ImageFormat.RGB_INT}
                else None)

    @property
    def num_channels(self):
        """int: Number of channels for the format."""
        return (3 if self in {ImageFormat.RGB_FLOAT, ImageFormat.RGB_INT,
                              ImageFormat.RGB_BINARY}
                else 1)


class ImageDatasetBase(FileDataset):
    """Base class for image datasets served from files. The images are always
    normalized so that the values span the full range of values of the selected
    format.

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
        orig_height (int): Height of the original images.
        orig_width (int): Width of the original images.
        orig_num_channels (int): Number of channels of the original images.
        format (ImageFormat): Image format.
        num_labels (int): Number of labels for each image.
        num_epochs (int): Number of epochs of produced data.
        batch_size (int): Size of a single batch.
        shuffle (bool): Shuffle data within each epoch.
        shuffle_batch (bool): Shuffle data when generating batches.
        ratio (int): Downsample by the given ratio.
        crop (int): Crop that many border pixels (after downsampling).
        min_after_dequeue (int): Min number of elements in the data queue after
                                 each dequeue. This is the minimum number of
                                 elements from which the shuffled batch will
                                 be drawn. Relevant only if ``shuffle_batch``
                                 is ``True``.
        num_threads (int): Number of threads enqueuing the example queue. If
                           larger than ``1``, the performance will be better,
                           but examples might not be in order even if ``shuffle``
                           is ``False``.
        allow_smaller_final_batch(bool): If ``False``, the last batch will be
                                         omitted if it has less elements than
                                         ``batch_size``.
        classes (list of int): Optional. If specified, only images with labels
                               listed here will be provided.
        seed (int): Optional. Seed used when shuffling.
    """

    def __init__(self, image_files, orig_height, orig_width, orig_num_channels,
                 format, num_labels, num_epochs, batch_size, shuffle,
                 shuffle_batch, ratio=1, crop=0, min_after_dequeue=None,
                 num_threads=1, allow_smaller_final_batch=False, classes=None,
                 seed=None):
        if not isinstance(orig_height, int) or orig_height <= 0:
            raise ValueError("orig_height must be an integer > 0")
        self._orig_height = orig_height
        if not isinstance(orig_width, int) or orig_width <= 0:
            raise ValueError("orig_width must be an integer > 0")
        self._orig_width = orig_width
        if not isinstance(orig_num_channels, int) or orig_num_channels <= 0:
            raise ValueError("orig_num_channels must be an integer > 0")
        self._orig_num_channels = orig_num_channels
        if format not in ImageFormat:
            raise ValueError("Incorrect format: %s" % format)
        self._format = format
        self._num_channels = format.num_channels
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
        super().__init__(image_files,
                         num_vars=(self._height * self._width * self._num_channels),
                         num_vals=format.num_vals,
                         num_labels=num_labels,
                         num_epochs=num_epochs,
                         batch_size=batch_size,
                         shuffle=shuffle,
                         shuffle_batch=shuffle_batch,
                         min_after_dequeue=min_after_dequeue,
                         num_threads=num_threads,
                         allow_smaller_final_batch=allow_smaller_final_batch,
                         classes=classes,
                         seed=seed)

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
                and self._orig_num_channels > 1):
            image = tf.image.rgb_to_grayscale(image)
        elif (self._format in {ImageFormat.RGB_FLOAT, ImageFormat.RGB_INT,
                               ImageFormat.RGB_BINARY}
              and self._orig_num_channels == 1):
            image = tf.image.grayscale_to_rgb(image)
        # Normalize
        image -= tf.reduce_min(image)
        image /= tf.reduce_max(image)
        # Convert to format
        if self._format in {ImageFormat.FLOAT, ImageFormat.RGB_FLOAT}:
            pass  # Already 1 channel float
        elif self._format in {ImageFormat.INT, ImageFormat.RGB_INT}:
            image = tf.image.convert_image_dtype(image, dtype=tf.uint8)
        elif self._format in {ImageFormat.BINARY, ImageFormat.RGB_BINARY}:
            image = tf.cast(tf.greater(image, 0.5), dtype=tf.uint8)
        # Flatten
        image = tf.reshape(image, [-1])
        return [image, label]


class ImageDataset(ImageDatasetBase):
    """A dataset serving images loaded from image files. The images are always
    normalized so that the values span the full range of values of the selected
    format.

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
        accurate (bool): If `True`, uses more accurate but slower JPEG image
                         decompression.
        num_threads (int): Number of threads enqueuing the example queue. If
                           larger than ``1``, the performance will be better,
                           but examples might not be in order even if ``shuffle``
                           is ``False``.
        allow_smaller_final_batch(bool): If ``False``, the last batch will be
                                         omitted if it has less elements than
                                         ``batch_size``.
        classes (list of int): Optional. If specified, only images with labels
                               listed here will be provided.
        seed (int): Optional. Seed used when shuffling.
    """

    def __init__(self, image_files, format, num_epochs, batch_size, shuffle,
                 ratio=1, crop=0, accurate=False, num_threads=1,
                 allow_smaller_final_batch=False, classes=None, seed=None):
        oh, ow, onc = self._guess_orig_shape(image_files, classes)
        super().__init__(image_files, orig_height=oh, orig_width=ow,
                         orig_num_channels=onc, format=format,
                         num_labels=1,  # Each image has a label, may be empty
                         num_epochs=num_epochs,
                         batch_size=batch_size, shuffle=shuffle,
                         # Since each image is in a separate file, and we
                         # shuffle all files, we do not need batch shuffling
                         shuffle_batch=False, min_after_dequeue=None,
                         ratio=ratio, crop=crop,
                         num_threads=num_threads,
                         allow_smaller_final_batch=allow_smaller_final_batch,
                         classes=classes,
                         seed=seed)
        if not isinstance(accurate, bool):
            raise ValueError("accurate must be a boolean")
        self._accurate = accurate

    @utils.docinherit(FileDataset)
    def generate_data(self):
        file, label = self._get_file_label_tensors()
        value = tf.read_file(file)
        image = self._decode_image(value, accurate=self._accurate)
        # Since decode_jpeg does not set the image shape, we need to set it manually
        # https://github.com/tensorflow/tensorflow/issues/521
        # https://stackoverflow.com/questions/34746777/why-do-i-get-valueerror-image-must-be-fully-defined-when-transforming-im
        image.set_shape((self._orig_height, self._orig_width, self._orig_num_channels))
        # Labels are reshaped so that in the batch they are of 2D shape (batch, 1)
        return image, tf.reshape(label, shape=(1,))

    @staticmethod
    def _guess_orig_shape(image_files, classes):
        """A trick to guess original image shape from one of the given images
        since TensorFlow does not set static image shape."""
        if isinstance(image_files, str):
            image_files = [image_files]
        try:
            fname = FileDataset._get_files_labels(image_files[0], classes)[0][0]
        except IndexError:
            raise RuntimeError("Cannot guess original image shape since"
                               " a representative file is not found")
        # Instead of
        # img = scipy.misc.imread(fname)
        # we use this to workaround a bug:
        # https://github.com/python-pillow/Pillow/issues/835
        # https://github.com/scikit-learn/scikit-learn/issues/3410
        with open(fname, 'rb') as img_file:
            with PIL.Image.open(img_file) as img_img:
                img = scipy.misc.fromimage(img_img)
        orig_height = img.shape[0]
        orig_width = img.shape[1]
        if len(img.shape) == 3:
            orig_num_channels = img.shape[2]
        else:
            orig_num_channels = 1
        return orig_height, orig_width, orig_num_channels

    @staticmethod
    def _decode_image(contents, accurate=False):
        """
        A convenience function decompressing both JPEG and PNG images, which also
        allows setting JPEG decompression quality.
        """
        with tf.name_scope('decode_image'):
            substr = tf.substr(contents, 0, 4)

            def _png():
                return tf.image.decode_png(contents)

            def _jpeg():
                return tf.image.decode_jpeg(
                    contents,
                    dct_method=("INTEGER_ACCURATE" if accurate else None))

            is_jpeg = tf.equal(substr, b'\xff\xd8\xff\xe0', name='is_jpeg')
            return tf.cond(is_jpeg, _jpeg, _png, name='cond_jpeg')
