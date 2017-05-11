# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

from abc import ABC, abstractmethod
import numpy as np
import os
import scipy
from libspn.log import get_logger


class DataWriter(ABC):
    """An abstract class defining the interface of a data writer."""

    @abstractmethod
    def write(self, data):
        """Write arrays of data.

        Args:
            data: An array, a list of arrays or a dictionary of arrays with
                  the data to write.
        """


class CSVDataWriter(DataWriter):
    """Writer that writes data in the CSV format.

    Args:
        path (str): Full path to the file.
        delimiter (str): A one-character string used to separate fields.
        fmt_int (str): The format used for storing integers.
        fmt_float (str): The format used for storing floats.
    """

    def __init__(self, path, delimiter=',', fmt_int="%d", fmt_float="%.18e"):
        self._path = os.path.expanduser(path)
        self._delimiter = delimiter
        self._fmt_int = fmt_int
        self._fmt_float = fmt_float
        self.__mode = 'wb'  # Initial mode

    def write(self, *data):
        """Write arrays of data. The first call to write erases any existing
        file, while all subsequent calls will append to the same file.

        Args:
            data: Arrays with the data to write.
        """
        # Get columns
        cols = []
        fmt = ['%d', '%f', '%f']
        for d in data:
            if d.ndim == 1:
                cols.append(d)
            elif d.ndim == 2:
                cols.extend(d.T)
            else:
                raise ValueError("Arrays must be 1 or 2 dimensional")

        # Get structured array and formats
        arr = np.rec.fromarrays(cols)
        fmt = [self._fmt_int
               if np.issubdtype(c.dtype, np.integer)
               else self._fmt_float
               for c in cols]

        # Write
        with open(self._path, self.__mode) as csvfile:
            np.savetxt(csvfile, arr, delimiter=',', fmt=fmt)

        # Append further writes
        self.__mode = 'ab'


class ImageDataWriter(DataWriter):
    """Writer writing flattened image data. The image format is selected by the
    extension given as part of the path. The images are normalized to [0, 1]
    before being saved.

    Args:
        path (str): Path to an image file. The path can be parameterized by
                    ``%n``, which will be replaced by the number of the image.
                    It can also be parameterized by ``%l`` which will be replaced
                    by any labels given to :func:`write`.
        shape (ImageShape): Shape of the image data.
        normalize (bool): Normalize data before saving.
        num_digits (int): Minimum number of digits of the image number.
    """

    logger = get_logger()
    debug1 = logger.debug1

    def __init__(self, path, shape, normalize=False, num_digits=4):
        self._path = os.path.expanduser(path)
        self._shape = shape
        self._normalize = normalize
        self._num_digits = num_digits
        self._image_no = 0

    def write(self, images, labels=None, image_no=None):
        """Write image data to file(s). `images` can be either a single image

        Args:
            images (array): An array containing a single image or multiple
                            images of the given shape.
            labels (value or array): A value or array of values used to label
                                     the image or images given.
            image_no (int): Optional. Number of the first image to write.
                            If not given, the number of the last written image
                            is incremented and used.
        """
        if not (issubclass(images.dtype.type, np.integer) or
                issubclass(images.dtype.type, np.floating)):
            raise ValueError("images must be of int or float dtype")
        if image_no is not None and not isinstance(image_no, int):
            raise ValueError("image_no must be integer")

        self.debug1("Batch size:%s dtype:%s max_min:%s min_max:%s" %
                    (images.shape[0], images.dtype,
                     np.amax(np.amin(images, axis=1)),
                     np.amin(np.amax(images, axis=1))))

        # Convert 1-image case to multi-image case
        if images.ndim == 1:
            images = images.reshape([1, -1])
            if labels is not None:
                labels = [labels]
        # Set image number
        if image_no is not None:
            self._image_no = image_no
        # Calculate shape for imsave
        shape = self._shape
        if self._shape[2] == 1:  # imshow wants single channel as MxN
            shape = self._shape[:2]
        # Save all images
        for i in range(images.shape[0]):
            # Generate path
            path = self._path
            if '%n' in path:
                path = path.replace(
                    '%n', ("%0" + str(self._num_digits) + "d") % (self._image_no))
            if labels is not None and '%l' in path:
                label = labels[i]
                if isinstance(label, bytes):
                    label = label.decode("utf-8")
                else:
                    label = str(label)
                path = path.replace('%l', label)
            # Normalize?
            # imsave normalizes float images, but not uint8 images
            if self._normalize:
                if issubclass(images.dtype.type, np.integer):
                    images = images.astype(np.float32)
            else:
                if issubclass(images.dtype.type, np.floating):
                    images *= 255.0
                images = images.astype(np.uint8)  # Convert also int32/64 to 8
            # Save
            scipy.misc.imsave(path, images[i].reshape(shape))
            self._image_no += 1
