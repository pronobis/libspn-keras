# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

"""LibSPN image visualization functions."""

from libspn.data.image import ImageShape
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def show_image(image, shape, normalize=False):
    """Show an image from flattened data.

    Args:
        image (array): Flattened image data
        shape (ImageShape): Shape of the image data.
        normalize (bool): Normalize data before displaying.
    """
    if not isinstance(shape, ImageShape):
        raise ValueError("shape must be ImageShape")
    if not (np.issubdtype(image.dtype, np.integer) or
            np.issubdtype(image.dtype, np.floating)):
        raise ValueError("image must be of int or float dtype")

    if shape[2] == 1:  # imshow wants single channel as MxN
        shape = shape[:2]
    image = image.reshape(shape)
    cmap = mpl.cm.get_cmap('Greys_r')
    if normalize:
        vmin = None
        vmax = None  # imshow will normalize
    else:
        if np.issubdtype(image.dtype, np.integer):
            vmin = 0
            vmax = 255
        elif np.issubdtype(image.dtype, np.floating):
            vmin = 0.0
            vmax = 1.0
    plt.imshow(image, cmap=cmap, interpolation='none',
               vmin=vmin, vmax=vmax)
    plt.grid(False)
    plt.show()
