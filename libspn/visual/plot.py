# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

"""LibSPN plotting functions."""
from libspn.data.image import ImageShape
import numpy as np
import matplotlib as mpl
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

# Seaborn has great plot examples:
# https://stanford.edu/~mwaskom/software/seaborn/index.html


def plot_2d(x, y, labels=None, probs=None, jitter=True):
    """Show a scatter plot of 2-dimensional data. If ``labels`` are given,
    indicate the first label of each data sample. If ``probs`` are given,
    visualize the probability of each sample.

    Args:
        data (array): 2D data points.
        labels (array): Optional. Integer labels of the data points.
        probs (array): Optional. Probabilities for each point.
        jitter (bool): If ``True``, categorical values will be jittered a bit.
    """
    # Setup figure
    plt.figure()  # figsize=(width, height)
    plt.style.use("ggplot")  # bmh, fivethirtyeight

    # Get colrs and markers
    cmap = mpl.cm.get_cmap('Paired')
    markers = mpl.markers.MarkerStyle.filled_markers  # 13 markers
    colors = cmap(np.linspace(0, 1, 12))  # Paired has 12 distinct colors
    colors = np.concatenate((colors[1::2], colors[0::2]))  # Darker colors first

    # Countour plot of probabilities
    if probs is not None:
        xi = np.linspace(x.min(), x.max(), 100)
        yi = np.linspace(y.min(), y.max(), 200)
        zi = mlab.griddata(x, y, probs, xi, yi, interp='linear')
        plt.contourf(xi, yi, zi, 25, cmap=plt.cm.bone_r)
        plt.colorbar()

    # Jitter points if data is categorical
    jitter_radius = 0.25
    if jitter and issubclass(x.dtype.type, np.integer):
        x = x.astype(float)
        x += np.random.uniform(-jitter_radius, jitter_radius, x.size)
        y = y.astype(float)
        y += np.random.uniform(-jitter_radius, jitter_radius, y.size)

    # Plot points
    if labels is None:
        plt.scatter(x=x, y=y, marker=markers[0],
                    c=colors[0], s=50)
    else:
        label_list = sorted(set(labels))
        for l in label_list:
            px = x[labels == l]
            py = y[labels == l]
            plt.scatter(x=px, y=py,
                        marker=markers[l % len(markers)],
                        c=colors[l % len(colors)], s=50)

    # TODO: display probabilities
    # http://matplotlib.org/examples/pylab_examples/griddata_demo.html
    plt.show()


def image(image, shape, normalize=False):
    """Show an image from flattened data.

    Args:
        image (array): Flattened image data
        shape (ImageShape): Shape of the image data.
        normalize (bool): Normalize data before displaying.
    """
    if not isinstance(shape, ImageShape):
        raise ValueError("shape must be ImageShape")
    if not (issubclass(image.dtype.type, np.integer) or
            issubclass(image.dtype.type, np.floating)):
        raise ValueError("image must be of int or float dtype")

    if shape[2] == 1:  # imshow wants single channel as MxN
        shape = shape[:2]
    image = image.reshape(shape)
    cmap = mpl.cm.get_cmap('Greys_r')
    if normalize:
        vmin = None
        vmax = None  # imshow will normalize
    else:
        if issubclass(image.dtype.type, np.integer):
            vmin = 0
            vmax = 255
        elif issubclass(image.dtype.type, np.floating):
            vmin = 0.0
            vmax = 1.0
    plt.imshow(image, cmap=cmap, interpolation='none',
               vmin=vmin, vmax=vmax)
    plt.grid(False)
    plt.show()
