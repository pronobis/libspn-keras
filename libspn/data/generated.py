# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

from collections import namedtuple
import tensorflow as tf
import numpy as np
import scipy.stats
from libspn.data.dataset import Dataset
from libspn import utils


class GaussianMixtureDataset(Dataset):
    """A random dataset sampled from a mixture of Gaussians. The samples
    are labeled by the label of the mixture component they were sampled from.

    The mixture is specified by ``components``, a list of :class:`Component`.
    Each component is specified in terms of its weight, mean, covariance and,
    optionally, label. If a label is not given, it is set to the index of the
    component in ``components``. The weights of the components are normalized
    internally to sum to 1.

    The data samples will be digitized if ``num_vals`` is specified.

    The data is returned as a tuple of tensors
    ``(samples, labels, likelihoods)``, where ``samples`` has shape
    ``[batch_size, num_dims]`` and contains the samples of dimensionality
    specified by the size of ``mean``/``cov`` of :class:`Component`, ``labels``
    has shape ``[batch_size, 1]`` and contains the labels of each sample, and
    ``likelihoods`` has shape ``[batch_size]`` and contains the true
    likelihoods of each sample.

    Args:
        components (list of Components): List of components of the mixture.
        num_samples (int): Number of random samples in the dataset.
        num_epochs (int): Number of epochs of produced data.
        batch_size (int): Size of a single batch.
        shuffle (bool): Shuffle data within each epoch.
        num_threads (int): Number of threads enqueuing the example queue. If
                           larger than ``1``, the performance will be better,
                           but examples might not be in order even if
                           ``shuffle`` is ``False``.
        allow_smaller_final_batch(bool): If ``False``, the last batch will be
                                         omitted if it has less elements than
                                         ``batch_size``.
        num_vals (int): Optional. If specified, the generated samples will be
                        digitized into ``num_vals`` discrete values.
        seed (int): Optional. Seed used when shuffling.
    """

    Component = namedtuple("Component", ['weight', 'mean', 'cov', 'label'])
    Component.__new__.__defaults__ = (None,)  # Set default value for label

    def __init__(self, components, num_samples, num_epochs, batch_size,
                 shuffle, num_threads=1, allow_smaller_final_batch=False,
                 num_vals=None, seed=None):
        if not isinstance(components, list):
            raise ValueError("components must be a list")
        components = [GaussianMixtureDataset.Component(
            c.weight, np.asarray(c.mean), np.asarray(c.cov), c.label)
            for c in components]
        for c in components:
            if not isinstance(c, GaussianMixtureDataset.Component):
                raise ValueError("component '%s' is not a Component" % c)
            if c.mean.ndim != 1:
                raise ValueError("mean array of a component be 1D")
            if c.cov.ndim != 2:
                raise ValueError("cov array of a component be 2D")
            if (c.mean.shape[0] != c.cov.shape[0] or
                    c.mean.shape[0] != c.cov.shape[1]):
                raise ValueError("dimensions of mean and cov must be the same")
            if (c.mean.shape[0] != components[0].mean.shape[0]
                    or c.cov.shape != components[0].cov.shape):
                raise ValueError("components must have the same number of dimensions")
        super().__init__(num_vars=components[0].mean.shape[0],
                         num_vals=num_vals, num_labels=1,
                         num_epochs=num_epochs, batch_size=batch_size,
                         shuffle=shuffle,
                         # We shuffle the samples in this class
                         # so batch shuffling is not needed
                         shuffle_batch=False, min_after_dequeue=None,
                         num_threads=num_threads,
                         allow_smaller_final_batch=allow_smaller_final_batch,
                         seed=seed)
        self._components = components
        self._num_samples = num_samples
        self._num_vals = num_vals
        self._samples = None
        self._labels = None
        self._likelihoods = None

    @property
    def samples(self):
        """array: Array of data samples."""
        return self._samples

    @property
    def labels(self):
        """array: Array of data labels."""
        return self._labels

    @property
    def likelihoods(self):
        """array: Array of data likelihoods."""
        return self._likelihoods

    @utils.docinherit(Dataset)
    def generate_data(self):
        self.generate_samples()
        # Add input producer that serves the generated samples
        # All data is shuffled independently of the capacity parameter
        producer = tf.train.slice_input_producer([self._samples,
                                                  self._labels,
                                                  self._likelihoods],
                                                 num_epochs=self._num_epochs,
                                                 # Shuffle data
                                                 shuffle=self._shuffle,
                                                 seed=self._seed)
        return producer

    @utils.docinherit(Dataset)
    def process_data(self, data):
        return data

    def generate_samples(self):
        """Generate Gaussian Mixture samples."""
        # Get the sum of all weights for normalization
        weight_sum = sum(c.weight for c in self._components)
        # Get numbers of samples per component
        num_component_samples = []
        weights_left = weight_sum
        samples_left = self._num_samples
        for comp in self._components:
            n = int(round(samples_left * (comp.weight / weights_left)))
            num_component_samples.append(n)
            weights_left -= comp.weight
            samples_left -= n
        # Generate all data samples, we assume they will fit in memory
        labels = [np.full((num_samples, 1),
                          i if comp.label is None else comp.label,
                          dtype=int)
                  for i, (comp, num_samples)
                  in enumerate(zip(self._components, num_component_samples))]
        samples = [np.random.multivariate_normal(
            comp.mean, comp.cov, num_samples)
            for comp, num_samples
            in zip(self._components, num_component_samples)]
        # Concatenate
        self._labels = np.concatenate(labels, axis=0)
        self._samples = np.concatenate(samples, axis=0)
        # Get the true likelihood for the samples
        self._likelihoods = np.zeros(self._num_samples)
        for comp in self._components:
            var = scipy.stats.multivariate_normal(mean=comp.mean, cov=comp.cov)
            self._likelihoods += var.pdf((self._samples.astype(float))) * (comp.weight / weight_sum)
        # Discretize the samples
        sample_min = self._samples.min(0)
        sample_max = self._samples.max(0)
        if self._num_vals is not None:
            digitized_samples = np.empty_like(self._samples, dtype=int)
            # Digitize each column separately
            for i in range(self._samples.shape[1]):
                digitized_samples[:, i] = (
                    np.digitize(self._samples[:, i],
                                np.linspace(sample_min[i], sample_max[i],
                                            self._num_vals))
                    - 1)  # So that values start with 0
            self._samples = digitized_samples


class IntGridDataset(Dataset):
    """A dataset containing all integer points on an N-dimensional grid from
    ``[0, 0, ..., 0]`` to ``[num_vals-1, num_vals-1, ..., num_vals-1]``.

    The data is returned as a single tensor.

    Args:
        num_dims (int): Number of dimensions.
        num_vals (int): Number of possible values.
        num_epochs (int): Number of epochs of produced data.
        batch_size (int): Size of a single batch.
        shuffle (bool): Shuffle data within each epoch.
        num_threads (int): Number of threads enqueuing the example queue. If
                           larger than ``1``, the performance will be better,
                           but examples might not be in order even if
                           ``shuffle`` is ``False``.
        allow_smaller_final_batch(bool): If ``False``, the last batch will be
                                         omitted if it has less elements than
                                         ``batch_size``.
        seed (int): Optional. Seed used when shuffling.
    """

    def __init__(self, num_dims, num_vals, num_epochs, batch_size, shuffle,
                 num_threads=1, allow_smaller_final_batch=False, seed=None):
        super().__init__(num_vars=num_dims,
                         num_vals=num_vals,
                         num_labels=0,
                         num_epochs=num_epochs, batch_size=batch_size,
                         shuffle=shuffle,
                         # We shuffle the samples in this class
                         # so batch shuffling is not needed
                         shuffle_batch=False, min_after_dequeue=None,
                         num_threads=num_threads,
                         allow_smaller_final_batch=allow_smaller_final_batch,
                         seed=seed)
        self._num_dims = num_dims
        self._num_vals = num_vals

    @utils.docinherit(Dataset)
    def generate_data(self):
        # Generate points
        values = np.arange(0, self._num_vals)
        points = np.array(
            np.meshgrid(*[values for i in range(self._num_dims)])).T
        points = points.reshape(-1, points.shape[-1])

        # Add input producer that serves the generated samples
        # All data is shuffled independently of the capacity parameter
        producer = tf.train.slice_input_producer([points],
                                                 num_epochs=self._num_epochs,
                                                 # Shuffle data
                                                 shuffle=self._shuffle,
                                                 seed=self._seed)
        return producer

    @utils.docinherit(Dataset)
    def process_data(self, data):
        return data
