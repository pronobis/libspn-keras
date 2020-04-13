import itertools

from tensorflow.keras import initializers
import numpy as np
import tensorflow as tf


class KMeans(initializers.Initializer):
    """
    Initializer learned through K-means from data. The centroids learned from K-means are
    used to initialize the location parameters of a location-scale leaf, such as a ``NormalLeaf``.
    This is particularly useful for variables with dimensionality of greater than 1.

    Notes:
        Currently only works for data with batch along the ``0`` axis (as it is for
        spatial SPNs).

    Args:
        data (numpy.ndarray): Data on which to perform K-means.
        samplewise_normalization (bool): Whether to normalize data before learning centroids.
        data_fraction (float): Fraction of the data to use for K-means (chosen randomly)
        normalization_epsilon (float): Normalization constant (only used when
            ``sample_normalization`` is ``True``.
        stop_epsilon: Non-zero constant for difference in MSE on which to stop K-means fitting.
        num_iters (int): Maximum number of iterations.
        group_centroids (bool): If ``True``, performs another round of K-means to group the
            centroids along the scope axes.
        max_num_clusters (int): Maximum number of clusters (use this to limit the memory needed)
        jitter_factor (float): If the number of clusters is larger than allowed according to
            ``max_num_clusters``, the learned ``max_num_clusters`` centroids are repeated and
            then jittered with noise generated from a truncated normal distribution with a
            standard deviation of ``jitter_factor``
        centroid_initialization (str): Centroid initialization algorithm. If ``"kmeans++"``, will
            iteratively initialize clusters far apart from each other. Otherwise, the centroids
            will be initialized from the data randomly.
    """

    def __init__(self, data=None, samplewise_normalization=True, data_fraction=0.2,
                 normalization_epsilon=1e-2, stop_epsilon=1e-4, num_iters=100,
                 group_centroids=True, max_num_clusters=8, jitter_factor=0.05,
                 centroid_initialization="kmeans++"):
        self._data = data
        self.samplewise_normalization = samplewise_normalization
        self.normalization_epsilon = normalization_epsilon
        self.data_fraction = data_fraction
        self.stop_epsilon = stop_epsilon
        self.group_centroids = group_centroids
        self.num_iters = num_iters
        self.max_num_clusters = max_num_clusters
        self.jitter_factor = jitter_factor
        self.centroid_initialization = centroid_initialization

    def __call__(self, shape, dtype=None, partition_info=None):

        num_components = shape[-2]
        if num_components > self.max_num_clusters and num_components % self.max_num_clusters != 0:
            raise ValueError("Number of components must be multiple of max number of clusters")

        if self.samplewise_normalization:
            axes = tuple(range(1, len(self._data.shape)))
            data = (self._data - np.mean(self._data, axis=axes, keepdims=True)) \
                   / (np.std(self._data, axis=axes, keepdims=True) + self.normalization_epsilon)
        else:
            data = self._data

        batch_size, *middle_dims, dimensionality = data.shape

        fraction_size = int(len(data) * self.data_fraction)
        indices = np.random.choice(np.arange(len(data)), size=fraction_size)

        data_by_kmeans_problem = data[indices].reshape(
            [fraction_size, -1, dimensionality]).transpose((1, 0, 2)).astype(np.float32)

        num_clusters = min(self.max_num_clusters, num_components)
        centroids = self._kmeans_tf(data_by_kmeans_problem, num_clusters=num_clusters)

        if self.group_centroids:
            centroids = self._group_centroids(centroids, num_clusters)
        if num_clusters < num_components:
            centroids = tf.expand_dims(centroids, axis=-2)
            noise = tf.random.normal(tf.concat(
                [centroids.shape[:-2], [num_components // num_clusters], centroids.shape[-1:]],
                axis=0),
                stddev=self.jitter_factor
            )
            centroids = centroids + noise

        return np.reshape(np.asarray(centroids), shape)

    def get_config(self):
        return {
            "samplewise_normalization": self.samplewise_normalization,
            "normalization_epsilon": self.normalization_epsilon,
            "num_iters": self.num_iters,
            "stop_epsilon": self.stop_epsilon,
            "group_centroids": self.group_centroids
        }

    def _kmeans_tf(self, data, num_clusters):

        num_problems, num_batch, num_dims = data.shape
        if self.centroid_initialization == "kmeans++":
            indices = tf.random.categorical(
                logits=tf.zeros([1, num_batch]),
                num_samples=num_problems
            )
            centroids = tf.gather(data, tf.transpose(indices, (1, 0)), axis=1, batch_dims=1)

            for _ in tf.range(num_clusters - 1):
                distances = tf.reduce_sum(
                    tf.math.squared_difference(
                        tf.expand_dims(centroids, axis=2),
                        tf.expand_dims(data, axis=1)
                    ),
                    axis=-1
                )
                min_distances = tf.reduce_min(distances, axis=1)
                logits = tf.math.log(min_distances)
                indices = tf.random.categorical(
                    logits=logits,
                    num_samples=1
                )
                new_centroids = tf.gather(data, indices, axis=1, batch_dims=1)
                centroids = tf.concat([centroids, new_centroids], axis=1)
        else:
            indices = tf.random.categorical(
                logits=tf.zeros([1, num_batch]),
                num_samples=num_clusters * num_problems
            )
            indices = tf.reshape(indices, (num_problems, num_clusters))
            centroids = tf.gather(data, indices, axis=1, batch_dims=1)
            centroids += tf.random.normal(centroids.shape, stddev=0.05, dtype=centroids.dtype)

        mse = None
        for _ in range(self.num_iters):
            centroids, mse_new = self._kmeans_step(data, centroids, num_clusters)
            if mse is not None and tf.abs(mse - mse_new) < self.stop_epsilon:
                break
            mse = mse_new
        return centroids

    def _group_centroids(self, centroids, num_clusters):
        flat_centroids = tf.reshape(centroids, (-1, centroids.shape[-1]))

        if self.centroid_initialization == "kmeans++":
            indices = tf.reshape(tf.random.categorical(
                logits=tf.zeros([1, flat_centroids.shape[0]]),
                num_samples=1
            ), (-1,))
            super_centroids = tf.gather(flat_centroids, indices, axis=0)
            for _ in tf.range(num_clusters - 1):
                distances = tf.reduce_sum(
                    tf.math.squared_difference(
                        tf.expand_dims(super_centroids, axis=1),
                        tf.expand_dims(flat_centroids, axis=0)
                    ),
                    axis=-1
                )
                min_distances = tf.reduce_min(distances, axis=0, keepdims=True)
                logits = tf.math.log(min_distances)
                indices = tf.random.categorical(
                    logits=logits, num_samples=1
                )
                new_super_centroids = tf.gather(flat_centroids, tf.reshape(indices, (-1,)), axis=0)
                super_centroids = tf.concat([super_centroids, new_super_centroids], axis=0)
        else:
            indices = tf.reshape(tf.random.categorical(
                logits=tf.zeros([1, flat_centroids.shape[0]]),
                num_samples=num_clusters
            ), (-1,))
            super_centroids = tf.gather(flat_centroids, indices, axis=0)

        def generate_permutations(n):
            return list(itertools.permutations(range(n)))

        permutations = generate_permutations(num_clusters)

        mse = tf.reduce_mean(
            tf.reduce_sum(tf.math.squared_difference(centroids, super_centroids), axis=-1),
        )
        for _ in range(self.num_iters):
            super_centroids, centroids, new_mse = self._assign_to_supercentroid(
                centroids, super_centroids, permutations)
            if mse is not None and tf.abs(new_mse - mse) < self.stop_epsilon:
                break
            mse = new_mse
        return centroids

    # @tf.function
    def _assign_to_supercentroid(self, centroids, super_centroids, permutations):
        num_clusters = centroids.shape[1]
        distances = tf.reduce_sum(tf.math.squared_difference(
            tf.expand_dims(tf.expand_dims(super_centroids, axis=0), axis=2),
            tf.expand_dims(centroids, axis=1)
        ), axis=-1)
        distances_flat = tf.reshape(distances, (-1, num_clusters * num_clusters))
        indices = permutations + tf.range(num_clusters) * num_clusters
        assignment_distances = tf.reduce_mean(
            tf.gather(distances_flat, indices, axis=1), axis=-1
        )
        assignment_argmin = tf.argmin(assignment_distances, axis=-1)
        assignments = tf.gather(permutations, assignment_argmin, axis=0)
        centroids = tf.gather(centroids, assignments, axis=1, batch_dims=1)
        super_centroids_new = tf.reduce_mean(centroids, axis=0)
        mse = tf.reduce_mean(
            tf.reduce_sum(tf.math.squared_difference(centroids, super_centroids_new), axis=-1),
        )
        return super_centroids_new, centroids, mse

    def _assign_to_centroid(self, data, centroids):
        distances = tf.reduce_sum(tf.math.squared_difference(
            tf.expand_dims(centroids, axis=1),
            tf.expand_dims(data, axis=2)
        ), axis=-1)
        mse = tf.reduce_mean(tf.reduce_min(distances, axis=2))
        return tf.argmin(distances, axis=2), mse


    @tf.function
    def _kmeans_step(self, data, centroids, num_clusters):
        assignment, mse = self._assign_to_centroid(data, centroids)

        def compute_mean(x):
            return tf.cast(
                tf.math.unsorted_segment_mean(x[0], x[1], num_segments=num_clusters),
                tf.float32
            )

        new_centroids = tf.map_fn(
            compute_mean, (data, assignment), dtype=tf.float32
        )
        return new_centroids, mse
