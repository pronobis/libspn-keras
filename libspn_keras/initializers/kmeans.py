import itertools
from typing import Any, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import initializers


class KMeans(initializers.Initializer):
    """
    Initializer learned through K-means from data.

    The centroids learned from K-means are used to initialize the location parameters of a location-scale
    leaf, such as a ``NormalLeaf``. This is particularly useful for variables with dimensionality of
    greater than 1.


    Notes:
        Currently only works for spatial SPNs.

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

    def __init__(
        self,
        data: Optional[np.ndarray] = None,
        samplewise_normalization: bool = True,
        data_fraction: float = 0.2,
        normalization_epsilon: float = 1e-2,
        stop_epsilon: float = 1e-4,
        num_iters: int = 100,
        group_centroids: bool = True,
        max_num_clusters: int = 8,
        jitter_factor: float = 0.05,
        centroid_initialization: str = "kmeans++",
        downsample: Optional[int] = None,
        use_groups: bool = False,
    ):
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
        self.downsample = downsample
        self.use_groups = use_groups

    def __call__(  # noqa: C901
        self, shape: Tuple[Optional[int], ...], dtype: Optional[tf.dtypes.DType] = None
    ) -> tf.Tensor:
        """
        Compute KMeans initializations on the last axis.

        Args:
            shape: Shape of Tensor to initialize.
            dtype: DType of Tensor to initialize.

        Returns:
            Initial value.

        Raises:
            ValueError: If shape cannot be determined.
        """
        data = self._data
        if data is None:
            raise ValueError(
                "Cannot compute KMeans initialization without provided data"
            )
        height, width = shape[1:3]
        if height is None:
            raise ValueError("Unknown height")
        if width is None:
            raise ValueError("Unknown width")
        if self.downsample is not None:
            data = tf.image.resize(
                data, size=(height // self.downsample, width // self.downsample)
            ).numpy()

        num_components = shape[-2]
        if num_components is None:
            raise ValueError("Unknown number of components")
        if (
            num_components > self.max_num_clusters
            and num_components % self.max_num_clusters != 0
        ):
            raise ValueError(
                "Number of components must be multiple of max number of clusters"
            )

        if self.samplewise_normalization:
            axes = tuple(range(1, len(data.shape)))
            data = (data - tf.reduce_mean(data, axis=axes, keepdims=True)) / (
                tf.math.reduce_std(data, axis=axes, keepdims=True)
                + self.normalization_epsilon
            )
            data = data.numpy()

        batch_size, *middle_dims, dimensionality = data.shape

        fraction_size = int(len(data) * self.data_fraction)
        indices = np.random.choice(np.arange(len(data)), size=fraction_size)

        data_by_kmeans_problem = (
            data[indices]
            .reshape([fraction_size, -1, dimensionality])
            .transpose((1, 0, 2))
            .astype(np.float32)
        )

        num_clusters = min(self.max_num_clusters, num_components)
        centroids = self._kmeans_tf(data_by_kmeans_problem, num_clusters=num_clusters)

        if self.group_centroids:
            centroids = self._group_centroids(centroids, num_clusters)
        if num_clusters < num_components:
            centroids = tf.expand_dims(centroids, axis=-2)
            noise = tf.random.normal(
                tf.concat(
                    [
                        centroids.shape[:-2],
                        [num_components // num_clusters],
                        centroids.shape[-1:],
                    ],
                    axis=0,
                ),
                stddev=self.jitter_factor,
            )
            centroids = centroids + noise

        if self.downsample:
            centroids = tf.reshape(
                centroids,
                (
                    1,
                    height // self.downsample,
                    width // self.downsample,
                    num_components,
                    centroids.shape[-1],
                ),
            )
            dims_prefix, last_two_dims = (
                tf.shape(centroids)[:-2],
                tf.shape(centroids)[-2:],
            )
            channels_and_centroids_on_last_axis = tf.reshape(
                centroids,
                tf.concat([dims_prefix, [last_two_dims[0] * last_two_dims[1]]], axis=0),
            )
            centroids = tf.reshape(
                tf.image.resize(channels_and_centroids_on_last_axis, (height, width)),
                [dims_prefix[0], height, width, last_two_dims[0], last_two_dims[1]],
            )

        return np.reshape(np.asarray(centroids), shape)

    def get_config(self) -> dict:
        """
        Obtain config.

        Returns:
            Key-value mapping of the configuration.
        """
        return {
            "samplewise_normalization": self.samplewise_normalization,
            "normalization_epsilon": self.normalization_epsilon,
            "num_iters": self.num_iters,
            "stop_epsilon": self.stop_epsilon,
            "group_centroids": self.group_centroids,
        }

    def _kmeans_tf(self, data: np.ndarray, num_clusters: int) -> tf.Tensor:

        num_problems, num_batch, num_dims = data.shape
        if self.centroid_initialization == "kmeans++":
            indices = tf.random.categorical(
                logits=tf.zeros([1, num_batch]), num_samples=num_problems
            )
            centroids = tf.gather(
                data, tf.transpose(indices, (1, 0)), axis=1, batch_dims=1
            )

            for _ in tf.range(num_clusters - 1):
                distances = tf.reduce_sum(
                    tf.math.squared_difference(
                        tf.expand_dims(centroids, axis=2), tf.expand_dims(data, axis=1)
                    ),
                    axis=-1,
                )
                min_distances = tf.reduce_min(distances, axis=1)
                logits = tf.math.log(min_distances)
                indices = tf.random.categorical(logits=logits, num_samples=1)
                new_centroids = tf.gather(data, indices, axis=1, batch_dims=1)
                centroids = tf.concat([centroids, new_centroids], axis=1)
        else:
            indices = tf.random.categorical(
                logits=tf.zeros([1, num_batch]), num_samples=num_clusters * num_problems
            )
            indices = tf.reshape(indices, (num_problems, num_clusters))
            centroids = tf.gather(data, indices, axis=1, batch_dims=1)
            centroids += tf.random.normal(
                centroids.shape, stddev=0.05, dtype=centroids.dtype
            )

        mse = None
        for _ in range(self.num_iters):
            centroids, mse_new = self._kmeans_step(data, centroids, num_clusters)
            if mse is not None and tf.abs(mse - mse_new) < self.stop_epsilon:
                break
            mse = mse_new
        return centroids

    def _group_centroids(self, centroids: tf.Tensor, num_clusters: int) -> tf.Tensor:
        flat_centroids = tf.reshape(centroids, (-1, centroids.shape[-1]))

        if self.centroid_initialization == "kmeans++":
            indices = tf.reshape(
                tf.random.categorical(
                    logits=tf.zeros([1, flat_centroids.shape[0]]), num_samples=1
                ),
                (-1,),
            )
            super_centroids = tf.gather(flat_centroids, indices, axis=0)
            for _ in tf.range(num_clusters - 1):
                distances = tf.reduce_sum(
                    tf.math.squared_difference(
                        tf.expand_dims(super_centroids, axis=1),
                        tf.expand_dims(flat_centroids, axis=0),
                    ),
                    axis=-1,
                )
                min_distances = tf.reduce_min(distances, axis=0, keepdims=True)
                logits = tf.math.log(min_distances)
                indices = tf.random.categorical(logits=logits, num_samples=1)
                new_super_centroids = tf.gather(
                    flat_centroids, tf.reshape(indices, (-1,)), axis=0
                )
                super_centroids = tf.concat(
                    [super_centroids, new_super_centroids], axis=0
                )
        else:
            indices = tf.reshape(
                tf.random.categorical(
                    logits=tf.zeros([1, flat_centroids.shape[0]]),
                    num_samples=num_clusters,
                ),
                (-1,),
            )
            super_centroids = tf.gather(flat_centroids, indices, axis=0)

        def generate_permutations(n: int) -> List[Tuple[Any]]:
            return list(*itertools.permutations(range(n)))

        permutations = generate_permutations(num_clusters)

        mse = tf.reduce_mean(
            tf.reduce_sum(
                tf.math.squared_difference(centroids, super_centroids), axis=-1
            ),
        )
        for _ in range(self.num_iters):
            super_centroids, centroids, new_mse = self._assign_to_supercentroid(
                centroids, super_centroids, permutations
            )
            if mse is not None and tf.abs(new_mse - mse) < self.stop_epsilon:
                break
            mse = new_mse

        if self.use_groups:
            return tf.tile(
                tf.expand_dims(super_centroids, axis=0), [tf.shape(centroids)[0], 1, 1]
            )

        return centroids

    def _assign_to_supercentroid(
        self, centroids: tf.Tensor, super_centroids: tf.Tensor, permutations: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        num_clusters = centroids.shape[1]
        distances = tf.reduce_sum(
            tf.math.squared_difference(
                tf.expand_dims(tf.expand_dims(super_centroids, axis=0), axis=2),
                tf.expand_dims(centroids, axis=1),
            ),
            axis=-1,
        )
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
            tf.reduce_sum(
                tf.math.squared_difference(centroids, super_centroids_new), axis=-1
            ),
        )
        return super_centroids_new, centroids, mse

    def _assign_to_centroid(
        self, data: tf.Tensor, centroids: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        distances = tf.reduce_sum(
            tf.math.squared_difference(
                tf.expand_dims(centroids, axis=1), tf.expand_dims(data, axis=2)
            ),
            axis=-1,
        )
        mse = tf.reduce_mean(tf.reduce_min(distances, axis=2))
        return tf.argmin(distances, axis=2), mse

    @tf.function
    def _kmeans_step(
        self, data: tf.Tensor, centroids: tf.Tensor, num_clusters: int
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        assignment, mse = self._assign_to_centroid(data, centroids)

        def _compute_mean(x: tf.Tensor) -> tf.Tensor:
            return tf.cast(
                tf.math.unsorted_segment_mean(x[0], x[1], num_segments=num_clusters),
                tf.float32,
            )

        new_centroids = tf.map_fn(_compute_mean, (data, assignment), dtype=tf.float32)
        return new_centroids, mse
