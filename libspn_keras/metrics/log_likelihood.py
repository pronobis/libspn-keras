from typing import Optional

import tensorflow as tf
from tensorflow import keras


class LogLikelihood(keras.metrics.Mean):
    r"""
    Compute log marginal :math:`1/N \sum \log(p(X))`.

    Assumes that the last layer of the SPN is a ``RootSum``. It ignores the ``y_true`` argument,
    since a target for :math:`Y` is absent in unsupervised learning.
    """

    def __init__(self, name: str = "llh", **kwargs):
        super(LogLikelihood, self).__init__(name=name, **kwargs)

    def update_state(
        self, _: tf.Tensor, y_pred: tf.Tensor, sample_weight: Optional[float] = None
    ) -> tf.Tensor:
        """
        Accumulates statistics for computing the reduction metric.

        For example, if `values` is [1, 3, 5, 7] and reduction=SUM_OVER_BATCH_SIZE,
        then the value of `result()` is 4. If the `sample_weight` is specified as
        [1, 1, 0, 0] then value of `result()` would be 2.

        Args:
            y_pred: Predictions coming from the model.
            sample_weight: Optional weighting of each example. Defaults to 1.

        Returns:
            Update op.
        """
        values = tf.reduce_logsumexp(y_pred, axis=-1)
        return super(LogLikelihood, self).update_state(
            values, sample_weight=sample_weight
        )
