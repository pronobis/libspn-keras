from typing import Optional

import tensorflow as tf
from tensorflow import keras


class LogJoint(keras.metrics.Mean):
    r"""
    Compute log joint :math:`1/N \sum \log(p(X,Y))`.

    Assumes that the last layer of the SPN is a ``RootSum`` with ``return_weighted_child_logits=True``.
    """

    def __init__(self, name: str = "log_joint", **kwargs):
        super(LogJoint, self).__init__(name=name, **kwargs)

    def update_state(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        sample_weight: Optional[float] = None,
    ) -> tf.Tensor:
        """
        Accumulates statistics for computing the reduction metric.

        For example, if `values` is [1, 3, 5, 7] and reduction=SUM_OVER_BATCH_SIZE,
        then the value of `result()` is 4. If the `sample_weight` is specified as
        [1, 1, 0, 0] then value of `result()` would be 2.

        Args:
            y_true: Prediction target.
            y_pred: Predictions coming from the model.
            sample_weight: Optional weighting of each example. Defaults to 1.

        Returns:
            Update op.
        """
        values = tf.gather(y_pred, tf.cast(y_true, tf.int32), batch_dims=1, axis=1)
        return super(LogJoint, self).update_state(values, sample_weight=sample_weight)
