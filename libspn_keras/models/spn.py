from typing import Dict, Tuple, Union

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.engine import data_adapter


class SumProductNetwork(keras.Model):
    """
    An SPN analogue of tensorflow.keras.Model that can be trained generatively.

    It does not expect labels y when calling .fit() if ``unsupervised`` == True.

    Args:
        unsupervised (bool): If ``True`` (default) the model does not expect label inputs in
            .fit() or .evaluate(). Also, losses and metrics should not expect a target output,
            just a y_hat.
    """

    def __init__(self, *args, unsupervised: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.unsupervised = unsupervised

    def _train_step_unsupervised(
        self, data: Union[tf.Tensor, Tuple[tf.Tensor, ...]]
    ) -> Dict[str, tf.Tensor]:
        x, sample_weight, _ = data_adapter.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            out = self(x, training=True)
            dummy_target = tf.stop_gradient(out)
            loss = self.compiled_loss(
                dummy_target, out, sample_weight, regularization_losses=self.losses
            )

        trainable_variables = self.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        self.compiled_metrics.update_state(dummy_target, out, sample_weight)
        return {m.name: m.result() for m in self.metrics}

    def _test_step_unsupervised(
        self, data: Union[tf.Tensor, Tuple[tf.Tensor, ...]]
    ) -> Dict[str, tf.Tensor]:
        x, sample_weight, _ = data_adapter.unpack_x_y_sample_weight(data)
        out = self(x, training=False)
        # Updates stateful loss metrics.
        dummy_target = tf.stop_gradient(out)
        self.compiled_loss(
            dummy_target, out, sample_weight, regularization_losses=self.losses
        )

        self.compiled_metrics.update_state(dummy_target, out, sample_weight)
        return {m.name: m.result() for m in self.metrics}

    def train_step(
        self, data: Union[Tuple[tf.Tensor, ...], tf.Tensor]
    ) -> Dict[str, tf.Tensor]:
        """
        Train for one step.

        Args:
            data: Nested structure of tensors.

        Returns:
            Dict of metrics.
        """
        if self.unsupervised:
            return self._train_step_unsupervised(data)
        else:
            return super(SumProductNetwork, self).train_step(data)

    def test_step(
        self, data: Union[tf.Tensor, Tuple[tf.Tensor, ...]]
    ) -> Dict[str, tf.Tensor]:
        """
        Test for one step.

        Args:
            data: Nested structure of tensors.

        Returns:
            Dict of metrics.
        """
        if self.unsupervised:
            return self._test_step_unsupervised(data)
        else:
            return super(SumProductNetwork, self).test_step(data)
