from typing import Dict, Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.engine import data_adapter

from libspn_keras.layers.temporal_dense_product import TemporalDenseProduct


class DynamicSumProductNetwork(keras.Model):
    """
    SPN that re-uses its nodes at each time step.

    The input is expected to be pre-padded sequences with a full tensor shape
    of [num_batch, max_sequence_len, num_variables].

    Args:
        template_network: Template network that is applied to the leaves and ends with nodes that
            cover all variables for each timestep.
        interface_network_t0: Interface network for t = t0, applied on top of the template network's
            output at the current timestep.
        interface_network_t_minus_1: Interface network for t = t0 - 1, applied to the output of the
            interfaced output of the previous timestep
        top_network: Network on top of the interfaced network at the current timestep (covers all
            variables of the current timestep, including those of previous timesteps). This network
            must end with a root sum layer.
        return_last_step (bool): Whether to return only the roots at the last step with shape
            [num_batch, root_num_out] or whether to [num_batch, max_sequence_len, root_num_out]
        unsupervised (bool):
    """

    def __init__(
        self,
        template_network: keras.Model,
        interface_network_t0: keras.Model,
        interface_network_t_minus_1: keras.Model,
        top_network: keras.Model,
        return_last_step: bool = True,
        unsupervised: bool = True,
        **kwargs,
    ):
        super(DynamicSumProductNetwork, self).__init__(**kwargs)
        self.template_network = template_network
        self.unsupervised = unsupervised
        self.top_network = top_network
        self.interface_network_t0 = interface_network_t0
        self.interface_network_t_minus_1 = interface_network_t_minus_1
        self.return_last_step = return_last_step
        self.temporal_product = TemporalDenseProduct()

    def _train_step_unsupervised(self, data: Tuple[tf.Tensor]) -> Dict[str, tf.Tensor]:
        x, sequence_lens, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            out = self([x, sequence_lens], training=True)
            dummy_target = tf.stop_gradient(out)
            loss = self.compiled_loss(
                dummy_target, out, sample_weight, regularization_losses=self.losses
            )

        trainable_variables = self.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        self.compiled_metrics.update_state(dummy_target, out, sample_weight)
        return {m.name: m.result() for m in self.metrics}

    def _test_step_unsupervised(self, data: Tuple[tf.Tensor]) -> Dict[str, tf.Tensor]:
        x, sequence_lens, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        out = self([x, sequence_lens], training=False)
        # Updates stateful loss metrics.
        dummy_target = tf.stop_gradient(out)
        self.compiled_loss(
            dummy_target, out, sample_weight, regularization_losses=self.losses
        )

        self.compiled_metrics.update_state(dummy_target, out, sample_weight)
        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def call(self, input_data: Tuple[tf.Tensor, ...]) -> tf.Tensor:
        """
        Compute forward pass of the SPN.

        Args:
            input_data: Tuple of data Tensor and sequence length Tensor.

        Returns:
            Log probability of the root at each time step or that of the last timestep.

        Raises:
            ValueError: If number of input Tensors does not equal 2.
        """
        if len(input_data) != 2:
            raise ValueError(
                f"Dynamic SPN must be fed with data and sequence lengths tensors, "
                f"now got {len(input_data)} tensors."
            )
        input_data, sequence_lens = input_data[0], input_data[1]
        input_data = tf.transpose(input_data, [1, 0, 2])
        batch_size = tf.shape(input_data)[0]
        outputs = tf.TensorArray(tf.float32, batch_size)

        interface_network_t_minus_1_shape = tf.concat(
            [
                [tf.shape(input_data)[1]],
                [1, 1, self.interface_network_t0.output_shape[-1]],
            ],
            axis=0,
        )
        interface_t_minus_1 = tf.zeros(interface_network_t_minus_1_shape)
        for i in tf.range(batch_size):
            step_mask = tf.cast(
                tf.greater_equal(i, batch_size - tf.cast(sequence_lens, tf.int32)),
                tf.float32,
            )
            step_mask_flat = step_mask
            step_mask = tf.reshape(
                step_mask, tf.concat([[tf.shape(input_data)[1]], [1, 1, 1]], axis=0)
            )

            template_out = self.template_network(input_data[i]) * step_mask

            interface_t_minus_1 = interface_t_minus_1 * step_mask

            interface_t0 = self.interface_network_t0(template_out) * step_mask
            interface_template_prod = (
                self.temporal_product([interface_t_minus_1, interface_t0]) * step_mask
            )

            output = self.top_network(interface_template_prod) * tf.expand_dims(
                step_mask_flat, axis=1
            )
            outputs = outputs.write(i, output)

            interface_t_minus_1 = self.interface_network_t_minus_1(
                interface_template_prod
            )

        output = tf.transpose(outputs.stack(), [1, 0, 2])
        if self.return_last_step:
            output = output[:, -1, :]
        return output

    def train_step(self, data: Tuple[tf.Tensor]) -> Dict[str, tf.Tensor]:
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
            return super(DynamicSumProductNetwork, self).train_step(data)

    def test_step(self, data: Tuple[tf.Tensor]) -> Dict[str, tf.Tensor]:
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
            return super(DynamicSumProductNetwork, self).test_step(data)
