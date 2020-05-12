import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.keras.engine.sequential import _get_shape_tuple, SINGLE_LAYER_OUTPUT_ERROR_MSG
from tensorflow.python.util import nest
from libspn_keras.layers import LocationScaleLeafBase, NormalizeStandardScore


class SequentialSumProductNetwork(keras.Sequential):
    """
    An analogue of ``tensorflow.keras.Sequential`` that can be trained in an unsupervised way. It does not expect
    labels y when calling .fit() if ``unsupervised`` == True. Inherits from ``keras.Sequential``, so layers are
    passed to it as a list.

    Args:
        unsupervised (bool): If ``True`` the model does not expect label inputs in .fit() or .evaluate().
            Also, losses and metrics should not expect a target output, just a y_hat. By default, it will be inferred
            from ``infer_no_evidence``.
        infer_no_evidence (bool): If ``True``, the model expects an evidence mask defined as a boolean tensor which is
            used to mask out variables that are not part of the evidence.
    """

    def __init__(self, *args, infer_no_evidence=False, unsupervised=None, **kwargs):
        if unsupervised is None:
            unsupervised = False if infer_no_evidence else True
        super().__init__(*args, **kwargs)
        self.unsupervised = unsupervised
        if infer_no_evidence and unsupervised:
            raise ValueError("Model cannot be unsupervised when evidence should be inferred")
        if infer_no_evidence:
            self._normalize_index = self._normalize_layer = None
            for i, layer in enumerate(self.layers):
                if isinstance(layer, NormalizeStandardScore):
                    self._normalize_index = i
                    self._normalize_layer = layer

                if isinstance(layer, LocationScaleLeafBase):
                    self._leaf_index = i
                    self._leaf_layer = layer
                    break
            else:
                raise ValueError("No LocationScaleLeafBase leaf layer found")
            self.call = self._call_backprop_to_leaves
            self.train_step = self._train_step_masked_leaves

    def _train_step_masked_leaves(self, data):
        x, evidence_mask, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            out = self([x, evidence_mask], training=True)
            no_evidence = tf.logical_not(evidence_mask)
            x_without_evidence = tf.boolean_mask(x, no_evidence)
            out_without_evidence = tf.boolean_mask(out, no_evidence)
            loss = self.compiled_loss(
                x_without_evidence, out_without_evidence, sample_weight, regularization_losses=self.losses)

        trainable_variables = self.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        self.compiled_metrics.update_state(x_without_evidence, out_without_evidence, sample_weight)
        return {m.name: m.result() for m in self.metrics}

    def _call_backprop_to_leaves(self, inputs, training=None, mask=None):  # pylint: disable=redefined-outer-name
        inputs, evidence_mask = inputs[0], inputs[1]
        if self._build_input_shape is None:
            input_shapes = nest.map_structure(_get_shape_tuple, inputs)
            self._build_input_shape = input_shapes

        outputs = inputs  # handle the corner case where self.layers is empty
        with tf.GradientTape() as tape:
            for i, layer in enumerate(self.layers):
                # During each iteration, `inputs` are the inputs to `layer`, and `outputs`
                # are the outputs of `layer` applied to `inputs`. At the end of each
                # iteration `inputs` is set to `outputs` to prepare for the next layer.
                kwargs = {}
                argspec = self._layer_call_argspecs[layer].args
                if 'training' in argspec:
                    kwargs['training'] = training

                if i == self._leaf_index:
                    leaf_inputs = inputs

                if i == self._normalize_index:
                    kwargs['return_stats'] = True
                    outputs, mean, stddev = layer(inputs, **kwargs)
                else:
                    outputs = layer(inputs, **kwargs)

                if i == self._leaf_index:
                    outputs = leaf_out = tf.where(evidence_mask, outputs, tf.zeros_like(outputs))
                    tape.watch(leaf_out)

                if len(nest.flatten(outputs)) != 1:
                    raise ValueError(SINGLE_LAYER_OUTPUT_ERROR_MSG)
                # `outputs` will be the inputs to the next layer.
                inputs = outputs

        leaf_grads = tape.gradient(outputs, leaf_out)
        modes = self._leaf_layer.get_modes()
        outputs = tf.reduce_sum(tf.expand_dims(leaf_grads, axis=-1) * modes, axis=3)
        outputs = tf.where(evidence_mask, leaf_inputs, outputs)
        if self._normalize_index is not None:
            outputs = outputs * stddev + mean

        return outputs

    def _train_step_unsupervised(self, data):
        x, sample_weight, _ = data_adapter.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            out = self(x, training=True)
            dummy_target = tf.stop_gradient(out)
            loss = self.compiled_loss(dummy_target, out, sample_weight, regularization_losses=self.losses)

        trainable_variables = self.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        self.compiled_metrics.update_state(dummy_target, out, sample_weight)
        return {m.name: m.result() for m in self.metrics}

    def _test_step_unsupervised(self, data):
        x, sample_weight, _ = data_adapter.unpack_x_y_sample_weight(data)
        out = self(x, training=False)
        # Updates stateful loss metrics.
        dummy_target = tf.stop_gradient(out)
        self.compiled_loss(dummy_target, out, sample_weight, regularization_losses=self.losses)

        self.compiled_metrics.update_state(dummy_target, out, sample_weight)
        return {m.name: m.result() for m in self.metrics}

    def train_step(self, data):
        if self.unsupervised:
            return self._train_step_unsupervised(data)
        else:
            return super(SequentialSumProductNetwork, self).train_step(data)

    def test_step(self, data):
        if self.unsupervised:
            return self._test_step_unsupervised(data)
        else:
            return super(SequentialSumProductNetwork, self).test_step(data)
