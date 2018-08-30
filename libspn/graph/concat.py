# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

from itertools import chain
import tensorflow as tf
from libspn.graph.node import OpNode, Input
from libspn import utils
from libspn.inference.type import InferenceType
from libspn.exceptions import StructureError
from libspn.utils.serialization import register_serializable


@register_serializable
class Concat(OpNode):
    """An op node that concatenates all inputs into a single output tensor.

    Args:
        *inputs: Inputs of this node. See :meth:`~libspn.Input.as_input` for
             possible values.
        name (str): Name of the node.
    """

    def __init__(self, *inputs, name="Concat"):
        super().__init__(InferenceType.MARGINAL, name)
        self.set_inputs(*inputs)

    def serialize(self):
        data = super().serialize()
        data['inputs'] = [(i.node.name, i.indices) for i in self._inputs]
        return data

    def deserialize(self, data):
        super().deserialize(data)
        self.set_inputs()

    def deserialize_inputs(self, data, nodes_by_name):
        super().deserialize_inputs(data, nodes_by_name)
        self._inputs = tuple(Input(nodes_by_name[nn], i)
                             for nn, i in data['inputs'])

    @property
    def inputs(self):
        return self._inputs

    def set_inputs(self, *inputs):
        """Set the inputs of this node. If no arguments are given, all existing
        inputs get disconnected.

        Args:
            *inputs (input_like): Inputs of this node. See
                :meth:`~libspn.Input.as_input` for possible inputs.
        """
        self._inputs = self._parse_inputs(*inputs)

    def add_inputs(self, *inputs):
        """Add more inputs to this node.

        Args:
            *inputs (input_like): Inputs of this node. See
                :meth:`~libspn.Input.as_input` for possible inputs.
        """
        self._inputs = self._inputs + self._parse_inputs(*inputs)

    @property
    def _const_out_size(self):
        return False

    @utils.docinherit(OpNode)
    def _compute_out_size(self, *input_out_sizes):
        if not self._inputs:
            raise StructureError("%s is missing inputs." % self)
        return sum(self._gather_input_sizes(*input_out_sizes))

    @utils.docinherit(OpNode)
    def _compute_scope(self, *input_scopes):
        if not self._inputs:
            raise StructureError("%s is missing inputs." % self)
        input_scopes = self._gather_input_scopes(*input_scopes)
        return list(chain.from_iterable(input_scopes))

    @utils.docinherit(OpNode)
    def _compute_valid(self, *input_scopes):
        if not self._inputs:
            raise StructureError("%s is missing inputs." % self)
        _, *input_scopes_ = self._gather_input_scopes(*input_scopes)
        # If already invalid, return None
        if any(s is None for s in input_scopes_):
            return None
        else:
            return self._compute_scope(*input_scopes)

    @utils.docinherit(OpNode)
    def _compute_value(self, *input_tensors):
        # Check inputs
        if not self._inputs:
            raise StructureError("%s is missing inputs." % self)
        @tf.custom_gradient
        def value_gradient(*input_tensors):
            def gradient(gradients):
                scattered_grads = self._compute_mpe_path(gradients, *input_tensors)
                return [sg for sg in scattered_grads if sg is not None]

            gathered_inputs = self._gather_input_tensors(*input_tensors)
            # Concatenate inputs
            return utils.concat_maybe(gathered_inputs, 1), gradient
        return value_gradient(*input_tensors)

    @utils.docinherit(OpNode)
    def _compute_log_value(self, *input_tensors):
        return self._compute_value(*input_tensors)

    @utils.docinherit(OpNode)
    def _compute_mpe_value(self, *input_tensors):
        return self._compute_value(*input_tensors)

    @utils.docinherit(OpNode)
    def _compute_log_mpe_value(self, *input_tensors):
        return self._compute_value(*input_tensors)

    def _compute_mpe_path(self, counts, *input_values, add_random=False,
                          use_unweighted=False):
        # Check inputs
        if not self._inputs:
            raise StructureError("%s is missing inputs." % self)
        # Split counts for each input
        input_sizes = self.get_input_sizes(*input_values)
        split = utils.split_maybe(counts, input_sizes, 1)
        return self._scatter_to_input_tensors(*[(t, v) for t, v in
                                                zip(split, input_values)])

    def _compute_log_mpe_path(self, counts, *value_values, add_random=False,
                              use_unweighted=False):
        return self._compute_mpe_path(counts, *value_values)
