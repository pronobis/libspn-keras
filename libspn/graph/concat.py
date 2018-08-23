# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

from itertools import chain
from libspn.graph.node import OpNode, Input
from libspn import utils
from libspn.inference.type import InferenceType
from libspn.exceptions import StructureError
from libspn.utils.serialization import register_serializable
from libspn.graph.convsum import ConvSum
from libspn.graph.localsum import LocalSum
from libspn.graph.convprod2d import ConvProd2D, _ConvProdNaive
import tensorflow as tf
import numpy as np


@register_serializable
class Concat(OpNode):
    """An op node that concatenates all inputs into a single output tensor.

    Args:
        *inputs: Inputs of this node. See :meth:`~libspn.Input.as_input` for
             possible values.
        name (str): Name of the node.
    """

    def __init__(self, *inputs, name="Concat", axis=1):
        super().__init__(InferenceType.MARGINAL, name)
        self.set_inputs(*inputs)
        self._axis = axis

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
        if self.is_spatial:
            input_shapes = self._gather_input_shapes()
            reshaped_scopes = [np.asarray(sc).reshape(s) for sc, s in
                               zip(input_scopes, input_shapes)]
            return np.concatenate(reshaped_scopes, axis=self._axis - 1).ravel().tolist()

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
        # Concatenate inputs
        input_tensors = self._gather_input_tensors(*input_tensors)
        input_shapes = self._gather_input_shapes()
        reshaped_tensors = [tf.reshape(t, (-1,) + s) for t, s in zip(input_tensors, input_shapes)]
        out = utils.concat_maybe(reshaped_tensors, axis=self._axis)
        if self.is_spatial:
            out = tf.reshape(out, (-1, int(np.prod(self.output_shape_spatial))))
        return out

    @property
    def output_shape_spatial(self):
        if self._axis != 3:
            raise AttributeError("Requested spatial output shape of a Concat node that is "
                                 "not spatial.")
        shapes = self._gather_input_shapes()
        concat_axis_sum = sum(s[self._axis - 1] for s in shapes)
        return shapes[0][:self._axis-1] + (concat_axis_sum,)
    
    def _gather_input_shapes(self):
        shapes = []
        for inp in self.inputs:
            if isinstance(inp.node, (ConvProd2D, _ConvProdNaive, ConvSum, LocalSum)):
                shapes.append(inp.node.output_shape_spatial)
            else:
                shapes.append((inp.node.get_out_size(),))
                
        if any(len(shapes[0]) != len(s) for s in shapes):
            raise StructureError("All shapes must be of same dimension, now have: {}".format(
                [len(s) for s in shapes]
            ))
        if any(shapes[0][:self._axis - 1] != s[:self._axis - 1] for s in shapes):
            raise StructureError("All non-concatenation axes must be identical.")
        return shapes
    
    def _num_channels_per_input(self):
        if not self.is_spatial:
            raise AttributeError("Requested number of channels per input while this Concat node "
                                 "is not spatial.")
        shapes = self._gather_input_shapes()
        return [s[self._axis - 1] for s in shapes]
    
    @utils.docinherit(OpNode)
    def _compute_log_value(self, *input_tensors):
        return self._compute_value(*input_tensors)

    @utils.docinherit(OpNode)
    def _compute_mpe_value(self, *input_tensors):
        return self._compute_value(*input_tensors)

    @utils.docinherit(OpNode)
    def _compute_log_mpe_value(self, *input_tensors):
        return self._compute_value(*input_tensors)
    
    @property
    def is_spatial(self):
        return self._axis == 3

    def _compute_mpe_path(self, counts, *input_values, add_random=False, use_unweighted=False):
        # Check inputs
        if not self._inputs:
            raise StructureError("%s is missing inputs." % self)
        # Split counts for each input
        input_sizes = self.get_input_sizes(*input_values)
        # input_shapes = self._gather_input_shapes()
        if self.is_spatial:
            input_shapes = self._gather_input_shapes()
            counts = tf.reshape(counts, (-1,) + self.output_shape_spatial)
            split = utils.split_maybe(counts, self._num_channels_per_input(), axis=self._axis)
            split = [tf.reshape(t, (-1, int(np.prod(s)))) for t, s in zip(split, input_shapes)]
        else:
            split = utils.split_maybe(counts, input_sizes, 1)
        return self._scatter_to_input_tensors(*[(t, v) for t, v in
                                                zip(split, input_values)])

    def _compute_log_mpe_path(self, counts, *value_values, add_random=False, use_unweighted=False):
        return self._compute_mpe_path(counts, *value_values)
