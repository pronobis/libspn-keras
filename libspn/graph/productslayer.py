# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

from itertools import chain, combinations, repeat
import numpy as np
import tensorflow as tf
from libspn.graph.scope import Scope
from libspn.graph.node import OpNode, Input
from libspn.inference.type import InferenceType
from libspn import utils
from libspn.exceptions import StructureError
from libspn.log import get_logger
from libspn.utils.serialization import register_serializable


@register_serializable
class ProductsLayer(OpNode):
    """A node representing all products in a layer in an SPN.

    Args:
        *values (input_like): Inputs providing input values to this node.
            See :meth:`~libspn.Input.as_input` for possible values.
        num_or_size_prods (int or list of int):
            Int: Number of product ops modelled by this node. In which case, all
            the products modelled will have a common size.
            List: Size of each product op modelled by this node. Number of
            products modelled would be the length of the list.
        name (str): Name of the node.
    """

    logger = get_logger()
    info = logger.info

    def __init__(self, *values, num_or_size_prods=1, name="ProductsLayer"):
        self._values = []
        super().__init__(InferenceType.MARGINAL, name)
        self.set_values(*values)

        # Total size of value input_size
        total_values_size = sum(
            len(v.indices) if v and v.indices else v.node.get_out_size() if v else 0
            for v in self._values)

        if isinstance(num_or_size_prods, int):  # Total number of prodcut ops to be modelled
            if not num_or_size_prods > 0:
                raise StructureError("In %s 'num_or_size_prods': %s need to be > 0"
                                     % self, num_or_size_prods)
            self._num_prods = num_or_size_prods
            self._prod_input_sizes = [total_values_size // self._num_prods] * self._num_prods
        elif isinstance(num_or_size_prods, list):  # Size of each modelled product op
            if not len(num_or_size_prods) > 0:
                raise StructureError("In %s 'num_or_size_prods': %s cannot be an empty list"
                                     % self, num_or_size_prods)
            self._prod_input_sizes = num_or_size_prods
            self._num_prods = len(num_or_size_prods)

        self._num_or_size_prods = num_or_size_prods

    def serialize(self):
        data = super().serialize()
        data['values'] = [(i.node.name, i.indices) for i in self._values]
        data['num_or_size_prods'] = self._num_or_size_prods
        return data

    def deserialize(self, data):
        super().deserialize(data)
        self.set_values()
        self._num_or_size_prods = data['num_or_size_prods']

    def deserialize_inputs(self, data, nodes_by_name):
        super().deserialize_inputs(data, nodes_by_name)
        self._values = tuple(Input(nodes_by_name[nn], i)
                             for nn, i in data['values'])

    @property
    @utils.docinherit(OpNode)
    def inputs(self):
        return self._values

    @property
    def num_prods(self):
        """int: Number of Product ops modelled by this node."""
        return self._num_prods

    def set_num_prods(self, num_prods=1):
        """Set the number of Product ops modelled by this node.

        Args:
            num_prods (int): Number of Product ops modelled by this node.
        """
        self._num_prods = num_prods

    @property
    def num_or_size_prods(self):
        """int: Number of Product ops modelled by this node."""
        return self._num_or_size_prods

    def set_num_or_size_prods(self, num_or_size_prods=1):
        """Set the number of Product ops modelled by this node.

        Args:
            num_prods (int): Number of Product ops modelled by this node.
        """
        self._num_or_size_prods = num_or_size_prods

    @property
    def values(self):
        """list of Input: List of value inputs."""
        return self._values

    def set_values(self, *values):
        """Set the inputs providing input values to this node. If no arguments
        are given, all existing value inputs get disconnected.

        Args:
            *values (input_like): Inputs providing input values to this node.
                See :meth:`~libspn.Input.as_input` for possible values.
        """
        self._values = self._parse_inputs(*values)

    def add_values(self, *values):
        """Add more inputs providing input values to this node.

        Args:
            *values (input_like): Inputs providing input values to this node.
                See :meth:`~libspn.Input.as_input` for possible values.
        """
        self._values = self._values + self._parse_inputs(*values)

    @property
    def _const_out_size(self):
        return True

    def _compute_out_size(self, *input_out_sizes):
        return self._num_prods

    def _compute_scope(self, *value_scopes):
        if not self._values:
            raise StructureError("%s is missing input values." % self)
        value_scopes = list(chain.from_iterable(self._gather_input_scopes(
                                                *value_scopes)))
        sublist_size = int(len(value_scopes) / self._num_prods)
        # Divide gathered value scopes into sublists, one per modelled Product node.
        value_scopes_sublists = [value_scopes[i:i+sublist_size] for i in
                                 range(0, len(value_scopes), sublist_size)]
        return [Scope.merge_scopes(vs) for vs in value_scopes_sublists]

    def _compute_valid(self, *value_scopes):
        if not self._values:
            raise StructureError("%s is missing input values." % self)
        value_scopes_ = self._gather_input_scopes(*value_scopes)
        # If already invalid, return None
        if any(s is None for s in value_scopes_):
            return None
        # Check product decomposability
        flat_value_scopes = list(chain.from_iterable(value_scopes_))
        values_per_product = int(len(flat_value_scopes) / self._num_prods)
        sub_value_scopes = [flat_value_scopes[i:(i + values_per_product)] for i in
                            range(0, len(flat_value_scopes), values_per_product)]
        for scopes in sub_value_scopes:
            for s1, s2 in combinations(scopes, 2):
                if s1 & s2:
                    ProductsLayer.info("%s is not decomposable with input value scopes %s",
                                       self, flat_value_scopes)
                    return None
        return self._compute_scope(*value_scopes)

    def _combine_values_and_indices(self, value_tensors):
        """
        Concatenates input tensors and returns the nested indices that are required for gathering
        all sum inputs to a reducible set of columns
        """
        # Chose list instead of dict to maintain order
        unique_tensors = []
        unique_offsets = []

        combined_indices = []
        flat_col_indices = []
        flat_tensor_offsets = []

        tensor_offset = 0
        for value_inp, value_tensor in zip(self._values, value_tensors):
            # Get indices. If not there, will be [0, 1, ... , len-1]
            indices = value_inp.indices if value_inp.indices else \
                np.arange(value_inp.node.get_out_size()).tolist()
            flat_col_indices.append(indices)
            if value_tensor not in unique_tensors:
                # Add the tensor and offsets ot unique
                unique_tensors.append(value_tensor)
                unique_offsets.append(tensor_offset)
                # Add offsets
                flat_tensor_offsets.append([tensor_offset for _ in indices])
                tensor_offset += value_tensor.shape[1].value
            else:
                # Find offset from list
                offset = unique_offsets[unique_tensors.index(value_tensor)]
                # After this, no need to update tensor_offset, since the current value_tensor will
                # wasn't added to unique
                flat_tensor_offsets.append([offset for _ in indices])

        # Flatten the tensor offsets and column indices
        flat_tensor_offsets = np.asarray(list(chain(*flat_tensor_offsets)))
        flat_col_indices = np.asarray(list(chain(*flat_col_indices)))

        # Offset in flattened arrays
        offset = 0
        for size in self._prod_input_sizes:
            # Now indices can be found by adding up column indices and tensor offsets
            indices = flat_col_indices[offset:offset + size] + \
                      flat_tensor_offsets[offset:offset + size]

            # Combined indices contains an array for each reducible set of columns
            combined_indices.append(indices)
            offset += size

        return combined_indices, utils.concat_maybe(unique_tensors, 1)

    def _compute_value_common(self, *value_tensors, padding_value=0.0):
        """Common actions when computing value."""
        # Check inputs
        if not self._values:
            raise StructureError("%s is missing input values." % self)
        # Prepare values
        if self._num_prods > 1:
            indices, value_tensor = self._combine_values_and_indices(value_tensors)
            # Create a 3D tensor with dimensions [batch, num-prods, max-prod-input-sizes]
            # The last axis will have zeros or ones (for log or non-log) when the
            # prod-input-size < max-prod-input-sizes
            reducible_values = utils.gather_cols_3d(value_tensor, indices,
                                                    pad_elem=padding_value)
            return reducible_values
        else:
            # Gather input tensors
            value_tensors = self._gather_input_tensors(*value_tensors)
            return utils.concat_maybe(value_tensors, 1)

    def _compute_value(self, *value_tensors):
        values = self._compute_value_common(*value_tensors, padding_value=1.0)
        return tf.reduce_prod(values, axis=-1, keep_dims=(False if
                              self._num_prods > 1 else True))

    def _compute_log_value(self, *value_tensors):
        values = self._compute_value_common(*value_tensors, padding_value=0.0)
        return tf.reduce_sum(values, axis=-1, keep_dims=(False if
                             self._num_prods > 1 else True))

    def _compute_mpe_value(self, *value_tensors):
        return self._compute_value(*value_tensors)

    def _compute_log_mpe_value(self, *value_tensors):
        return self._compute_log_value(*value_tensors)

    def _compute_mpe_path(self, counts, *value_values, add_random=False, use_unweighted=False):
        # Check inputs
        if not self._values:
            raise StructureError("%s is missing input values." % self)

        value_sizes = self.get_input_sizes(*value_values)

        # (1-3) Tile counts of each prod based on prod-input-size, by gathering
        indices = list(chain.from_iterable([repeat(r, p_inp_size) for r, p_inp_size
                                           in zip(range(self._num_prods),
                                                  self._prod_input_sizes)]))
        gathered_counts = utils.gather_cols(counts, indices)

        # (4) Split gathered countes based on value_sizes
        value_counts = tf.split(gathered_counts, value_sizes, axis=1)
        counts_values_paired = [(v_count, v_value) for v_count, v_value in
                                zip(value_counts, value_values)]

        # (5) scatter_cols (num_inputs)
        return self._scatter_to_input_tensors(*counts_values_paired)

    def _compute_log_mpe_path(self, counts, *value_values, add_random=False, use_unweighted=False):
        return self._compute_mpe_path(counts, *value_values)
