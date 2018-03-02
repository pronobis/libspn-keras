# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

import numpy as np
import tensorflow as tf
from collections import OrderedDict, defaultdict
from itertools import chain, combinations
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
        Concatenates input tensors and returns the nested indices that are
        required for gathering all product inputs to a reducible set of columns.
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

    def _collect_count_indices_per_input(self):
        """
        For every unique (input, index) pair in the node's values list, collects
        and returns all column-indices of the counts tensor, for which the unique
        pair is a child of.
        """
        # Create a list of each input, paired with all the indices assosiated
        # with it
        # Eg: self._values = [(A, [0, 2, 3]),
        #                     (B, 1),
        #                     (A, None),
        #                     (B, [1, 2])]
        # expanded_inputs_list = [(A, 0), (A, 2), (A, 3),
        #                         (B, 1),
        #                         (A, 0), (A, 1), (A, 2), (A, 3),
        #                         (B, 1), (B, 2)]
        expanded_inputs_list = []
        for inp in self._values:
            if inp.indices is None:
                for i in range(inp.node.get_out_size()):
                    expanded_inputs_list.append((inp.node, i))
            elif isinstance(inp.indices, list):
                for i in inp.indices:
                    expanded_inputs_list.append((inp.node, i))
            elif isinstance(inp.indices, int):
                expanded_inputs_list.append((inp.node, inp.indices))

        # Create a list grouping together all inputs to each product modelled
        # Eg: self._prod_input_sizes = [2, 3, 2, 1, 2]
        #     prod_inputs_lists = [[(A, 0), (A, 2)],        # Prod-0
        #                          [(A, 3), (B, 1),(A, 0)], # Prod-1
        #                          [(A, 1), (A, 2)],        # Prod-2
        #                          [(A, 3)],                # Prod-3
        #                          [(B, 1), (B, 2)]]        # Prod-4
        prod_input_sizes = np.cumsum(np.array(self._prod_input_sizes)).tolist()
        prod_input_sizes.insert(0, 0)
        prod_inputs_lists = [expanded_inputs_list[start:stop] for start, stop in
                             zip(prod_input_sizes[:-1], prod_input_sizes[1:])]

        # Create a dictionary with each unique input and index pair as it's key,
        # and a list of product-indices as the corresponding value
        # Eg: unique_inps_inds_dict = {(A, 0): [0, 1], # Prod-0 and  Prod-1
        #                              (A, 1): [2],    # Prod-2
        #                              (A, 2): [0, 2], # Prod-0 and  Prod-2
        #                              (A, 3): [1],    # Prod-1
        #                              (B, 1): [1, 4], # Prod-1 and  Prod-4
        #                              (B, 2): [4]}    # Prod-4
        unique_inps_inds = defaultdict(list)
        for idx, inps in enumerate(prod_inputs_lists):
            for inp in inps:
                unique_inps_inds[inp] += [idx]

        # Sort dictionary based on key - Sorting ensures avoiding scatter op when
        # the original inputs is passed without indices
        unique_inps_inds = OrderedDict(sorted(unique_inps_inds.items()))

        # Collect all product indices as a nested list of indices to gather from
        # counts tensor
        # Eg: gather_counts_indices = [[0, 1],
        #                              [2],
        #                              [0, 2],
        #                              [1],
        #                              [1, 4],
        #                              [4]]
        gather_counts_indices = [v for v in unique_inps_inds.values()]

        # Create an ordered dictionary of unique inputs to this node as key,
        # and a list of unique indices per input as the corresponding value
        # Eg: unique_inps = {A: [0, 1, 2, 3]
        #                    B: [1, 2]}
        unique_inps = OrderedDict()
        for inp, ind in unique_inps_inds.keys():
            unique_inps[inp] = []
        for inp, ind in unique_inps_inds.keys():
            unique_inps[inp] += [ind]

        return gather_counts_indices, unique_inps

    def _compute_mpe_path(self, counts, *value_values, add_random=False,
                          use_unweighted=False):
        # Check inputs
        if not self._values:
            raise StructureError("%s is missing input values." % self)

        # For each unique (input, index) pair in the values list, collect counts
        # index of all counts for which the pair is a child of
        gather_counts_indices, unique_inputs = self._collect_count_indices_per_input()

        # Gather columns from the counts tensor, per unique (input, index) pair
        reducible_values = utils.gather_cols_3d(counts, gather_counts_indices)

        # Sum gathered counts together per unique (input, index) pair
        summed_counts = tf.reduce_sum(reducible_values, axis=-1)

        # For each unique input in the values list, calculate the number of
        # unique indices
        unique_inp_sizes = [len(v) for v in unique_inputs.values()]

        # Split the summed-counts tensor per unique input, based on input-sizes
        unique_input_counts = tf.split(summed_counts, unique_inp_sizes, axis=1)

        # Scatter each unique-counts tensor to the respective input, only once
        # per unique input in the values list
        scattered_counts = [None] * len(self._values)
        for (node, inds), cnts in zip(unique_inputs.items(), unique_input_counts):
            for i, (inp, val) in enumerate(zip(self._values, value_values)):
                if inp.node == node:
                    scattered_counts[i] = utils.scatter_cols(
                        cnts, inds, int(val.get_shape()[0 if val.get_shape().ndims
                                                        == 1 else 1]))
                    break

        return scattered_counts

    def _compute_log_mpe_path(self, counts, *value_values, add_random=False,
                              use_unweighted=False):
        return self._compute_mpe_path(counts, *value_values)
