# ------------------------------------------------------------------------
# Copyright (C) 2016 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

from itertools import chain
import tensorflow as tf
from libspn.graph.scope import Scope
from libspn.graph.node import OpNode, Input
from libspn.inference.type import InferenceType
from libspn.graph.ivs import IVs
from libspn.graph.weights import Weights
from libspn import utils
from libspn.exceptions import StructureError
from libspn.log import get_logger
from libspn import conf
import functools
from libspn.utils.serialization import register_serializable
import numpy as np
from collections import OrderedDict, deque
import operator
import itertools


@register_serializable
class SumsLayer(OpNode):
    """A node representing multiple sums in an SPN, where each sum has it's own
    and seperate input.

    Args:
        *values (input_like): Inputs providing input values to this node.
            See :meth:`~libspn.Input.as_input` for possible values.
        num_sums (int): Number of Sum ops modelled by this node.
        weights (input_like): Input providing weights node to this sum node.
            See :meth:`~libspn.Input.as_input` for possible values. If set
            to ``None``, the input is disconnected.
        ivs (input_like): Input providing IVs of an explicit latent variable
            associated with this sum node. See :meth:`~libspn.Input.as_input`
            for possible values. If set to ``None``, the input is disconnected.
        name (str): Name of the node.

    Attributes:
        inference_type(InferenceType): Flag indicating the preferred inference
                                       type for this node that will be used
                                       during value calculation and learning.
                                       Can be changed at any time and will be
                                       used during the next inference/learning
                                       op generation.
    """

    logger = get_logger()
    info = logger.info

    def __init__(self, *values, n_sums_or_sizes=None, weights=None, ivs=None,
                 inference_type=InferenceType.MARGINAL, name="SumsLayer"):
        super().__init__(inference_type, name)

        self.set_values(*values)
        self.set_weights(weights)
        self.set_ivs(ivs)

        _sum_index_lengths = sum(
            len(v.indices) if v and v.indices else v.node.get_out_size() if v else 0
            for v in self._values)

        if isinstance(n_sums_or_sizes, int):
            # Check if we can evenly divide the selected value inputs over the sums being modeled
            if not _sum_index_lengths % n_sums_or_sizes == 0:
                raise StructureError("Cannot divide total number of value inputs ({}) over the "
                                     "requested  number of sums ({})."
                                     .format(_sum_index_lengths, n_sums_or_sizes))
            if n_sums_or_sizes == 0:
                raise ZeroDivisionError("Attempted to divide by zero. Please specify a "
                                        "non-zero number of sums.")
            self._sum_input_sizes = [_sum_index_lengths // n_sums_or_sizes] * n_sums_or_sizes
        elif isinstance(n_sums_or_sizes, list) and \
                all(isinstance(elem, int) for elem in n_sums_or_sizes):
            # A list of sum sizes is given
            self._sum_input_sizes = n_sums_or_sizes
            if n_sums_or_sizes and sum(n_sums_or_sizes) != _sum_index_lengths:
                raise StructureError(
                    "The specified total number of sums is incompatible with the value input "
                    "indices. \nTotal number of sums: {}, total indices in value inputs: "
                    "{}".format(sum(n_sums_or_sizes), _sum_index_lengths))
        elif n_sums_or_sizes is None:
            # Sum input sizes is set to size of each value input
            self._sum_input_sizes = [len(v.indices) if v.indices else v.node.get_out_size()
                                     for v in self._values]
        else:
            raise ValueError("The value of n_sums_or_sizes must be an int or a list of ints.")

        # Set the total number of sums being modeled
        self._num_sums = len(self._sum_input_sizes)

        # This flag is set for potential optimization of MPE path computation
        self._must_gather_for_mpe_path = False

    # TODO
    def serialize(self):
        data = super().serialize()
        data['values'] = [(i.node.name, i.indices) for i in self._values]
        data['sum_input_sizes'] = self._sum_input_sizes
        if self._weights:
            data['weights'] = (self._weights.node.name, self._weights.indices)
        if self._ivs:
            data['ivs'] = (self._ivs.node.name, self._ivs.indices)
        return data

    # TODO
    def deserialize(self, data):
        super().deserialize(data)
        self.set_values()
        self._sum_input_sizes = data['sum_input_sizes']
        self.set_weights()
        self.set_ivs()

    # TODO
    def deserialize_inputs(self, data, nodes_by_name):
        super().deserialize_inputs(data, nodes_by_name)
        self._values = tuple(Input(nodes_by_name[nn], i) for nn, i in data['values'])
        weights = data.get('weights', None)
        if weights:
            self._weights = Input(nodes_by_name[weights[0]], weights[1])
        ivs = data.get('ivs', None)
        if ivs:
            self._ivs = Input(nodes_by_name[ivs[0]], ivs[1])

    @property
    @utils.docinherit(OpNode)
    def inputs(self):
        return (self._weights, self._ivs) + self._values

    @property
    def weights(self):
        """Input: Weights input."""
        return self._weights

    def set_weights(self, weights=None):
        """Set the weights input.

        Args:
            weights (input_like): Input providing weights node to this sum node.
                See :meth:`~libspn.Input.as_input` for possible values. If set
                to ``None``, the input is disconnected.
        """
        weights, = self._parse_inputs(weights)
        if weights and not isinstance(weights.node, Weights):
            raise StructureError("%s is not Weights" % weights.node)
        self._weights = weights

    @property
    def ivs(self):
        """Input: IVs input."""
        return self._ivs

    def set_ivs(self, ivs=None):
        """Set the IVs input.

        ivs (input_like): Input providing IVs of an explicit latent variable
            associated with this sum node. See :meth:`~libspn.Input.as_input`
            for possible values. If set to ``None``, the input is disconnected.
        """
        self._ivs, = self._parse_inputs(ivs)

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

    def _build_mask(self):
        """
        Constructs mask that could be used to cancel out 'columns' that are padded as a result of
        varying reduction sizes. Returns a Boolean mask.
        """
        max_size = max(self._sum_input_sizes)
        sizes = np.asarray(self._sum_input_sizes).reshape((self._num_sums, 1))
        indices = np.arange(max_size).reshape((1, max_size))

        # Use broadcasting
        return indices < sizes

    def generate_weights(self, init_value=1, trainable=True,
                         input_sizes=None, name=None):
        """Generate a weights node matching this sum node and connect it to
        this sum.

        The function calculates the number of weights based on the number
        of input values of this sum. Therefore, weights should be generated
        once all inputs are added to this node.

        Args:
            init_value: Initial value of the weights. For possible values, see
                :meth:`~libspn.utils.broadcast_value`.
            trainable (bool): See :class:`~libspn.Weights`.
            input_sizes (list of int): Pre-computed sizes of each input of
                this node.  If given, this function will not traverse the graph
                to discover the sizes.
            name (str): Name of the weighs node. If ``None`` use the name of the
                        sum + ``_Weights``.

        Return:
            Weights: Generated weights node.
        """
        if not self._values:
            raise StructureError("%s is missing input values" % self)
        if name is None:
            name = self._name + "_Weights"
        # Set sum node sizes either from input or from inferred _sum_input_sizes
        sum_input_sizes = input_sizes or self._sum_input_sizes
        max_size = max(sum_input_sizes)
        sum_size = sum(sum_input_sizes)

        # Mask is used to select the indices to assign the value to, since the weights tensor can
        # be larger than the total number of weights being modeled due to padding
        mask = self._build_mask().reshape((-1,))

        init_padded_flat = np.zeros(self._num_sums * max_size)
        if isinstance(init_value, int) and init_value == 1:
            # If an int, just broadcast its value to the sum dimensions
            init_padded_flat[mask] = init_value
        elif hasattr(init_value, '__iter__'):
            # If the init value is iterable, check if number of elements matches number of
            init_flat = np.asarray(init_value).reshape((-1,))
            if init_flat.size == sum_size:
                init_padded_flat[mask] = init_flat
            else:
                raise ValueError("Incorrect initializer size {}, use an int or an iterable of size"
                                 " {}.".format(init_flat.size, sum_size))
        else:
            raise ValueError("Initialization value {} of type {} not usable, use an int or an "
                             "iterable of size {}."
                             .format(init_value, type(init_value), sum_size))
        # Generate weights
        init_value = init_padded_flat.reshape((self._num_sums, max_size))
        weights = Weights(init_value=init_value,
                          num_weights=max_size,
                          num_sums=len(sum_input_sizes),
                          trainable=trainable, name=name)
        self.set_weights(weights)
        return weights

    def generate_ivs(self, feed=None, name=None):
        """Generate an IVs node matching this sum node and connect it to
        this sum.

        IVs should be generated once all inputs are added to this node,
        otherwise the number of IVs will be incorrect.

        Args:
            feed (Tensor): See :class:`~libspn.IVs`.
            name (str): Name of the IVs node. If ``None`` use the name of the
                        sum + ``_IVs``.

        Return:
            IVs: Generated IVs node.
        """
        if not self._values:
            raise StructureError("%s is missing input values" % self)
        if name is None:
            name = self._name + "_IVs"

        ivs = IVs(feed=feed, num_vars=self._num_sums, num_vals=max(self._sum_input_sizes),
                  name=name)
        self.set_ivs(ivs)
        return ivs

    @property
    def _const_out_size(self):
        return True

    def _compute_out_size(self, *input_out_sizes):
        return self._num_sums

    def _compute_scope(self, weight_scopes, ivs_scopes, *value_scopes):
        if not self._values:
            raise StructureError("%s is missing input values" % self)
        _, ivs_scopes, *value_scopes = self._gather_input_scopes(weight_scopes,
                                                                 ivs_scopes,
                                                                 *value_scopes)
        flat_value_scopes = np.asarray(list(chain.from_iterable(value_scopes)))
        split_indices = np.cumsum(self._sum_input_sizes)[:-1]
        # Divide gathered value scopes into sublists, one per modelled Sum node
        value_scopes_sublists = [arr.tolist() for arr in
                                 np.split(flat_value_scopes, split_indices)]
        if self._ivs:
            # Divide gathered ivs scopes into sublists, one per modelled Sum node
            ivs_scopes_sublists = [arr.tolist() for arr in
                                   np.split(ivs_scopes, split_indices)]
            # Add respective ivs scope to value scope list of each Sum node
            for val, ivs in zip(value_scopes_sublists, ivs_scopes_sublists):
                val.extend(ivs)
        return [Scope.merge_scopes(val_scope) for val_scope in
                value_scopes_sublists]

    def _compute_valid(self, weight_scopes, ivs_scopes, *value_scopes):
        if not self._values:
            raise StructureError("%s is missing input values" % self)
        _, ivs_scopes_, *value_scopes_ = self._gather_input_scopes(weight_scopes,
                                                                   ivs_scopes,
                                                                   *value_scopes)
        # If already invalid, return None
        if (any(s is None for s in value_scopes_)
                or (self._ivs and ivs_scopes_ is None)):
            return None

        # Split the flat value scopes based on value input sizes
        split_indices = np.cumsum(self._sum_input_sizes)[:-1]
        flat_value_scopes = np.asarray(list(chain.from_iterable(value_scopes_)))

        # IVs
        if self._ivs:
            # Verify number of IVs
            if len(ivs_scopes_) != len(flat_value_scopes):
                raise StructureError("Number of IVs (%s) and values (%s) does "
                                     "not match for %s"
                                     % (len(ivs_scopes_), len(flat_value_scopes),
                                        self))

            # Go over IVs involved for each sum. Scope size should be exactly one
            for iv_scopes_for_sum in np.split(ivs_scopes_, split_indices):
                if len(Scope.merge_scopes(iv_scopes_for_sum)) != 1:
                    return None

        # Go over value input scopes for each sum being modeled. Within a single sum, the scope of
        # all the inputs should be the same
        for scope_slice in np.split(flat_value_scopes, split_indices):
            first_scope = scope_slice[0]
            if any(s != first_scope for s in scope_slice[1:]):
                SumsLayer.info("%s is not complete with input value scopes %s",
                               self, flat_value_scopes)
                return None

        return self._compute_scope(weight_scopes, ivs_scopes, *value_scopes)

    def _combine_values_and_indices(self, value_tensors):
        """
        Concatenates input tensors and returns the nested indices that are required for gathering
        all sum inputs to a reducible set of columns
        """
        # Get flattened column indices and tensor offsets. The tensor offsets are indicating at
        # which index on axis 1 the tensors will end up in the concatenation of the unique tensors
        flat_col_indices, flat_tensor_offsets, unique_tensors_offsets_dict = \
            self._flat_indices_offsets_and_unique_tensors(value_tensors)

        # Offset in flattened arrays
        offset = 0
        max_size = max(self._sum_input_sizes)
        nested_multi_sum_indices = []
        for size in self._sum_input_sizes:
            # Now indices can be found by adding up column indices and tensor offsets
            single_sum_indices = flat_col_indices[offset:offset + size] + \
                      flat_tensor_offsets[offset:offset + size]
            # If there is padding within a single tensor, we have to perform an additional gather
            # step when computing the MPE path
            if size < max_size:
                self._must_gather_for_mpe_path = True
            # Combined indices contains an array for each reducible set of columns
            nested_multi_sum_indices.append(single_sum_indices)
            offset += size

        # Concatenate the unique tensors
        unique_tensors = list(unique_tensors_offsets_dict.keys())
        return nested_multi_sum_indices, utils.concat_maybe(unique_tensors, axis=1)

    def _flat_indices_offsets_and_unique_tensors(self, value_tensors):
        # Ordered dict since we want the offsets per tensor, but we also want the order of
        # occurrence for concatenation later
        unique_tensors_offsets_dict = OrderedDict()
        # Initialize lists to hold column indices and tensor indices
        column_indices = []
        tensor_offsets = []
        tensor_offset = 0

        for value_inp, value_tensor in zip(self._values, value_tensors):
            # Get indices. If not there, will be [0, 1, ... , len-1]
            indices = value_inp.indices if value_inp.indices else \
                np.arange(value_inp.node.get_out_size()).tolist()
            column_indices.extend(indices)
            if value_tensor not in unique_tensors_offsets_dict:
                # Add the tensor and offsets ot unique
                unique_tensors_offsets_dict[value_tensor] = tensor_offset
                # Add offsets
                tensor_offsets.extend([tensor_offset for _ in indices])
                tensor_offset += value_tensor.shape[1].value
            else:
                # Find offset from dict
                offset = unique_tensors_offsets_dict[value_tensor]
                # After this, no need to update tensor_offset, since the current value_tensor
                # wasn't added to unique
                tensor_offsets.extend([offset for _ in indices])

        # Flatten the tensor offsets and column indices
        flat_tensor_offsets = np.asarray(tensor_offsets)
        flat_col_indices = np.asarray(column_indices)
        return flat_col_indices, flat_tensor_offsets, unique_tensors_offsets_dict

    def _compute_value_common(self, cwise_op, reduction_fn, weight_tensor, ivs_tensor,
                              *value_tensors, weighted=True):
        """ Common actions when computing value. """
        # Check inputs
        if not self._values:
            raise StructureError("%s is missing input values" % self)
        if not self._weights:
            raise StructureError("%s is missing weights" % self)
        # Prepare values
        weight_tensor, ivs_tensor = self._gather_input_tensors(weight_tensor, ivs_tensor)

        # Builds reducible value tensor
        reducible_values = self._reducible_values(value_tensors)

        if weighted:
            # Use component wise op for weighting
            reducible_values = cwise_op(reducible_values, weight_tensor)
        if self._ivs:
            # Reshape IVs and apply them component-wise
            iv_reshape = (-1, self._num_sums, max(self._sum_input_sizes))
            ivs_tensor_reshaped = tf.reshape(ivs_tensor, iv_reshape)
            reducible_values = cwise_op(reducible_values, ivs_tensor_reshaped)

        # Reduce on last axis
        return reduction_fn(reducible_values)

    @functools.lru_cache()
    def _reducible_values(self, value_tensors):
        indices, values = self._combine_values_and_indices(value_tensors)
        # Create a 3D tensor with dimensions [batch, sum node, sum input]
        # The last axis will have zeros when the sum size is less than the max sum size
        if all(np.array_equal(indices[0], ind) for ind in indices):
            # In case all sum nodes model the same sum, we can just use broadcasting
            reducible_values = tf.reshape(utils.gather_cols(values, indices[0]),
                                          (-1, 1, self._sum_input_sizes[0]))
        elif len(set(self._sum_input_sizes)) == 1:
            # In case all sum sizes are the same, use gather and reshape accordingly
            indices_flat = list(itertools.chain(*indices))
            reducible_values = tf.reshape(utils.gather_cols(values, indices_flat),
                                          (-1, self._num_sums, self._sum_input_sizes[0]))
        else:
            reducible_values = utils.gather_cols_3d(values, indices, name="GatherToReducible")
        return reducible_values

    def _compute_value(self, weight_tensor, ivs_tensor, *value_tensors):
        """ Computes value in non-log space """
        reduce_sum = functools.partial(tf.reduce_sum, axis=2)
        return self._compute_value_common(
            tf.multiply, reduce_sum, weight_tensor, ivs_tensor, *value_tensors)

    def _compute_log_value(self, weight_tensor, ivs_tensor, *value_tensors):
        """ Computes value in log space """
        reduce_logsum = functools.partial(utils.reduce_log_sum_3D, transpose=False)
        return self._compute_value_common(
            tf.add, reduce_logsum, weight_tensor, ivs_tensor, *value_tensors)

    def _compute_mpe_value(self, weight_tensor, ivs_tensor, *value_tensors):
        """ Computes MPE value in non-log space """
        reduce_max = functools.partial(tf.reduce_max, axis=2)
        return self._compute_value_common(
            tf.multiply, reduce_max, weight_tensor, ivs_tensor, *value_tensors)

    def _compute_log_mpe_value(self, weight_tensor, ivs_tensor, *value_tensors):
        """ Computes MPE value in log space """
        reduce_max = functools.partial(tf.reduce_max, axis=2)
        return self._compute_value_common(
            tf.add, reduce_max, weight_tensor, ivs_tensor, *value_tensors)

    def _compute_mpe_path_common(self, values_weighted, counts, weight_value,
                                 ivs_value, *value_values, sum_counts=True):
        """ Common operations for log and non-log MPE path """
        # Propagate the counts to the max value
        max_indices = tf.argmax(values_weighted, dimension=2)

        max_counts = utils.scatter_values(params=counts, indices=max_indices,
                                          num_out_cols=values_weighted.shape[2].value)

        _, _, *value_sizes = self.get_input_sizes(None, None, *value_values)

        # Reshape max counts to a wide 2D tensor of shape 'Batch X (num_sums * max_size)'
        max_size = max(self._sum_input_sizes)
        reshape = (-1, self._num_sums * max_size)

        max_counts_reshaped = tf.reshape(max_counts, shape=reshape)

        if conf.add_counts_in_sums_layer:
            # Get the flat indices for the columns for each input and the tensor offsets in the
            # concatenation of the unique tensors
            flat_col_indices, flat_tensor_offsets, unique_tensors_offsets_dict = \
                self._flat_indices_offsets_and_unique_tensors(value_values)
            unique_tensors = list(unique_tensors_offsets_dict.keys())
            if len(unique_tensors) < len(value_values):
                # Compute mask for selecting the non-zero rows in sparse matrix
                padding_mask = []
                for i, size in enumerate(self._sum_input_sizes):
                    padding_mask.append(np.ones(size, dtype=np.int64))
                    if max_size - size > 0:
                        sub_mask = np.zeros(max_size - size, dtype=np.int64)
                        padding_mask.append(sub_mask)
                flat_padding_mask = np.concatenate(padding_mask).astype(np.bool)

                # Determine sparse indices
                rows = np.arange(flat_padding_mask.size)[flat_padding_mask]
                columns = flat_col_indices + flat_tensor_offsets
                # Sort rows and columns in row-major order
                rows, columns = zip(*sorted(zip(rows, columns), key=operator.itemgetter(0, 1)))

                # Determine dense shape
                unique_tensor_sizes = [tensor.shape[1].value for tensor in unique_tensors]
                dense_shape = (flat_padding_mask.size, sum(unique_tensor_sizes))

                # In this matrix, *zero* rows are aligned with padded columns in
                # max_counts_reshaped. Some columns will have multiple non-zero elements, in which
                # case we sum the columns from max_counts_reshaped that correspond to the
                # *rows* of these non-zero elements
                sum_counts_mat = np.zeros(dense_shape, dtype=np.float32)
                sum_counts_mat[rows, columns] = np.ones(flat_col_indices.size)

                # Sum the counts per unique input tensor using the matrix that we just constructed
                max_counts_reshaped = tf.matmul(max_counts_reshaped, sum_counts_mat)

                # Potentially split the result
                max_counts_split = tf.split(max_counts_reshaped, unique_tensor_sizes, axis=1) \
                    if len(unique_tensors) > 1 else [max_counts_reshaped]

                # In this case, we already 'scattered' the counts, so we only do it explicitly for
                # our weights and IVs. Currently, the algorithms for up and down traversal of the
                # graph expect a counts tensor coming from each parent when computing the MPE path.
                # The code below is part of a hacky workaround by putting 'None's for any
                max_counts_split_with_None = []
                max_counts_split = deque(max_counts_split)
                unique_tensors = deque(unique_tensors)
                next_tensor = unique_tensors.popleft()
                for tensor in value_values:
                    if tensor == next_tensor:
                        max_counts_split_with_None.append(max_counts_split.popleft())
                        if unique_tensors:
                            next_tensor = unique_tensors.popleft()
                        else:
                            next_tensor = None
                    else:
                        max_counts_split_with_None.append(None)

                return self._scatter_to_input_tensors(
                    (max_counts, weight_value),  # Weights
                    (max_counts_reshaped, ivs_value),  # IVs
                ) + tuple(max_counts_split_with_None)

        # This flag is set to True if we have had to pad within an input
        if self._must_gather_for_mpe_path:
            # Will hold indices to gather
            indices = []
            offset = 0
            for size in self._sum_input_sizes:
                indices.extend([offset + i for i in range(size)])
                offset += max_size
            # Gather so that padded parts are left out
            max_counts_reshaped = utils.gather_cols(max_counts_reshaped, indices)

        # TODO we should be able to split the tensor without having to gather and ignore 'padded'
        # parts if the padding always occurred between two tensors and never within a single tensor

        # Split the reshaped max counts to value inputs
        max_counts_split = tf.split(max_counts_reshaped, value_sizes, 1)
        return self._scatter_to_input_tensors(
            (max_counts, weight_value),  # Weights
            (max_counts_reshaped, ivs_value),  # IVs
            *[(t, v) for t, v in zip(max_counts_split, value_values)])  # Values

    def _compute_mpe_path(self, counts, weight_value, ivs_value, *value_values,
                          add_random=None, use_unweighted=False, sum_counts=True):
        values_selected_weighted = self._compute_value_common(
            tf.multiply, lambda x: x, weight_value, ivs_value, *value_values)
        return self._compute_mpe_path_common(values_selected_weighted, counts,
                                             weight_value, ivs_value, *value_values,
                                             sum_counts=sum_counts)

    def _compute_log_mpe_path(self, counts, weight_value, ivs_value,
                              *value_values, add_random=None,
                              use_unweighted=False, sum_counts=True):
        # Get weighted, IV selected values
        values_reshaped = self._compute_value_common(
            tf.add, lambda x: x, weight_value, ivs_value, *value_values, weighted=False)

        # WARN USING UNWEIGHTED VALUE
        if not use_unweighted or any(v.node.is_var for v in self._values):
            values_weighted = values_reshaped + weight_value
        else:
            values_weighted = values_reshaped
        # / USING UNWEIGHTED VALUE

        # WARN ADDING RANDOM NUMBERS
        if add_random is not None:
            values_weighted = tf.add(values_weighted, tf.random_uniform(
                shape=(tf.shape(values_weighted)[0], 1,
                       values_weighted.shape[2].value),
                minval=0, maxval=add_random,
                dtype=conf.dtype))
        # /ADDING RANDOM NUMBERS

        return self._compute_mpe_path_common(
            values_weighted, counts, weight_value, ivs_value, *value_values,
            sum_counts=sum_counts)
