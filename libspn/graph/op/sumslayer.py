# ------------------------------------------------------------------------
# Copyright (C) 2016 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------
import tensorflow as tf
from libspn.graph.scope import Scope
from libspn.inference.type import InferenceType
from libspn.graph.weights import Weights
from libspn.graph.op.base_sum import BaseSum
from libspn import utils
from libspn.exceptions import StructureError
from libspn import conf
import numpy as np
from collections import OrderedDict, deque, defaultdict
import itertools


@utils.register_serializable
class SumsLayer(BaseSum):
    """A node representing multiple sums in an SPN, where each sum possibly has its own
    and separate input and the sums that are modeled can have differently sized inputs.

    Args:
        *values (input_like): Inputs providing input values to this node.
            See :meth:`~libspn.Input.as_input` for possible values.
        num_or_size_sums (int or list of ints): Number of Sum ops modelled by
            this node or the size of each sum in case of a list. Default is None.
            If None, it will compute one sum per input. If int, it will attempt
            to construct num_or_size_sums sums, each of size
            total_input_size // num_or_size_sums. If a list of ints, it will
            construct a sum for each size given in the list.
        weights (input_like): Input providing weights node to this sum node.
            See :meth:`~libspn.Input.as_input` for possible values. If set
            to ``None``, the input is disconnected.
        latent_indicators (input_like): Input providing IndicatorLeaf of an explicit latent variable
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

    def __init__(self, *values, num_or_size_sums=None, weights=None, latent_indicators=None,
                 inference_type=InferenceType.MARGINAL, sample_prob=None,
                 name="SumsLayer"):
        if isinstance(num_or_size_sums, int) or num_or_size_sums is None:
            num_sums = num_or_size_sums
            sum_sizes = None
        else:
            num_sums = len(num_or_size_sums)
            sum_sizes = num_or_size_sums
        super().__init__(
            *values, num_sums=num_sums, sum_sizes=sum_sizes, weights=weights,
            latent_indicators=latent_indicators, inference_type=inference_type,
            sample_prob=sample_prob, name=name, masked=True)

    @property
    def is_layer(self):
        return True

    @utils.docinherit(BaseSum)
    def _reset_sum_sizes(self, num_sums=None, sum_sizes=None):
        _sum_index_lengths = sum(
            len(v.indices) if v and v.indices else v.node.get_out_size() if v else 0
            for v in self._values)
        if num_sums or sum_sizes:
            if sum_sizes and all(isinstance(elem, int) for elem in sum_sizes):
                # A list of sum sizes is given
                if sum_sizes and sum(sum_sizes) != _sum_index_lengths:
                    raise StructureError(
                        "The specified total number of sums is incompatible with the value input "
                        "indices. \nTotal number of sums: {}, total indices in value inputs: "
                        "{}".format(sum(sum_sizes), _sum_index_lengths))
            elif not sum_sizes:
                # Check if we can evenly divide the value inputs over the sums being modeled
                if not _sum_index_lengths % num_sums == 0:
                    raise StructureError("Cannot divide total number of value inputs ({}) over the "
                                         "requested  number of sums ({})."
                                         .format(_sum_index_lengths, num_sums))
                if num_sums == 0:
                    raise ZeroDivisionError("Attempted to divide by zero. Please specify a "
                                            "non-zero number of sums.")
                sum_sizes = [_sum_index_lengths // num_sums] * num_sums
        else:
            # Sum input sizes is set to size of each value input
            sum_sizes = [len(v.indices) if v.indices else v.node.get_out_size()
                         for v in self._values]
        self._num_sums = len(sum_sizes)
        self._sum_sizes = sum_sizes
        self._max_sum_size = max(sum_sizes) if sum_sizes else 0

    def set_sum_sizes(self, sizes):
        """
        Sets the sum sizes. The sum of the sizes given should match the total number of inputs.

        Args:
            sizes (list): A ``list`` of ints corresponding to the sizes of the sums.
        """
        self._reset_sum_sizes(sum_sizes=sizes)

    @utils.docinherit(BaseSum)
    def _compute_scope(self, weight_scopes, latent_indicators_scopes, *value_scopes):
        flat_value_scopes, latent_indicators_scopes, *value_scopes = self._get_flat_value_scopes(
            weight_scopes, latent_indicators_scopes, *value_scopes)
        split_indices = np.cumsum(self._sum_sizes)[:-1]
        # Divide gathered value scopes into sublists, one per modelled Sum node
        value_scopes_sublists = [arr.tolist() for arr in
                                 np.split(flat_value_scopes, split_indices)]
        if self._latent_indicators:
            # Divide gathered latent_indicators scopes into sublists, one per modelled Sum node
            latent_indicators_scopes_sublists = [arr.tolist() for arr in
                                   np.split(latent_indicators_scopes, split_indices)]
            # Add respective latent_indicators scope to value scope list of each Sum node
            for val, latent_indicators in zip(
                    value_scopes_sublists, latent_indicators_scopes_sublists):
                val.extend(latent_indicators)
        return [Scope.merge_scopes(val_scope) for val_scope in
                value_scopes_sublists]

    @utils.docinherit(BaseSum)
    def _compute_valid(self, weight_scopes, latent_indicators_scopes, *value_scopes):
        flat_value_scopes, latent_indicators_scopes_, *value_scopes_ = self._get_flat_value_scopes(
            weight_scopes, latent_indicators_scopes, *value_scopes)
        # If already invalid, return None
        if (any(s is None for s in value_scopes_)
                or (self._latent_indicators and latent_indicators_scopes_ is None)):
            return None

        # Split the flat value scopes based on value input sizes
        split_indices = np.cumsum(self._sum_sizes)[:-1]

        # IndicatorLeaf
        if self._latent_indicators:
            # Verify number of IndicatorLeaf
            if len(latent_indicators_scopes_) != len(flat_value_scopes):
                raise StructureError("Number of IndicatorLeaf (%s) and values (%s) does "
                                     "not match for %s"
                                     % (len(latent_indicators_scopes_), len(flat_value_scopes),
                                        self))

            # Go over IndicatorLeaf involved for each sum. Scope size should be exactly one
            for iv_scopes_for_sum in np.split(latent_indicators_scopes_, split_indices):
                if len(Scope.merge_scopes(iv_scopes_for_sum)) != 1:
                    return None

        # Go over value input scopes for each sum being modeled. Within a single sum, the scope of
        # all the inputs should be the same
        for scope_slice in np.split(flat_value_scopes, split_indices):
            first_scope = scope_slice[0]
            if any(s != first_scope for s in scope_slice[1:]):
                self.info("%s is not complete with input value scopes %s", self, flat_value_scopes)
                return None

        return self._compute_scope(weight_scopes, latent_indicators_scopes, *value_scopes)

    def _build_mask(self):
        """Constructs mask that could be used to cancel out 'columns' that are padded as a result of
        varying reduction sizes. Returns a Boolean mask.

        Returns:
            A ``numpy.ndarray`` with ``np.bool``s indicating the mask to applied to the weights.
        """
        sizes = np.asarray(self._sum_sizes).reshape((self._num_sums, 1))
        indices = np.arange(self._max_sum_size).reshape((1, self._max_sum_size))
        return np.less(indices, sizes)  # Use broadcasting

    def generate_weights(self, initializer=tf.initializers.constant(1.0), trainable=True,
                         input_sizes=None, log=False, name=None):
        """Generate a weights node matching this sum node and connect it to
        this sum.

        The function calculates the number of weights based on the number
        of input values of this sum. Therefore, weights should be generated
        once all inputs are added to this node.

        Args:
            initializer: Initial value of the weights.
            trainable (bool): See :class:`~libspn.Weights`.
            input_sizes (list of int): Pre-computed sizes of each input of
                this node.  If given, this function will not traverse the graph
                to discover the sizes.
            log (bool): If "True", the weights are represented in log space.
            name (str): Name of the weighs node. If ``None`` use the name of the
                        sum + ``_Weights``.

        Return:
            Weights: Generated weights node.
        """
        if not self._values:
            raise StructureError("%s is missing input values" % self)
        if name is None:
            name = self._name + "_Weights"

        # Set sum node sizes to inferred _sum_input_sizes
        sum_input_sizes = self._sum_sizes
        max_size = self._max_sum_size

        # Mask is used to select the indices to assign the value to, since the weights tensor can
        # be larger than the total number of weights being modeled due to padding
        mask = self._build_mask().reshape((-1,))

        # Generate weights
        weights = Weights(initializer=initializer, num_weights=max_size,
                          num_sums=len(sum_input_sizes), log=log,
                          trainable=trainable, mask=mask.tolist(), name=name)
        self.set_weights(weights)
        return weights

    def _combine_values_and_indices(self, value_tensors):
        """Concatenates input tensors and returns the nested indices that are required for gathering
        all sum inputs to a reducible set of columns.

        Args:
            value_tensors (list): A list of ``Tensor``s of value inputs connected to this node
                with potential duplicates.

        Returns:
            A nested ``list`` of indices to gather from the concatenation of the unique
            ``Tensors``s in ``value_tensors``. The concatenation is the second return value.
        """
        # Get flattened column indices and tensor offsets. The tensor offsets are indicating at
        # which index on axis 1 the tensors will end up in the concatenation of the unique tensors
        flat_col_indices, flat_tensor_offsets, unique_tensors_offsets_dict = \
            self._flat_indices_and_uniq_tensors(value_tensors)
        split_indices = np.cumsum(self._sum_sizes)[:-1]
        nested_multi_sum_indices = np.split(flat_col_indices + flat_tensor_offsets, split_indices)
        # Concatenate the unique tensors
        unique_tensors = list(unique_tensors_offsets_dict.keys())
        return nested_multi_sum_indices, tf.concat(unique_tensors, axis=self._op_axis)

    def _flat_indices_and_uniq_tensors(self, value_tensors):
        """Determines the flattened column indices to gather from the concatenated unique value
        tensors as well as the flattened value tensor offsets corresponding to the concatenation
        of the unique value tensors.

        Args:
            value_tensors (list): A ``list`` of ``Tensor``s corresponding to the output values of
                other nodes connected to this node which potentially have duplicates.

        Returns:
            An array of flat columns from the concatenated unique values, an array of flat
            tensor offsets for the concatenated unique values and a list of unique values.
        """
        unique_tensors = list(OrderedDict.fromkeys(value_tensors))
        tensor_offsets = np.cumsum([0] + [t.shape[1].value for t in unique_tensors[:-1]])

        # Initialize flat column indices
        flat_col_indices = []
        flat_tensor_offsets = []
        for value_inp, value_tensor in zip(self._values, value_tensors):
            # Get index of current tensor
            tensor_index = unique_tensors.index(value_tensor)

            # Get indices. If not there, will be [0, 1, ... , len-1]
            indices = value_inp.indices if value_inp.indices else \
                np.arange(value_inp.node.get_out_size()).tolist()
            flat_col_indices.extend(indices)
            flat_tensor_offsets.extend([tensor_offsets[tensor_index]] * len(indices))

        # Flatten the tensor offsets and column indices
        flat_tensor_offsets = np.asarray(flat_tensor_offsets)
        flat_col_indices = np.asarray(flat_col_indices)
        unique_tensors_offsets_dict = OrderedDict(zip(unique_tensors, tensor_offsets))
        return flat_col_indices, flat_tensor_offsets, unique_tensors_offsets_dict

    @utils.docinherit(BaseSum)
    @utils.lru_cache
    def _prepare_component_wise_processing(
            self, w_tensor, latent_indicators_tensor, *input_tensors, zero_prob_val=0.0):
        indices, values = self._combine_values_and_indices(input_tensors)
        # Create a 3D tensor with dimensions [batch, sum node, sum input]
        # The last axis will have zeros when the sum size is less than the max sum size
        if all(np.array_equal(indices[0], ind) for ind in indices):
            # In case all sum nodes model the same sum, we can just use broadcasting
            reducible_values = tf.reshape(
                tf.gather(values, indices[0], axis=1), (-1, 1, self._max_sum_size))
        elif len(set(self._sum_sizes)) == 1:
            # In case all sum sizes are the same, use gather and reshape accordingly
            indices_flat = list(itertools.chain(*indices))
            reducible_values = tf.reshape(tf.gather(values, indices_flat, axis=1),
                                          (-1, self._num_sums, self._max_sum_size))
        else:
            reducible_values = utils.gather_cols_3d(
                values, indices, pad_elem=zero_prob_val, name="GatherToReducible")
        w_tensor = tf.expand_dims(w_tensor, axis=self._batch_axis)
        if latent_indicators_tensor is not None:
            latent_indicators_tensor = tf.reshape(
                latent_indicators_tensor, shape=(-1, self._num_sums, self._max_sum_size))
        return w_tensor, latent_indicators_tensor, reducible_values

    @utils.docinherit(BaseSum)
    @utils.lru_cache
    def _compute_mpe_path_common(
            self, reducible_tensor, counts, w_tensor, latent_indicators_tensor, *input_tensors,
            accumulate_weights_batch=False, sample=False, sample_prob=None):
        sample_prob = utils.maybe_first(sample_prob, self._sample_prob)
        num_samples = 1 if reducible_tensor.shape[1] != 1 else self._num_sums
        if sample:
            max_indices = self._reduce_sample_log(reducible_tensor, sample_prob=sample_prob,
                                                  num_samples=num_samples)
        else:
            max_indices = self._reduce_argmax(reducible_tensor, num_samples=num_samples)
        max_counts = utils.scatter_values(
            params=counts, indices=max_indices, num_out_cols=self._max_sum_size)
        max_counts_split = self._accumulate_and_split_to_children(max_counts, *input_tensors)
        if accumulate_weights_batch:
            w_counts = tf.reduce_sum(max_counts, axis=self._batch_axis)
        else:
            w_counts = max_counts

        return self._scatter_to_input_tensors(
            (w_counts, w_tensor),  # Weights
            (max_counts, latent_indicators_tensor)
        ) + tuple(max_counts_split)

    @utils.docinherit(BaseSum)
    @utils.lru_cache
    def _compute_log_gradient(self, gradients, w_tensor, latent_indicators_tensor, *value_tensors,
                              accumulate_weights_batch=False):
        reducible = self._compute_reducible(w_tensor, latent_indicators_tensor, *value_tensors)
        log_sum = tf.expand_dims(
            self._reduce_marginal_inference_log(reducible), axis=self._reduce_axis)

        # A number - (-inf) is undefined. In fact, the gradient in those cases should be zero
        log_sum = tf.where(tf.is_inf(log_sum), tf.zeros_like(log_sum), log_sum)
        w_grad = tf.expand_dims(gradients, axis=self._reduce_axis) * tf.exp(
            reducible - log_sum)
        inp_grad_split = self._accumulate_and_split_to_children(w_grad, *value_tensors)
        latent_indicators_grads = w_grad
        if accumulate_weights_batch:
            w_grad = tf.reduce_sum(w_grad, axis=self._batch_axis)

        return self._scatter_to_input_tensors(
            (w_grad, w_tensor),
            (latent_indicators_grads, latent_indicators_tensor)
        ) + tuple(inp_grad_split)

    @utils.docinherit(BaseSum)
    @utils.lru_cache
    def _get_differentiable_inputs(self, w_tensor, latent_indicators_tensor, *value_tensors):
        unique_tensors = list(OrderedDict.fromkeys(value_tensors))
        return [w_tensor] + (
            [latent_indicators_tensor] if self._latent_indicators else []) + unique_tensors

    @utils.docinherit(BaseSum)
    @utils.lru_cache
    def _accumulate_and_split_to_children(self, x, *input_tensors):
        flat_col_indices, flat_tensor_offsets, unique_tensors_offsets_dict = \
            self._flat_indices_and_uniq_tensors(input_tensors)
        # In this case we gather, sum by reducing and finally create scatterable tensors with
        # corresponding indices
        x = tf.reshape(x, (-1, self._num_sums * self._max_sum_size))

        segmented = conf.sumslayer_count_sum_strategy == 'segmented'
        tensor_to_scatter_indices, unique_input_counts = self._accumulate_uniq_values_and_split(
            flat_col_indices, flat_tensor_offsets, x, unique_tensors_offsets_dict,
            gather_segments_only=segmented)
        # Assign the splits to the right index in the output tuple
        max_counts_split_with_None = []
        max_counts_split = deque(unique_input_counts)
        unique_tensors = deque(unique_tensors_offsets_dict.keys())
        next_tensor = unique_tensors.popleft()
        for tensor in input_tensors:
            if tensor == next_tensor:
                cnts = max_counts_split.popleft()
                # Scatter the counts
                scattered = utils.scatter_cols(
                    cnts, tensor_to_scatter_indices[tensor], tensor.shape[1].value)
                max_counts_split_with_None.append(scattered)
                if unique_tensors:
                    next_tensor = unique_tensors.popleft()
                else:
                    next_tensor = None
            else:
                max_counts_split_with_None.append(None)
        return max_counts_split_with_None

    def _accumulate_uniq_values_and_split(
            self, flat_col_indices, flat_tensor_offsets, x, unique_tensors_offsets_dict,
            gather_segments_only=False):
        """Helper method that is used for summing counts within the layer before passing it on
        by means of gathering from the (padded) weighted values and reducing afterwards.

        Args:
            flat_col_indices (numpy.ndarray): An array containing the flattened column indices to
                gather from the concatenation of unqiue value tensors.
            flat_tensor_offsets (numpy.ndarray): An array containing the flattened tensor offsets
                in the concatenation of the unique value tensors.
            x (Tensor): A ``Tensor`` to gather, accumulate per unique value tensor and finally
                split for scattering.
            unique_tensors_offsets_dict (OrderedDict): A mapping of ``Tensor`` -> offset
                corresponding to the unique value tensors and their offsets in the concatenation.
            gather_segments_only (bool): If ``True``, will transpose and gather on the zeroth
                axis, without 'zero-probability' padding so that the result can be accumulated
                using tf.segment_sum.

        Returns:
            A list of indices to be used for scattering the values of the list in the second
            return value, which is a list of accumulated values corresponding to the unique
            value Inputs of this node.
        """
        # Make a flat list containing the sum index for each of the 'concatenated' inputs
        sum_indices = []
        for i, size in enumerate(self._sum_sizes):
            sum_indices.extend([i for _ in range(size)])

        # For each unique tensor and index pair, we should have a list of indices to gather from
        # the reducible values tensor
        max_size = max(self._sum_sizes)
        unique_tensor_gather_indices = OrderedDict()
        unique_tensors_offsets_inverse = {v: k for k, v in unique_tensors_offsets_dict.items()}

        old_sum_index = 0
        start_of_current_sum = 0
        for i, (col, tensor_offset, sum_index) in enumerate(zip(
                flat_col_indices, flat_tensor_offsets, sum_indices)):
            # Give the offset of the current flat (axis 1) index, we get the input tensor that
            # feeds its value to it.
            tensor = unique_tensors_offsets_inverse[tensor_offset]
            if tensor not in unique_tensor_gather_indices:
                unique_tensor_gather_indices[tensor] = defaultdict(list)
            # For this tensor-column combination, we register the corresponding index to gather
            # from the padded 2D reducible tensor
            if sum_index != old_sum_index:
                old_sum_index = sum_index
                start_of_current_sum = i

            # Index of the column within the sum
            index_within_sum = i - start_of_current_sum

            # Given the index of the sum and the index of the column within, we can find the index
            # to gather for this particular column of the input tensor
            unique_tensor_gather_indices[tensor][col].append(
                index_within_sum + sum_index * max_size)

        # For each tensor that we have, we compute the scatter indices. Here we construct the
        # nested gather indices needed for gather_cols_3d.
        nested_gather_indices = []
        unique_tensor_lengths = []
        tensor_scatter_indices = OrderedDict()
        for tensor, col_to_gather_col in unique_tensor_gather_indices.items():
            gather_indices_sub = []
            tensor_scatter_indices[tensor] = []
            # Go over all possible indices
            for i in range(tensor.shape[1].value):
                # If this index is registered as one to gather for...
                if i in col_to_gather_col:
                    # ... then we append the gathering columns to the currently considered
                    # tensor column
                    gather_indices_sub.append(col_to_gather_col[i])
                    tensor_scatter_indices[tensor].append(i)
            # Length of the list of columns for each unique input value tensor
            unique_tensor_lengths.append(len(gather_indices_sub))
            # Will contain a list of lists. Inner lists correspond to columns to gather, while
            # outer list corresponds to the individual 'indexed' input nodes
            nested_gather_indices.extend(gather_indices_sub)

        # Gather columns from the counts tensor, per unique (input, index) pair
        if gather_segments_only:
            segment_ids = []
            for i, ind in enumerate(nested_gather_indices):
                segment_ids.extend([i for _ in range(len(ind))])
            num_sums_to_scatter = len(nested_gather_indices)
            nested_gather_indices = list(itertools.chain(*nested_gather_indices))
            transposed = tf.transpose(x)
            gathered = tf.gather(transposed, indices=nested_gather_indices)
            acccumulated = tf.reshape(
                tf.segment_sum(gathered, segment_ids=segment_ids), (num_sums_to_scatter, -1))
            acccumulated = tf.transpose(acccumulated)
        else:
            reducible_values = utils.gather_cols_3d(x, nested_gather_indices)
            # Sum gathered counts together per unique (input, index) pair
            acccumulated = tf.reduce_sum(reducible_values, axis=-1)

        # Split the summed-counts tensor per unique input, based on input-sizes
        accumulated_unique_tensor_values = tf.split(
            acccumulated, unique_tensor_lengths, axis=-1) \
            if len(unique_tensor_lengths) > 1 else [acccumulated]
        return tensor_scatter_indices, accumulated_unique_tensor_values
