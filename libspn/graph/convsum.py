# ------------------------------------------------------------------------
# Copyright (C) 2016 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

from libspn.inference.type import InferenceType
from libspn.graph.basesum import BaseSum
import libspn.utils as utils
import tensorflow as tf
from libspn.exceptions import StructureError
from libspn.graph.weights import Weights
import numpy as np
from libspn.graph.scope import Scope
from libspn.graph.node import OpNode


@utils.register_serializable
class ConvSum(BaseSum):
    """A container representing convolutional sums (which share the same input) in an SPN.

    Args:
        *values (input_like): Inputs providing input values to this container.
            See :meth:`~libspn.Input.as_input` for possible values.
        num_sums (int): Number of Sum ops modelled by this container.
        weights (input_like): Input providing weights container to this sum container.
            See :meth:`~libspn.Input.as_input` for possible values. If set
            to ``None``, the input is disconnected.
        ivs (input_like): Input providing IVs of an explicit latent variable
            associated with this sum container. See :meth:`~libspn.Input.as_input`
            for possible values. If set to ``None``, the input is disconnected.
        name (str): Name of the container.

    Attributes:
        inference_type(InferenceType): Flag indicating the preferred inference
                                       type for this container that will be used
                                       during value calculation and learning.
                                       Can be changed at any time and will be
                                       used during the next inference/learning
                                       op generation.
    """

    def __init__(self, *values, num_channels=1, weights=None, ivs=None,
                 inference_type=InferenceType.MARGINAL, name="ConvSums",
                 grid_dim_sizes=None):

        if not grid_dim_sizes:
            raise NotImplementedError(
                "{}: Must also provide grid_dim_sizes at this point.".format(self))

        self._grid_dim_sizes = grid_dim_sizes or [-1] * 2
        self._channel_axis = 3
        super().__init__(
            *values, num_sums=num_channels, weights=weights, ivs=ivs,
            inference_type=inference_type, name=name, reduce_axis=4, op_axis=[1, 2])

    @utils.docinherit(BaseSum)
    @utils.lru_cache
    def _prepare_component_wise_processing(
            self, w_tensor, ivs_tensor, *input_tensors, zero_prob_val=0.0):
        shape_suffix = [self._num_sums, self._max_sum_size]
        w_tensor = tf.reshape(w_tensor, [1] * 3 + shape_suffix)

        input_tensors = [self._spatial_reshape(t) for t in input_tensors]

        reducible_inputs = utils.concat_maybe(input_tensors, axis=self._reduce_axis)
        if ivs_tensor is not None:
            ivs_tensor = tf.reshape(ivs_tensor, shape=[-1] + self._grid_dim_sizes + shape_suffix)

        return w_tensor, ivs_tensor, reducible_inputs

    def _spatial_reshape(self, t, forward=True):
        """Reshapes a Tensor ``t``` to one that represents the spatial dimensions.

        Args:
            t (Tensor): The ``Tensor`` to reshape.
            forward (bool): Whether to reshape for forward inference. If True, reshapes to
                ``[batch, rows, cols, 1, input_channels]``. Otherwise, reshapes to
                ``[batch, rows, cols, output_channels, input_channels]``.
        Returns:
             A reshaped ``Tensor``.
        """
        non_batch_dim_size = self._non_batch_dim_prod(t)
        if forward:
            input_channels = non_batch_dim_size // np.prod(self._grid_dim_sizes)
            return tf.reshape(t, [-1] + self._grid_dim_sizes + [1, input_channels])
        return tf.reshape(t, [-1] + self._grid_dim_sizes + [non_batch_dim_size // (
            self._max_sum_size * np.prod(self._grid_dim_sizes)), self._max_sum_size])

    def _non_batch_dim_prod(self, t):
        """Computes the product of the non-batch dimensions to be used for reshaping purposes.

        Args:
            t (Tensor): A ``Tensor`` for which to compute the product.

        Returns:
            An ``int``: product of non-batch dimensions.
        """
        non_batch_dim_size = np.prod([ds for i, ds in enumerate(t.shape.as_list())
                                      if i != self._batch_axis])
        return int(non_batch_dim_size)

    def _get_input_num_channels(self):
        """Returns a list of number of input channels for each value Input.

        Returns:
            A list of ints containing the number of channels.
        """
        _, _, *input_sizes = self.get_input_sizes()
        return [int(s // np.prod(self._grid_dim_sizes)) for s in input_sizes]

    @utils.docinherit(BaseSum)
    def _get_sum_sizes(self, num_sums):
        num_values = sum(self._get_input_num_channels())  # Skip ivs, weights
        return num_sums * int(np.prod(self._grid_dim_sizes)) * [num_values]

    @utils.docinherit(BaseSum)
    def _compute_out_size(self):
        return int(np.prod(self._grid_dim_sizes) * self._num_sums)

    @utils.lru_cache
    @utils.docinherit(BaseSum)
    def _compute_value(self, w_tensor, ivs_tensor, *input_tensors):
        val = super(ConvSum, self)._compute_value(w_tensor, ivs_tensor, *input_tensors)
        return tf.reshape(val, (-1, self._compute_out_size()))

    @utils.lru_cache
    @utils.docinherit(BaseSum)
    def _compute_log_value(self, w_tensor, ivs_tensor, *input_tensors):
        val = super(ConvSum, self)._compute_log_value(w_tensor, ivs_tensor, *input_tensors)
        return tf.reshape(val, (-1, self._compute_out_size()))

    @utils.lru_cache
    @utils.docinherit(BaseSum)
    def _compute_mpe_value(self, w_tensor, ivs_tensor, *input_tensors):
        val = super(ConvSum, self)._compute_mpe_value(w_tensor, ivs_tensor, *input_tensors)
        return tf.reshape(val, (-1, self._compute_out_size()))

    @utils.lru_cache
    @utils.docinherit(BaseSum)
    def _compute_log_mpe_value(self, w_tensor, ivs_tensor, *input_tensors):
        val = super(ConvSum, self)._compute_log_mpe_value(w_tensor, ivs_tensor, *input_tensors)
        return tf.reshape(val, (-1, self._compute_out_size()))

    @utils.lru_cache
    @utils.docinherit(BaseSum)
    def _compute_mpe_path_common(
            self, reducible_tensor, counts, w_tensor, ivs_tensor, *input_tensors):
        """Common operations for computing the MPE path.

        Args:
            reducible_tensor (Tensor): A (weighted) ``Tensor`` of (log-)values of this container.
            counts (Tensor): A ``Tensor`` that contains the accumulated counts of the parents
                             of this container.
            w_tensor (Tensor):  A ``Tensor`` containing the (log-)value of the weights.
            ivs_tensor (Tensor): A ``Tensor`` containing the (log-)value of the IVs.
            input_tensors (list): A list of ``Tensor``s with outputs of the child nodes.

        Returns:
            A ``list`` of ``tuple``s [(MPE counts, input tensor), ...] where the first corresponds
            to the Weights of this container, the second corresponds to the IVs and the remaining
            tuples correspond to the nodes in ``self._values``.
        """
        max_indices = self._reduce_argmax(reducible_tensor)
        max_indices = tf.reshape(max_indices, (-1, self._compute_out_size()))
        max_counts = utils.scatter_values(
            params=counts, indices=max_indices, num_out_cols=self._max_sum_size)
        weight_counts, input_counts = self._accumulate_and_split_to_children(max_counts)
        return self._scatter_to_input_tensors(
            (weight_counts, w_tensor),  # Weights
            (max_counts, ivs_tensor),  # IVs
            *[(t, v) for t, v in zip(input_counts, input_tensors)])  # Values

    @utils.lru_cache
    @utils.docinherit(BaseSum)
    def _accumulate_and_split_to_children(self, x, *input_tensors):
        x = self._spatial_reshape(x, forward=False)
        x_acc_op = tf.reduce_sum(x, axis=self._op_axis)
        x_acc_channel_split = tf.split(
            tf.reduce_sum(x, axis=self._channel_axis),
            num_or_size_splits=self._get_input_num_channels(), axis=self._channel_axis)
        return x_acc_op, [self._flatten(t) for t in x_acc_channel_split]

    @utils.lru_cache
    def _flatten(self, t):
        """Flattens a Tensor ``t`` so that the resulting shape is [batch, non_batch]

        Args:
            t (Tensor): A ``Tensor```to flatten

        Returns:
            A flattened ``Tensor``.
        """
        if self._batch_axis != 0:
            raise NotImplementedError("{}: Cannot flatten if batch axis isn't equal to zero."
                                      .format(self))
        non_batch_dim_size = self._non_batch_dim_prod(t)
        return tf.reshape(t, (-1, non_batch_dim_size))

    @utils.docinherit(BaseSum)
    def _compute_gradient(self, gradients, w_tensor, ivs_tensor, *input_tensors, with_ivs=True):
        raise NotImplementedError("{}: No gradient implementation available.".format(self))

    @utils.docinherit(BaseSum)
    def _compute_log_gradient(
            self, gradients, w_tensor, ivs_tensor, *value_tensors, with_ivs=True,
            sum_weight_grads=False):
        raise NotImplementedError("{}: No log-gradient implementation available.".format(self))

    @utils.docinherit(OpNode)
    def _compute_scope(self, weight_scopes, ivs_scopes, *value_scopes):
        flat_value_scopes, ivs_scopes, *value_scopes = self._get_flat_value_scopes(
            weight_scopes, ivs_scopes, *value_scopes)

        value_scopes_grid = [
            np.asarray(vs).reshape(self._grid_dim_sizes + [-1]) for vs in value_scopes]
        value_scopes_concat = np.concatenate(value_scopes_grid, axis=2)
        if self._ivs:
            raise NotImplementedError("{}: no support for computing scope when node has latent IVs."
                                      .format(self))
        return list(map(Scope.merge_scopes, value_scopes_concat.repeat(self._num_sums).reshape(
            (-1, self._max_sum_size))))

    @utils.docinherit(OpNode)
    def _compute_valid(self, weight_scopes, ivs_scopes, *value_scopes):
        self.logger.warn("{}: validity is skipped for convolutional sum layer.".format(self))
        return self._compute_scope(weight_scopes, ivs_scopes, *value_scopes)
