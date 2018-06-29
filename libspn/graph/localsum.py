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
class LocalSum(BaseSum):
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
                 inference_type=InferenceType.MARGINAL, name="LocalSum",
                 grid_dim_sizes=None):

        if not grid_dim_sizes:
            raise NotImplementedError(
                "{}: Must also provide grid_dim_sizes at this point.".format(self))

        self._grid_dim_sizes = grid_dim_sizes or [-1] * 2
        self._grid_dim_sizes = list(grid_dim_sizes) if isinstance(grid_dim_sizes, tuple) \
            else grid_dim_sizes
        self._channel_axis = 3
        self._num_channels = num_channels
        num_sums = int(np.prod(self._grid_dim_sizes) * num_channels)
        super().__init__(
            *values, num_sums=num_sums, weights=weights, ivs=ivs,
            inference_type=inference_type, name=name, reduce_axis=4, op_axis=[1, 2])

    @utils.docinherit(BaseSum)
    @utils.lru_cache
    def _prepare_component_wise_processing(
            self, w_tensor, ivs_tensor, *input_tensors, zero_prob_val=0.0):
        shape_suffix = [self._num_channels, self._max_sum_size]
        w_tensor = tf.reshape(w_tensor, [1] + self._grid_dim_sizes + shape_suffix)

        input_tensors = [self._spatial_reshape(t) for t in input_tensors]

        reducible_inputs = utils.concat_maybe(input_tensors, axis=self._reduce_axis)

        if ivs_tensor is not None:
            ivs_tensor = tf.reshape(ivs_tensor, shape=[-1] + self._grid_dim_sizes + shape_suffix)

        return w_tensor, ivs_tensor, reducible_inputs

    @property
    def output_shape_spatial(self):
        return tuple(self._grid_dim_sizes + [self._num_channels])

    def generate_weights(self, init_value=1, trainable=True, input_sizes=None,
                         log=False, name=None):
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
        # Count all input values
        num_values = max(self._sum_sizes)
        # Generate weights
        weights = Weights(
            init_value=init_value, num_weights=num_values, num_sums=self._num_sums,
            log=log, trainable=trainable, name=name)
        self.set_weights(weights)
        return weights

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
        return num_sums * [num_values]

    @utils.docinherit(BaseSum)
    def _compute_out_size(self, *input_out_sizes):
        return self._num_sums

    @utils.lru_cache
    @utils.docinherit(BaseSum)
    def _compute_value(self, w_tensor, ivs_tensor, *input_tensors,
                           dropconnect_keep_prob=None, dropout_keep_prob=None):
        val = super(LocalSum, self)._compute_value(
            w_tensor, ivs_tensor, *input_tensors, dropconnect_keep_prob=dropconnect_keep_prob,
            dropout_keep_prob=dropout_keep_prob)
        return tf.reshape(val, (-1, self._compute_out_size()))

    @utils.lru_cache
    @utils.docinherit(BaseSum)
    def _compute_log_value(self, w_tensor, ivs_tensor, *input_tensors,
                           dropconnect_keep_prob=None, dropout_keep_prob=None):
        val = super(LocalSum, self)._compute_log_value(
            w_tensor, ivs_tensor, *input_tensors, dropconnect_keep_prob=dropconnect_keep_prob,
            dropout_keep_prob=dropout_keep_prob)
        return tf.reshape(val, (-1, self._compute_out_size()))

    @utils.lru_cache
    @utils.docinherit(BaseSum)
    def _compute_mpe_value(self, w_tensor, ivs_tensor, *input_tensors,
                           dropconnect_keep_prob=None, dropout_keep_prob=None):
        val = super(LocalSum, self)._compute_mpe_value(
            w_tensor, ivs_tensor, *input_tensors, dropconnect_keep_prob=dropconnect_keep_prob,
            dropout_keep_prob=dropout_keep_prob)
        return tf.reshape(val, (-1, self._compute_out_size()))

    @utils.lru_cache
    @utils.docinherit(BaseSum)
    def _compute_log_mpe_value(self, w_tensor, ivs_tensor, *input_tensors,
                           dropconnect_keep_prob=None, dropout_keep_prob=None):
        val = super(LocalSum, self)._compute_log_mpe_value(
            w_tensor, ivs_tensor, *input_tensors, dropconnect_keep_prob=dropconnect_keep_prob,
            dropout_keep_prob=dropout_keep_prob)
        return tf.reshape(val, (-1, self._compute_out_size()))

    @utils.lru_cache
    @utils.docinherit(BaseSum)
    def _compute_mpe_path_common(
            self, reducible_tensor, counts, w_tensor, ivs_tensor, *input_tensors, log=True,
            sample=False, sample_prob=None, sample_rank_based=None):
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
        sample_prob = utils.maybe_first(sample_prob, self._sample_prob)
        if sample:
            if log:
                max_indices = self._reduce_sample_log(
                    reducible_tensor, sample_prob=sample_prob, rank_based=sample_rank_based)
            else:
                max_indices = self._reduce_sample(
                    reducible_tensor, sample_prob=sample_prob, rank_based=sample_rank_based)
        else:
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
        # x_acc_op = tf.reduce_sum(x, axis=self._op_axis)
        x_acc_op = tf.reshape(x, (-1, int(np.prod(self.output_shape_spatial)), self._max_sum_size))
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
    def _compute_scope(self, weight_scopes, ivs_scopes, *value_scopes, check_valid=False):
        flat_value_scopes, ivs_scopes, *value_scopes = self._get_flat_value_scopes(
            weight_scopes, ivs_scopes, *value_scopes)

        value_scopes_grid = [
            np.asarray(vs).reshape(self._grid_dim_sizes + [-1]) for vs in value_scopes]
        value_scopes_concat = np.concatenate(value_scopes_grid, axis=2)
        
        if check_valid:
            for scope_list in value_scopes_concat.reshape((-1, self._max_sum_size)):
                if any(s != scope_list[0] for s in scope_list[1:]):
                    return None
        
        if self._ivs:
            raise NotImplementedError("{}: no support for computing scope when node has latent IVs."
                                      .format(self))
        return list(map(Scope.merge_scopes, value_scopes_concat.repeat(self._num_channels).reshape(
            (-1, self._max_sum_size))))

    @utils.docinherit(OpNode)
    def _compute_valid(self, weight_scopes, ivs_scopes, *value_scopes):
        return self._compute_scope(weight_scopes, ivs_scopes, *value_scopes, check_valid=True)
