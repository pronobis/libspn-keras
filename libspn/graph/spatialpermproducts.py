import itertools

import numpy as np
import tensorflow as tf

from libspn import utils
from libspn.graph.node import OpNode
from libspn.log import get_logger
from libspn.inference.type import InferenceType
import libspn.conf as conf
from libspn.exceptions import StructureError
from libspn.graph.scope import Scope
from libspn.utils.math import maybe_random_0toN_permutations


@utils.register_serializable
class SpatialPermProducts(OpNode):
    """A container representing convolutional products in an SPN.

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

    logger = get_logger()

    def __init__(self, *values, num_channels=None, inference_type=InferenceType.MARGINAL,
                 name="SpatialPermProduct", grid_dim_sizes=None, num_channels_max=512):
        self._batch_axis = 0
        self._channel_axis = 3
        super().__init__(inference_type=inference_type, name=name)

        if grid_dim_sizes is None:
            raise NotImplementedError(
                "{}: Must also provide grid_dim_sizes at this point.".format(self))

        self._num_channels_max = num_channels_max
        self._grid_dim_sizes = grid_dim_sizes or [-1] * 2
        if isinstance(self._grid_dim_sizes, tuple):
            self._grid_dim_sizes = list(self._grid_dim_sizes)
        self.set_values(*values)


    def set_values(self, *values):
        """Set the inputs providing input values to this node. If no arguments
        are given, all existing value inputs get disconnected.

        Args:
            *values (input_like): Inputs providing input values to this node.
                See :meth:`~libspn.Input.as_input` for possible values.
        """
        self._values = self._parse_inputs(*values)
        self._permute_connections, self._num_channels = self._generate_permutation_matrix()

    def generate_permutations(self):
        permutations = maybe_random_0toN_permutations(
            self._num_channels_per_input(), max_size=self._num_channels_max)
        return permutations

    def _generate_permutation_matrix(self):
        # Get total number of input channels and input channels per input
        num_channels_per_input = self._num_channels_per_input()
        num_inp_channels = sum(num_channels_per_input)

        # Generate sparse indiices
        permutations = self.generate_permutations()
        num_out_channels = len(permutations)
        offsets = np.cumsum([0] + num_channels_per_input[:-1]).reshape((1, len(self._values)))
        permutation_sparse_rows = (permutations + offsets).ravel()
        permutation_sparse_cols = np.repeat(np.arange(num_out_channels), repeats=len(self._values))
        sparse_indices = (permutation_sparse_rows, permutation_sparse_cols)

        # Generate Dense matrix
        dense_mat = np.zeros((num_inp_channels, num_out_channels))
        dense_mat[sparse_indices] = 1.0
        return dense_mat, num_out_channels

    @utils.lru_cache
    def _spatial_concat(self, *input_tensors):
        """Concatenates input tensors spatially. Makes sure to reshape them before.

        Args:
            input_tensors (tuple): A tuple of `Tensor`s to concatenate along the channel axis.

        Returns:
            The concatenated tensor.
        """
        input_tensors = [self._spatial_reshape(t) for t in input_tensors]
        return utils.concat_maybe(input_tensors, axis=self._channel_axis)

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
            return tf.reshape(t, [-1] + self._grid_dim_sizes + [input_channels])
        return tf.reshape(t, [-1] + self._grid_dim_sizes + [self._num_channels])

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

    def _num_channels_per_input(self):
        """Returns a list of number of input channels for each value Input.

        Returns:
            A list of ints containing the number of channels.
        """
        input_sizes = self.get_input_sizes()
        return [int(s // np.prod(self._grid_dim_sizes)) for s in input_sizes]

    def _num_input_channels(self):
        """Computes the total number of input channels.

        Returns:
            An int indicating the number of input channels for the convolution operation.
        """
        return sum(self._num_channels_per_input())

    def _compute_out_size_spatial(self, *input_out_sizes):
        """Computes spatial output shape.

        Returns:
            A tuple with (num_rows, num_cols, num_channels).
        """
        return tuple(self._grid_dim_sizes + [self._num_channels])

    def _compute_out_size(self, *input_out_sizes):
        return int(np.prod(self._compute_out_size_spatial(*input_out_sizes)))

    @property
    def output_shape_spatial(self):
        """tuple: The spatial shape of this node, formatted as (rows, columns, channels). """
        return self._compute_out_size_spatial()

    def _compute_value(self, *input_tensors):
        raise NotImplementedError("{}: No linear value implementation for ConvProd".format(self))

    @utils.lru_cache
    def _compute_log_value(self, *input_tensors):
        # Concatenate along channel axis
        concat_inp = self._spatial_concat(*input_tensors)
        flat_inner = tf.reshape(concat_inp, (-1, self._num_input_channels()))
        matmul_out = tf.matmul(flat_inner, self.permutation_matrix(), b_is_sparse=True)
        return tf.reshape(matmul_out, [-1, self._compute_out_size()])

    def permutation_matrix(self):
        return tf.constant(self._permute_connections, dtype=conf.dtype)

    def _compute_mpe_value(self, *input_tensors):
        raise NotImplementedError("{}: No linear MPE value implementation for ConvProd"
                                  .format(self))

    def _compute_log_mpe_value(self, *input_tensors):
        return self._compute_log_value(*input_tensors)

    def _compute_mpe_path_common(self, counts, *input_tensors):
        if not self._values:
            raise StructureError("{} is missing input values.".format(self))
        # Concatenate inputs along channel axis, should already be done during forward pass
        counts_flat_inner = tf.reshape(counts, (-1, self._num_channels))
        counts_inp_flat_inner = tf.matmul(
            counts_flat_inner, self.permutation_matrix(), b_is_sparse=True, transpose_b=True)
        counts_inp = tf.reshape(
            counts_inp_flat_inner, [-1] + self._grid_dim_sizes + [self._num_input_channels()])
        return self._split_to_children(counts_inp)

    def _compute_log_mpe_path(self, counts, *input_values, add_random=False,
                              use_unweighted=False, with_ivs=False, sample=False, sample_prob=None):
        return self._compute_mpe_path_common(counts, *input_values)

    def _compute_mpe_path(self, counts, *input_values, add_random=False,
                          use_unweighted=False, with_ivs=False, sample=False, sample_prob=None):
        return self._compute_mpe_path_common(counts, *input_values)

    @utils.lru_cache
    def _split_to_children(self, x):
        if len(self.inputs) == 1:
            return [self._flatten(x)]
        x_split = tf.split(x, num_or_size_splits=self._num_channels_per_input(),
                           axis=self._channel_axis)
        return [self._flatten(t) for t in x_split]

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

    def _compute_gradient(self, *input_tensors, with_ivs=True):
        raise NotImplementedError("{}: No gradient implementation available.".format(self))

    def _compute_log_gradient(
            self, *value_tensors, with_ivs=True, sum_weight_grads=False):
        raise NotImplementedError("{}: No log-gradient implementation available.".format(self))

    @utils.docinherit(OpNode)
    def _compute_scope(self, *value_scopes, check_valid=False):
        flat_value_scopes = self._gather_input_scopes(*value_scopes)

        value_scopes_grid = [
            np.asarray(vs).reshape(self._grid_dim_sizes + [-1]) for vs in flat_value_scopes]
        value_scopes_concat = np.concatenate(value_scopes_grid, axis=2)

        num_rows, num_cols = self._grid_dim_sizes
        permutation_mat = self._permute_connections.T
        scope_list = []
        for row in range(num_rows):
            for col in range(num_cols):
                for perm_row in permutation_mat:
                    sc = []
                    for i, e in enumerate(perm_row):
                        if e:
                            sc.append(value_scopes_concat[row, col, i])
                    if check_valid:
                        for sc0, sc1 in itertools.combinations(sc, 2):
                            if sc0 & sc1:
                                self.logger.warn("{} is not decomposable with input scopes {}..."
                                                 .format(self, flat_value_scopes[:10]))
                                return None
                    scope_list.append(Scope.merge_scopes(sc))
        return scope_list
    
    @utils.docinherit(OpNode)
    def _compute_valid(self, *value_scopes):
        return self._compute_scope(*value_scopes, check_valid=True)

    def deserialize_inputs(self, data, nodes_by_name):
        pass

    @property
    def inputs(self):
        return self._values

    def serialize(self):
        pass

    def deserialize(self, data):
        pass

    @property
    def _const_out_size(self):
        return True