# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

"""LibSPN math functions."""

import tensorflow as tf
import numpy as np
import collections
from libspn import conf
from libspn.ops import ops
from libspn.utils.serialization import register_serializable


class ValueType:

    """A class specifying various types of values that be passed to the SPN
    graph."""

    @register_serializable
    class RANDOM_UNIFORM:

        """A random value from a uniform distribution.

        Attributes:
            min_val: The lower bound of the range of random values.
            max_val: The upper bound of the range of random values.
        """

        def __init__(self, min_val=0, max_val=1):
            self.min_val = min_val
            self.max_val = max_val

        def __repr__(self):
            return ("ValueType.RANDOM_UNIFORM(min_val=%s, max_val=%s)" %
                    (self.min_val, self.max_val))

        def serialize(self):
            return {'min_val': self.min_val,
                    'max_val': self.max_val}

        def deserialize(self, data):
            self.min_val = data['min_val']
            self.max_val = data['max_val']


def gather_cols(params, indices, name=None):
    """Gather columns of a 2D tensor or values of a 1D tensor.

    Args:
        params (Tensor): A 1D or 2D tensor.
        indices (array_like): A 1D integer array.
        name (str): A name for the operation (optional).

    Returns:
        Tensor: Has the same dtype and number of dimensions and type as ``params``.
    """
    with tf.name_scope(name, "gather_cols", [params, indices]):
        params = tf.convert_to_tensor(params, name="params")
        indices = np.asarray(indices)
        # Check params
        param_shape = params.get_shape()
        param_dims = param_shape.ndims
        if param_dims == 1:
            param_size = param_shape[0].value
        elif param_dims == 2:
            param_size = param_shape[1].value
        else:
            raise ValueError("'params' must be 1D or 2D")
        # We need the size defined for optimizations
        if param_size is None:
            raise RuntimeError("The indexed dimension of 'params' is not specified")
        # Check indices
        if indices.ndim != 1:
            raise ValueError("'indices' must be 1D")
        if indices.size < 1:
            raise ValueError("'indices' cannot be empty")
        if not np.issubdtype(indices.dtype, np.integer):
            raise ValueError("'indices' must be integer, not %s"
                             % indices.dtype)
        if np.any((indices < 0) | (indices >= param_size)):
            raise ValueError("'indices' must fit the the indexed dimension")
        # Define op
        if param_size == 1:
            if indices.size == 1:
                # Single column tensor with a single indices, which should include
                # it, so just forward tensor
                return params
            else:
                # Single column tensor with multiple indices - case of tiling
                return tf.tile(params, ([indices.size] if param_dims == 1
                                        else [1, indices.size]))
        elif indices.size == param_size and np.all(np.ediff1d(indices) == 1):
            # Indices index all params in the original order, pass through
            return params
        elif indices.size == 1:
            # Gathering a single column
            if param_dims == 1:
                # Gather is faster than custom for 1D.
                # It is as fast as slice for int64, and generates smaller graph
                return tf.gather(params, indices)
            else:
                if conf.custom_gather_cols:
                    return ops.gather_cols(params, indices)
                else:
                    return tf.slice(params, [0, indices[0]], [-1, 1])
        else:
            # Gathering multiple columns from multi-column tensor
            if param_dims == 1:
                # Gather is faster than custom for 1D.
                return tf.gather(params, indices)
            else:
                if conf.custom_gather_cols:
                    return ops.gather_cols(params, indices)
                else:
                    return tf.gather(params, indices, axis=1)


def gather_cols_3d(params, indices, pad_elem=0, name=None):
    """Gather columns of a 2D tensor or values of a 1D tensor into a 1D, 2D or 3D
       tensor, based on the dimension of params and list size of indices.
       The output tensor contains a slice of gathered columns, per 1D-indices array
       in the indices-list. If the size of 1D-indices array in the indices-list is
       uneven, then the empty columns in the corresponding slice are padded with '0'.

    Args:
        params (Tensor): A 1D or 2D tensor.
        indices (array_like): A list of 1D integer array.
        name (str): A name for the operation (optional).

    Returns:
        Tensor: Has the same dtype as ``params``, and rank = R_params + R_indices - 1 .
    """
    with tf.name_scope(name, "gather_cols_3d", [params, indices, pad_elem]):
        params = tf.convert_to_tensor(params, name="params")
        ind_2D = False
        if isinstance(indices[0], collections.Iterable):
            # Convert indices into a list of 1D np.arrays
            indices = [np.asarray(ind) for ind in indices]
            ind_2D = True
        else:
            indices = [np.asarray(indices)]
        # Check params
        param_shape = params.get_shape()
        param_dims = param_shape.ndims
        if param_dims == 1:
            param_size = param_shape[0].value
        elif param_dims == 2:
            param_size = param_shape[1].value
        else:
            raise ValueError("'params' must be 1D or 2D")
        # We need the size defined for optimizations
        if param_size is None:
            raise RuntimeError("The indexed dimension of 'params' is not specified")
        # Check indices
        if any(ind.ndim != 1 for ind in indices):
            raise ValueError("Each 'indices' must be 1D")
        if any(ind.size < 1 for ind in indices):
            raise ValueError("None of the 'indices' can be empty")
        if any(not np.issubdtype(ind.dtype, np.integer) for ind in indices):
            raise ValueError("'indices' can only be integer type")
        if any(np.any((ind < 0) | (ind >= param_size)) for ind in indices):
            raise ValueError("All 'indices' must fit the the indexed dimension")

        # Define op
        if len(indices) == 1:
            # Single indices (1D)
            if param_size == 1 and indices[0].size == 1:
                # Single column tensor, with a single column to be gathered,
                # indices should include it, so just forward tensor
                return_tensor = params
            elif indices[0].size == param_size and np.all(np.ediff1d(indices[0]) == 1):
                # Indices contains all the columns, and in the correct order. So
                # just forward tensor
                return_tensor = params
            else:
                # If not, then just pass it to gather_cols() function
                return_tensor = gather_cols(params, indices[0])

            if ind_2D:
                # Indices is 2D, so insert an extra dimension to the output
                return tf.expand_dims(return_tensor, axis=-2)
            else:
                return return_tensor
        else:
            # Multiple rows of indices
            indices_cols = max([ind.size for ind in indices])
            padding = False
            for i, ind in enumerate(indices):
                if ind.size < indices_cols:
                    padding = True
                    indices[i] = np.append(ind, np.ones(indices_cols-ind.size,
                                                        dtype=ind.dtype)*-1)
            # Convert the list of indices arrays into an indices matrix
            indices = np.vstack(indices)
            if not padding and indices_cols == param_size and \
               all(np.all(np.ediff1d(ind) == 1) for ind in indices):
                indices_rows = indices.shape[0]
                if param_dims == 1:
                    return tf.reshape(tf.tile(params, [indices_rows]),
                                      (-1, indices_cols))
                else:
                    return tf.reshape(tf.tile(params, [1, indices_rows]),
                                      (-1, indices_rows, indices_cols))
            else:
                if conf.custom_gather_cols_3d:
                    return ops.gather_cols_3d(params, indices, padding, pad_elem)
                else:
                    pad_elem = np.array(pad_elem).astype(tf.DType(params.dtype).as_numpy_dtype)
                    if param_dims == 1:
                        axis = 0
                        if padding:
                            augmented = tf.concat([[tf.constant(pad_elem, dtype=params.dtype)],
                                                   params], axis=axis)
                            gathered = tf.gather(augmented, indices=indices.ravel() + 1, axis=axis)
                        else:
                            gathered = tf.gather(params, indices=indices.ravel(), axis=axis)
                        return tf.reshape(gathered, indices.shape)
                    # else:
                    axis = 1
                    if padding:
                        augmented = tf.concat([
                            tf.fill((tf.shape(params)[0], 1), value=tf.constant(
                                pad_elem, dtype=params.dtype)),
                            params
                        ], axis=axis)
                        gathered = tf.gather(augmented, indices=indices.ravel() + 1, axis=axis)
                    else:
                        gathered = tf.gather(params, indices=indices.ravel(), axis=axis)
                    return tf.reshape(gathered, (-1,) + indices.shape)


def scatter_cols(params, indices, num_out_cols, name=None):
    """Scatter columns of a 2D tensor or values of a 1D tensor into a tensor
    with the same number of dimensions and ``num_out_cols`` columns or values.

    Args:
        params (Tensor): A 1D or 2D tensor.
        indices (array_like): A 1D integer array indexing the columns in the
                              output array to which ``params`` is scattered.
        num_cols (int): The number of columns in the output tensor.
        name (str): A name for the operation (optional).

    Returns:
        Tensor: Has the same dtype and number of dimensions as ``params``.
    """
    with tf.name_scope(name, "scatter_cols", [params, indices]):
        # Check input
        params = tf.convert_to_tensor(params, name="params")
        indices = np.asarray(indices)
        # Check params
        param_shape = params.get_shape()
        param_dims = param_shape.ndims
        if param_dims == 1:
            param_size = param_shape[0].value
        elif param_dims == 2:
            param_size = param_shape[1].value
        else:
            raise ValueError("'params' must be 1D or 2D")
        # We need the size defined for optimizations
        if param_size is None:
            raise RuntimeError("The indexed dimension of 'params' is not specified")
        # Check num_out_cols
        if not isinstance(num_out_cols, int):
            raise ValueError("'num_out_cols' must be integer, not %s"
                             % type(num_out_cols))
        if num_out_cols < param_size:
            raise ValueError("'num_out_cols' must be larger than or equal to the size of "
                             "the indexed dimension of 'params'")
        # Check indices
        if indices.ndim != 1:
            raise ValueError("'indices' must be 1D")
        if indices.size != param_size:
            raise ValueError("Sizes of 'indices' and the indexed dimension of "
                             "'params' must be the same")
        if not np.issubdtype(indices.dtype, np.integer):
            raise ValueError("'indices' must be integer, not %s"
                             % indices.dtype)
        if np.any((indices < 0) | (indices >= num_out_cols)):
            raise ValueError("'indices' must be smaller than 'num_out_cols'")
        if len(set(indices)) != len(indices):
            raise ValueError("'indices' cannot contain duplicates")
        # Define op
        if num_out_cols == 1:
            # Scatter to a single column tensor, it must be from 1 column
            # tensor and the indices must include it. Just forward the tensor.
            return params
        elif num_out_cols == indices.size and np.all(np.ediff1d(indices) == 1):
            # Output equals input
            return params
        elif param_size == 1:
            # Scatter a single column tensor to a multi-column tensor
            if param_dims == 1:
                # Just pad with zeros, pad is fastest and offers smallest graph
                return tf.pad(params, [[indices[0], num_out_cols - indices[0] - 1]])
            else:
                # Currently pad is fastest (for GPU) and builds smaller graph
                # if conf.custom_scatter_cols:
                #     return ops.scatter_cols(
                #         params, indices,
                #         pad_elem=tf.constant(0, dtype=params.dtype),
                #         num_out_col=num_out_cols)
                # else:
                return tf.pad(params, [[0, 0],
                                       [indices[0], num_out_cols - indices[0] - 1]])
        else:
            # Scatter a multi-column tensor to a multi-column tensor
            if param_dims == 1:
                if conf.custom_scatter_cols:
                    return ops.scatter_cols(
                        params, indices,
                        pad_elem=tf.constant(0, dtype=params.dtype),
                        num_out_col=num_out_cols)
                else:
                    with_zeros = tf.concat(values=([0], params), axis=0)
                    gather_indices = np.zeros(num_out_cols, dtype=int)
                    gather_indices[indices] = np.arange(indices.size) + 1
                    return gather_cols(with_zeros, gather_indices)
            else:
                if conf.custom_scatter_cols:
                    return ops.scatter_cols(
                        params, indices,
                        pad_elem=tf.constant(0, dtype=params.dtype),
                        num_out_col=num_out_cols)
                else:
                    zero_col = tf.zeros((tf.shape(params)[0], 1),
                                        dtype=params.dtype)
                    with_zeros = tf.concat(values=(zero_col, params), axis=1)
                    gather_indices = np.zeros(num_out_cols, dtype=int)
                    gather_indices[indices] = np.arange(indices.size) + 1
                    return gather_cols(with_zeros, gather_indices)


def scatter_values(params, indices, num_out_cols, name=None):
    """Scatter values of a rank R (1D or 2D) tensor into a rank R+1 (2D or 3D)
    tensor, with the inner-most dimensions having size ``num_out_cols``.

    Args:
        params (Tensor): A 1D or 2D tensor.
        indices (array_like): A 1D or 2D (same dimension as params) integer
                              array indexing the columns in the output tensor
                              to which the respective value in ``params`` is
                              scattered to.
        num_cols (int): The number of columns in the output tensor.
        name (str): A name for the operation (optional).

    Returns:
        Tensor: Has the same dtype but an additional dimension as ``params``.
    """
    with tf.name_scope(name, "scatter_cols", [params, indices]):
        # Check input
        params = tf.convert_to_tensor(params, name="params")
        indices = tf.convert_to_tensor(indices, name="indices")
        # Check params
        param_shape = params.get_shape()
        param_dims = param_shape.ndims
        if param_dims == 1:
            param_size = param_shape[0].value
        elif param_dims == 2:
            param_size = param_shape[1].value
        else:
            raise ValueError("'params' must be 1D or 2D but it is %dD" %
                             param_dims)
        # Check num_out_cols
        if not isinstance(num_out_cols, int):
            raise ValueError("'num_out_cols' must be integer, not %s"
                             % type(num_out_cols))
        # Check indices
        indices_shape = indices.get_shape()
        indices_dims = indices_shape.ndims
        if indices_dims != param_dims:
            raise ValueError("Dimension of 'indices': %dD," % indices_dims,
                             " and dimension of 'params': %dD must be the same" %
                             param_dims)
        if indices_dims == 1:
            indices_size = indices_shape[0].value
        elif indices_dims == 2:
            indices_size = indices_shape[1].value
        if indices_size != param_size:
            raise ValueError("Sizes of 'indices' and 'params' must be the same")
        # TODO: Need a way for tensor bound-checking that 0 <= indices < num_out_cols
        # if np.any((indices < 0) | (indices >= num_out_cols)):
        #     raise ValueError("'indices' must be smaller than 'num_out_cols'")
        # Define op
        if num_out_cols == 1:
            # Scatter to a single column tensor, it must be from 1 column
            # tensor and the indices must include it. Just expand the dimension
            # and forward the tensor.
            return tf.expand_dims(params, axis=-1)
        else:
            if conf.custom_scatter_values:
                return ops.scatter_values(params, indices,
                                          num_out_cols=num_out_cols)
            else:  # OneHot
                if param_dims == 1:
                    return tf.one_hot(indices, num_out_cols, dtype=params.dtype) \
                           * tf.expand_dims(params, axis=1)
                else:
                    return tf.one_hot(indices, num_out_cols, dtype=params.dtype) \
                           * tf.expand_dims(params, axis=2)


def broadcast_value(value, shape, dtype, name=None):
    """Broadcast the given value to the given shape and dtype. If ``value`` is
    one of the members of :class:`~libspn.ValueType`, the requested value will
    be generated and placed in every element of a tensor of the requested shape
    and dtype. If ``value`` is a 0-D tensor or a Python value, it will be
    broadcasted to the requested shape and converted to the requested dtype.
    Otherwise, the value is used as is.

    Args:
        value: The input value.
        shape: The shape of the output.
        dtype: The type of the output.

    Return:
        Tensor: A tensor containing the broadcasted and converted value.
    """
    with tf.name_scope(name, "broadcast_value", [value]):
        # Recognize ValueTypes
        if isinstance(value, ValueType.RANDOM_UNIFORM):
            return tf.random_uniform(shape=shape,
                                     minval=value.min_val,
                                     maxval=value.max_val,
                                     dtype=dtype)

        # Broadcast tensors and scalars
        tensor = tf.convert_to_tensor(value, dtype=dtype)
        if tensor.get_shape() == tuple():
            return tf.fill(dims=shape, value=tensor)

        # Return original input if we cannot broadcast
        return tensor


def normalize_tensor(tensor, name=None):
    """Normalize the tensor so that all elements sum to 1.

    Args:
        tensor (Tensor): Input tensor.

    Returns:
        Tensor: Normalized tensor.
    """
    with tf.name_scope(name, "normalize_tensor", [tensor]):
        tensor = tf.convert_to_tensor(tensor)
        s = tf.reduce_sum(tensor)
        return tf.truediv(tensor, s)


def normalize_tensor_2D(tensor, num_weights=1, num_sums=1, name=None):
    """Reshape weight vector to a 2D tensor, and normalize such each row sums to 1.

    Args:
        tensor (Tensor): Input tensor.

    Returns:
        Tensor: Normalized tensor.
    """
    with tf.name_scope(name, "normalize_tensor_2D", [tensor]):
        tensor = tf.convert_to_tensor(tensor)
        tensor = tf.reshape(tensor, [num_sums, num_weights])
        s = tf.reduce_sum(tensor, axis=1, keep_dims=True)
        return tf.truediv(tensor, s)


def normalize_log_tensor_2D(tensor, num_weights=1, num_sums=1, name=None):
    """Reshape weight vector to a 2D tensor, and normalize such each row sums to 1.

    Args:
        tensor (Tensor): Input tensor.

    Returns:
        Tensor: Normalized tensor.
    """
    with tf.name_scope(name, "normalize_log_tensor_2D", [tensor]):
        tensor = tf.convert_to_tensor(tensor)
        tensor = tf.reshape(tensor, [num_sums, num_weights])
        log_sum = reduce_log_sum(tensor)
        # Normalize assuming that log_sum does not contain -inf
        normalized_log_tensor = tf.subtract(tensor, log_sum)
        return normalized_log_tensor


def reduce_log_sum(log_input, name=None):
    """Calculate log of a sum of elements of a tensor containing log values
    row-wise.

    Args:
        log_input (Tensor): Tensor containing log values.

    Returns:
        Tensor: The reduced tensor of shape ``(None, 1)``, where the first
        dimension corresponds to the first dimension of ``log_input``.
    """
    with tf.name_scope(name, "reduce_log_sum", [log_input]):
        log_max = tf.reduce_max(log_input, 1, keep_dims=True)
        # Compute the value assuming at least one input is not -inf
        log_rebased = tf.subtract(log_input, log_max)
        out_normal = log_max + tf.log(tf.reduce_sum(tf.exp(log_rebased),
                                                    1, keep_dims=True))
        # Check if all input values in a row are -inf (all non-log inputs are 0)
        # and produce output for that case
        # We use float('inf') for compatibility with Python<3.5
        # For Python>=3.5 we can use math.inf instead
        all_zero = tf.equal(log_max,
                            tf.constant(-float('inf'), dtype=log_input.dtype))
        out_zeros = tf.fill(tf.shape(out_normal),
                            tf.constant(-float('inf'), dtype=log_input.dtype))
        # Choose the output for each row
        return tf.where(all_zero, out_zeros, out_normal)


# log(x + y) = log(x) + log(1 + exp(log(y) - log(x)))
def reduce_log_sum_3D(log_input, transpose=True, name=None):
    """Calculate log of a sum of elements of a 3D tensor containing log values
    row-wise, with each slice representing a single sum node.

    Args:
        log_input (Tensor): Tensor containing log values.

    Returns:
        Tensor: The reduced tensor of shape ``(None, num_sums)``, where the first
         and the second dimensions corresponds to the second and first  dimensions
         of ``log_input``.
    """
    with tf.name_scope(name, "reduce_log_sum_3D", [log_input]):
        # log(x)
        log_max = tf.reduce_max(log_input, axis=-1, keep_dims=True)
        # Compute the value assuming at least one input is not -inf
        # r = log(y) - log(x)
        log_rebased = tf.subtract(log_input, log_max)
        # log(x) + log(1 + exp(r))???
        out_normal = log_max + tf.log(tf.reduce_sum(tf.exp(log_rebased),
                                                    axis=-1, keep_dims=True))
        # Check if all input values in a row are -inf (all non-log inputs are 0)
        # and produce output for that case
        all_zero = tf.equal(log_max,
                            tf.constant(-float('inf'), dtype=log_input.dtype))
        out_zeros = tf.fill(tf.shape(out_normal),
                            tf.constant(-float('inf'), dtype=log_input.dtype))
        # Choose the output for each row
        if transpose:
            return tf.transpose(tf.squeeze(tf.where(all_zero, out_zeros,
                                                    out_normal), -1))
        else:
            return tf.squeeze(tf.where(all_zero, out_zeros, out_normal), -1)


def concat_maybe(values, axis, name='concat'):
    """Concatenate ``values`` if there is more than one value. Oherwise, just
    forward value as is.

    Args:
        values (list of Tensor): Values to concatenate

    Returns:
        Tensor: Concatenated values.
    """
    if len(values) > 1:
        return tf.concat(values=values, axis=axis, name=name)
    else:
        return values[0]


def split_maybe(value, split_sizes, axis, name='split'):
    """Split ``value`` into multiple tensors of sizes given by ``split_sizes``.
    ``split_sizes`` must sum to the size of ``split_dim``. If only one split_size
    is given, the function does nothing and just forwards the value as the only
    split.

    Args:
        value (Tensor): The tensor to split.
        split_sizes (list of int): Sizes of each split.
        axis (int): The dimensions along which to split.

    Returns:
        list of Tensor: List of resulting tensors.
    """
    if len(split_sizes) > 1:
        return tf.split(value=value, num_or_size_splits=split_sizes,
                        axis=axis, name=name)
    else:
        return [value]


def print_tensor(*tensors):
    return tf.Print(tensors[0], tensors)
