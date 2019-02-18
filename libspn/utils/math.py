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
from libspn import conf, utils as utils
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
                return_tensor = tf.gather(params, indices[0], axis=1)

            if ind_2D:
                # Indices is 2D, so insert an extra dimension to the output
                return tf.expand_dims(return_tensor, axis=-2)
            else:
                return return_tensor
        else:
            # Multiple rows of indices
            indices_cols = max([ind.size for ind in indices])
            padding = False
            for i, indf in enumerate(indices):
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
                with_zeros = tf.concat(values=([0], params), axis=0)
                gather_indices = np.zeros(num_out_cols, dtype=int)
                gather_indices[indices] = np.arange(indices.size) + 1
                return tf.gather(with_zeros, gather_indices, axis=1)
            else:
                zero_col = tf.zeros((tf.shape(params)[0], 1),
                                    dtype=params.dtype)
                with_zeros = tf.concat(values=(zero_col, params), axis=1)
                gather_indices = np.zeros(num_out_cols, dtype=int)
                gather_indices[indices] = np.arange(indices.size) + 1
                return tf.gather(with_zeros, gather_indices, axis=1)


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


def print_tensor(*tensors):
    return tf.Print(tensors[0], tensors)


@utils.lru_cache
def cwise_add(a, b):
    """Component-wise addition of two tensors. Added explicitly for readability elsewhere and
    for straightforward memoization.

    Args:
        a (Tensor): Left-hand side.
        b (Tensor): Right-hand side.

    Returns:
        A component wise addition of ``a`` and ``b``.
    """
    return a + b


@utils.lru_cache
def cwise_mul(a, b):
    """Component-wise multiplication of two tensors. Added explicitly for readability elsewhere
    and for straightforward memoization.

    Args:
        a (Tensor): Left-hand side.
        b (Tensor): Right-hand side.

    Returns:
        A component wise multiplication of ``a`` and ``b``.
    """
    return a * b