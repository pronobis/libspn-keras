# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

from itertools import chain, combinations
import tensorflow as tf
from libspn.graph.scope import Scope
from libspn.graph.node import OpNode, Input
from libspn.inference.type import InferenceType
from libspn import utils
from libspn.exceptions import StructureError
from libspn.log import get_logger


class Products(OpNode):
    """A node representing a multiple products in an SPN.

    Args:
        *values (input_like): Inputs providing input values to this node.
            See :meth:`~libspn.Input.as_input` for possible values.
        num_prods (input_like): Input providing numbe of products modeled by
           this single products node.
        name (str): Name of the node.
    """

    logger = get_logger()
    info = logger.info

    def __init__(self, *values, num_prods=1, name="Products"):
        if not num_prods > 0:
            raise StructureError("In %s num_prods: %s need to be > 0" % self, num_prods)

        self._values = []
        self._num_prods = num_prods
        super().__init__(InferenceType.MARGINAL, name)
        self.set_values(*values)

    def serialize(self):
        data = super().serialize()
        data['values'] = [(i.node.name, i.indices) for i in self._values]
        return data

    def deserialize(self, data):
        super().deserialize(data)
        self.set_values()

    def deserialize_inputs(self, data, nodes_by_name):
        super().deserialize_inputs(data, nodes_by_name)
        self._values = tuple(Input(nodes_by_name[nn], i)
                             for nn, i in data['values'])

    @property
    @utils.docinherit(OpNode)
    def inputs(self):
        return self._values

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
        value_scopes = self._gather_input_scopes(*value_scopes)
        return [Scope.merge_scopes(chain.from_iterable(value_scopes))]

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
                    Products.info("%s is not decomposable with input value scopes %s",
                                  self, flat_value_scopes)
                    return None
        return self._compute_scope(*value_scopes)

    def _compute_value_common(self, *value_tensors):
        """Common actions when computing value."""
        # Check inputs
        if not self._values:
            raise StructureError("%s is missing input values." % self)
        # Prepare values
        value_tensors = self._gather_input_tensors(*value_tensors)
        if len(value_tensors) > 1:
            values = tf.concat(values=value_tensors, axis=1)
        else:
            values = value_tensors[0]
        if self._num_prods > 1:
            # Shape of values tensor = [Batch, (num_prods * num_vals)]
            # First, split the values tensor into 'num_prods' smaller tensors.
            # Then pack the split tensors together such that the new shape
            # of values tensor = [Batch, num_prods, num_vals]
            reshape = (-1, self._num_prods, int(values.shape[1].value /
                                                self._num_prods))
            reshaped_values = tf.reshape(values, shape=reshape)
            return reshaped_values
        else:
            return values

    def _compute_value(self, *value_tensors):
        values = self._compute_value_common(*value_tensors)
        return tf.reduce_prod(values, axis=-1, keep_dims=(False if
                              self._num_prods > 1 else True))

    def _compute_log_value(self, *value_tensors):
        values = self._compute_value_common(*value_tensors)
        return tf.reduce_sum(values, axis=-1,
                             keep_dims=(False if self._num_prods > 1 else True))

    def _compute_mpe_value(self, *value_tensors):
        return self._compute_value(*value_tensors)

    def _compute_log_mpe_value(self, *value_tensors):
        return self._compute_log_value(*value_tensors)

    def _compute_mpe_path(self, counts, *value_values, add_random=False, use_unweighted=False):
        # Check inputs
        if not self._values:
            raise StructureError("%s is missing input values." % self)

        def process_input(v_input, v_value, count):
            input_size = v_input.get_size(v_value)

            # Tile the counts if input is larger than 1
            return (tf.tile(count, [1, input_size])
                    if input_size > 1 else count)

        num_inputs_per_product = int(len(self.get_input_sizes()) / self._num_prods)

        # Split the counts matrix into num_prod columns, then clone each column
        # number-of-inputs-per-product times
        counts_list = list(chain(*[[c] * num_inputs_per_product for c in
                           tf.split(counts, self._num_prods, axis=1)]))

        # For each input of each product node, pass relevant counts to all
        # elements selected by indices
        value_counts = [(process_input(v_input, v_value, count), v_value)
                        for v_input, v_value, count
                        in zip(self._values, value_values, counts_list)]

        # TODO: Scatter to input tensors can be merged with tiling to reduce
        # the amount of operations.
        return self._scatter_to_input_tensors(*value_counts)

    def _compute_log_mpe_path(self, counts, *value_values, add_random=False, use_unweighted=False):
        return self._compute_mpe_path(counts, *value_values)
