# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
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
from libspn.utils.serialization import register_serializable
from libspn.graph.basesum import BaseSum


class Sum(BaseSum):

    def __init__(self, *values, weights=None, ivs=None, inference_type=InferenceType.MARGINAL,
                 name="Sum"):
        super().__init__(
            *values, num_sums=1, weights=weights, ivs=ivs, inference_type=inference_type,
            name=name)

    def _compute_scope(self, weight_scopes, ivs_scopes, *value_scopes):
        if not self._values:
            raise StructureError("%s is missing input values" % self)
        _, ivs_scopes, *value_scopes = self._gather_input_scopes(weight_scopes,
                                                                 ivs_scopes,
                                                                 *value_scopes)
        flat_value_scopes = list(chain.from_iterable(value_scopes))
        if self._ivs:
            flat_value_scopes.extend(ivs_scopes)
        return [Scope.merge_scopes(flat_value_scopes)]

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
        flat_value_scopes = list(chain.from_iterable(value_scopes_))
        # IVs
        if self._ivs:
            # Verify number of IVs
            if len(ivs_scopes_) != len(flat_value_scopes):
                raise StructureError("Number of IVs (%s) and values (%s) does "
                                     "not match for %s"
                                     % (len(ivs_scopes_), len(flat_value_scopes),
                                        self))
            # Check if scope of all IVs is just one and the same variable
            if len(Scope.merge_scopes(ivs_scopes_)) > 1:
                return None
        # Check sum for completeness wrt values
        first_scope = flat_value_scopes[0]
        if any(s != first_scope for s in flat_value_scopes[1:]):
            self.info("%s is not complete with input value scopes %s",
                      self, flat_value_scopes)
            return None
        return self._compute_scope(weight_scopes, ivs_scopes, *value_scopes)
#
#
#
# @register_serializable
# class Sum(OpNode):
#     """A node representing a single sum in an SPN.
#
#     Args:
#         *values (input_like): Inputs providing input values to this node.
#             See :meth:`~libspn.Input.as_input` for possible values.
#         weights (input_like): Input providing weights node to this sum node.
#             See :meth:`~libspn.Input.as_input` for possible values. If set
#             to ``None``, the input is disconnected.
#         ivs (input_like): Input providing IVs of an explicit latent variable
#             associated with this sum node. See :meth:`~libspn.Input.as_input`
#             for possible values. If set to ``None``, the input is disconnected.
#         name (str): Name of the node.
#
#     Attributes:
#         inference_type(InferenceType): Flag indicating the preferred inference
#                                        type for this node that will be used
#                                        during value calculation and learning.
#                                        Can be changed at any time and will be
#                                        used during the next inference/learning
#                                        op generation.
#     """
#
#     __logger = get_logger()
#     __info = __logger.info
#
#     def __init__(self, *values, weights=None, ivs=None,
#                  inference_type=InferenceType.MARGINAL, name="Sum"):
#         super().__init__(inference_type, name)
#         self.set_values(*values)
#         self.set_weights(weights)
#         self.set_ivs(ivs)
#
#     def serialize(self):
#         data = super().serialize()
#         data['values'] = [(i.node.name, i.indices) for i in self._values]
#         if self._weights:
#             data['weights'] = (self._weights.node.name, self._weights.indices)
#         if self._ivs:
#             data['ivs'] = (self._ivs.node.name, self._ivs.indices)
#         return data
#
#     def deserialize(self, data):
#         super().deserialize(data)
#         self.set_values()
#         self.set_weights()
#         self.set_ivs()
#
#     def deserialize_inputs(self, data, nodes_by_name):
#         super().deserialize_inputs(data, nodes_by_name)
#         self._values = tuple(Input(nodes_by_name[nn], i)
#                              for nn, i in data['values'])
#         weights = data.get('weights', None)
#         if weights:
#             self._weights = Input(nodes_by_name[weights[0]], weights[1])
#         ivs = data.get('ivs', None)
#         if ivs:
#             self._ivs = Input(nodes_by_name[ivs[0]], ivs[1])
#
#     @property
#     @utils.docinherit(OpNode)
#     def inputs(self):
#         return (self._weights, self._ivs) + self._values
#
#     @property
#     def weights(self):
#         """Input: Weights input."""
#         return self._weights
#
#     def set_weights(self, weights=None):
#         """Set the weights input.
#
#         Args:
#             weights (input_like): Input providing weights node to this sum node.
#                 See :meth:`~libspn.Input.as_input` for possible values. If set
#                 to ``None``, the input is disconnected.
#         """
#         weights, = self._parse_inputs(weights)
#         if weights and not isinstance(weights.node, Weights):
#             raise StructureError("%s is not Weights" % weights.node)
#         self._weights = weights
#
#     @property
#     def ivs(self):
#         """Input: IVs input."""
#         return self._ivs
#
#     def set_ivs(self, ivs=None):
#         """Set the IVs input.
#
#         ivs (input_like): Input providing IVs of an explicit latent variable
#             associated with this sum node. See :meth:`~libspn.Input.as_input`
#             for possible values. If set to ``None``, the input is disconnected.
#         """
#         self._ivs, = self._parse_inputs(ivs)
#
#     @property
#     def values(self):
#         """list of Input: List of value inputs."""
#         return self._values
#
#     def set_values(self, *values):
#         """Set the inputs providing input values to this node. If no arguments
#         are given, all existing value inputs get disconnected.
#
#         Args:
#             *values (input_like): Inputs providing input values to this node.
#                 See :meth:`~libspn.Input.as_input` for possible values.
#         """
#         self._values = self._parse_inputs(*values)
#
    # def add_values(self, *values):
    #     """Add more inputs providing input values to this node.
    #
    #     Args:
    #         *values (input_like): Inputs providing input values to this node.
    #             See :meth:`~libspn.Input.as_input` for possible values.
    #     """
    #     self._values = self._values + self._parse_inputs(*values)
#
#     def generate_weights(self, init_value=1, trainable=True, input_sizes=None,
#                          log=False, name=None):
#         """Generate a weights node matching this sum node and connect it to
#         this sum.
#
#         The function calculates the number of weights based on the number
#         of input values of this sum. Therefore, weights should be generated
#         once all inputs are added to this node.
#
#         Args:
#             init_value: Initial value of the weights. For possible values, see
#                 :meth:`~libspn.utils.broadcast_value`.
#             trainable (bool): See :class:`~libspn.Weights`.
#             input_sizes (list of int): Pre-computed sizes of each input of
#                 this node.  If given, this function will not traverse the graph
#                 to discover the sizes.
#             log (bool): If "True", the weights are represented in log space.
#             name (str): Name of the weighs node. If ``None`` use the name of the
#                         sum + ``_Weights``.
#
#         Return:
#             Weights: Generated weights node.
#         """
#         if not self._values:
#             raise StructureError("%s is missing input values" % self)
#         if name is None:
#             name = self._name + "_Weights"
#         # Count all input values
#         if not input_sizes:
#             input_sizes = self.get_input_sizes()
#         num_values = sum(input_sizes[2:])  # Skip ivs, weights
#         # Generate weights
#         weights = Weights(init_value=init_value,
#                           num_weights=num_values,
#                           log=log,
#                           trainable=trainable, name=name)
#         self.set_weights(weights)
#         return weights
#
#     def generate_ivs(self, feed=None, name=None):
#         """Generate an IVs node matching this sum node and connect it to
#         this sum.
#
#         IVs should be generated once all inputs are added to this node,
#         otherwise the number of IVs will be incorrect.
#
#         Args:
#             feed (Tensor): See :class:`~libspn.IVs`.
#             name (str): Name of the IVs node. If ``None`` use the name of the
#                         sum + ``_IVs``.
#
#         Return:
#             IVs: Generated IVs node.
#         """
#         if not self._values:
#             raise StructureError("%s is missing input values" % self)
#         if name is None:
#             name = self._name + "_IVs"
#         # Count all input values
#         num_values = sum(len(v.indices) if v.indices is not None
#                          else v.node.get_out_size()
#                          for v in self._values)
#         ivs = IVs(feed=feed, num_vars=1, num_vals=num_values, name=name)
#         self.set_ivs(ivs)
#         return ivs
#
#     @property
#     def _const_out_size(self):
#         return True
#
#     def _compute_out_size(self, *input_out_sizes):
#         return 1
#
#     def _compute_scope(self, weight_scopes, ivs_scopes, *value_scopes):
#         if not self._values:
#             raise StructureError("%s is missing input values" % self)
#         _, ivs_scopes, *value_scopes = self._gather_input_scopes(weight_scopes,
#                                                                  ivs_scopes,
#                                                                  *value_scopes)
#         flat_value_scopes = list(chain.from_iterable(value_scopes))
#         if self._ivs:
#             flat_value_scopes.extend(ivs_scopes)
#         return [Scope.merge_scopes(flat_value_scopes)]
#
#     def _compute_valid(self, weight_scopes, ivs_scopes, *value_scopes):
#         if not self._values:
#             raise StructureError("%s is missing input values" % self)
#         _, ivs_scopes_, *value_scopes_ = self._gather_input_scopes(weight_scopes,
#                                                                    ivs_scopes,
#                                                                    *value_scopes)
#         # If already invalid, return None
#         if (any(s is None for s in value_scopes_)
#                 or (self._ivs and ivs_scopes_ is None)):
#             return None
#         flat_value_scopes = list(chain.from_iterable(value_scopes_))
#         # IVs
#         if self._ivs:
#             # Verify number of IVs
#             if len(ivs_scopes_) != len(flat_value_scopes):
#                 raise StructureError("Number of IVs (%s) and values (%s) does "
#                                      "not match for %s"
#                                      % (len(ivs_scopes_), len(flat_value_scopes),
#                                         self))
#             # Check if scope of all IVs is just one and the same variable
#             if len(Scope.merge_scopes(ivs_scopes_)) > 1:
#                 return None
#         # Check sum for completeness wrt values
#         first_scope = flat_value_scopes[0]
#         if any(s != first_scope for s in flat_value_scopes[1:]):
#             self.__info("%s is not complete with input value scopes %s",
#                         self, flat_value_scopes)
#             return None
#         return self._compute_scope(weight_scopes, ivs_scopes, *value_scopes)
#
#     @utils.lru_cache
#     def _compute_value_common(self, weight_tensor, ivs_tensor, *value_tensors):
#         """Common actions when computing value."""
#         # Check inputs
#         if not self._values:
#             raise StructureError("%s is missing input values" % self)
#         if not self._weights:
#             raise StructureError("%s is missing weights" % self)
#         # Prepare values
#         weight_tensor, ivs_tensor, *value_tensors = self._gather_input_tensors(
#             weight_tensor, ivs_tensor, *value_tensors)
#         values = utils.concat_maybe(value_tensors, 1)
#         return weight_tensor, ivs_tensor, values
#
#     @utils.lru_cache
#     def _compute_value(self, weight_tensor, ivs_tensor, *value_tensors):
#         weight_tensor, ivs_tensor, values = self._compute_value_common(
#             weight_tensor, ivs_tensor, *value_tensors)
#         values_selected = values * ivs_tensor if self._ivs else values
#         return tf.matmul(values_selected, tf.reshape(weight_tensor, [-1, 1]))
#
#     @utils.lru_cache
#     def _compute_log_value(self, weight_tensor, ivs_tensor, *value_tensors):
#         weight_tensor, ivs_tensor, values = self._compute_value_common(
#             weight_tensor, ivs_tensor, *value_tensors)
#         values_selected = values + ivs_tensor if self._ivs else values
#         values_weighted = values_selected + weight_tensor
#         return utils.reduce_log_sum(values_weighted)
#
#     @utils.lru_cache
#     def _compute_mpe_value(self, weight_tensor, ivs_tensor, *value_tensors):
#         weight_tensor, ivs_tensor, values = self._compute_value_common(
#             weight_tensor, ivs_tensor, *value_tensors)
#         values_selected = values * ivs_tensor if self._ivs else values
#         values_weighted = values_selected * weight_tensor
#         return tf.reduce_max(values_weighted, 1, keep_dims=True)
#
#     @utils.lru_cache
#     def _compute_log_mpe_value(self, weight_tensor, ivs_tensor, *value_tensors):
#         weight_tensor, ivs_tensor, values = self._compute_value_common(
#             weight_tensor, ivs_tensor, *value_tensors)
#         values_selected = values + ivs_tensor if self._ivs else values
#         values_weighted = values_selected + weight_tensor
#         return tf.reduce_max(values_weighted, 1, keep_dims=True)
#
#     @utils.lru_cache
#     def _compute_mpe_path_common(self, values_weighted, counts, weight_value,
#                                  ivs_value, *value_values):
#         # Propagate the counts to the max value
#         max_indices = tf.argmax(values_weighted, dimension=1)
#         max_counts = utils.scatter_values(params=tf.squeeze(counts, axis=1),
#                                           indices=max_indices,
#                                           num_out_cols=values_weighted.shape[1].value)
#         # Split the counts to value inputs
#         _, _, *value_sizes = self.get_input_sizes(None, None, *value_values)
#         max_counts_split = utils.split_maybe(max_counts, value_sizes, 1)
#         return self._scatter_to_input_tensors(
#             (max_counts, weight_value),  # Weights
#             (max_counts, ivs_value),  # IVs
#             *[(t, v) for t, v in zip(max_counts_split, value_values)])  # Values

    # @utils.lru_cache
    # def _compute_mpe_path(self, counts, weight_value, ivs_value, *value_values,
    #                       add_random=None, use_unweighted=False, with_ivs=True):
    #     # Get weighted, IV selected values
    #     weight_value, ivs_value, values = self._compute_value_common(
    #         weight_value, ivs_value, *value_values)
    #     values_selected = values * ivs_value if self._ivs and with_ivs else values
    #     values_weighted = values_selected * weight_value
    #     return self._compute_mpe_path_common(
    #         values_weighted, counts, weight_value, ivs_value, *value_values)
    #
    # @utils.lru_cache
    # def _compute_log_mpe_path(self, counts, weight_value, ivs_value, *value_values,
    #                           add_random=None, use_unweighted=False, with_ivs=True):
    #     # Get weighted, IV selected values
    #     weight_value, ivs_value, values = self._compute_value_common(
    #         weight_value, ivs_value, *value_values)
    #     values_selected = values + ivs_value if self._ivs and with_ivs else values
    #
    #     # WARN USING UNWEIGHTED VALUE
    #     if not use_unweighted or any(v.node.is_var for v in self._values):
    #         values_weighted = values_selected + weight_value
    #     else:
    #         values_weighted = values_selected
    #
    #     # / USING UNWEIGHTED VALUE
    #
    #     # WARN ADDING RANDOM NUMBERS
    #     if add_random is not None:
    #         values_weighted = tf.add(values_weighted, tf.random_uniform(
    #             shape=(tf.shape(values_weighted)[0],
    #                    values_weighted.shape[1].value),
    #             minval=0, maxval=add_random,
    #             dtype=conf.dtype))
    #     # /ADDING RANDOM NUMBERS
    #
    #     return self._compute_mpe_path_common(
    #         values_weighted, counts, weight_value, ivs_value, *value_values)
#
#     @utils.lru_cache
#     def _compute_gradient(self, gradients, weight_value, ivs_value,
#                           *value_values, with_ivs=True):
#         weight_value, ivs_value, values = self._compute_value_common(
#             weight_value, ivs_value, *value_values)
#
#         weight_gradients = gradients * values
#         output_gradients = gradients * weight_value
#
#         # Split the output_gradients to value inputs
#         _, _, *value_sizes = self.get_input_sizes(None, None, *value_values)
#         output_gradients_split = utils.split_maybe(output_gradients, value_sizes, 1)
#
#         return self._scatter_to_input_tensors(
#             (weight_gradients, weight_value),  # Weights
#             (weight_gradients, ivs_value),  # IVs
#             *[(t, v) for t, v in zip(output_gradients_split, value_values)])  # Values
#
#     @utils.lru_cache
#     def _compute_log_gradient(self, gradients, weight_value, ivs_value,
#                               *value_values, with_ivs=True):
#         weight_value, ivs_value, values = self._compute_value_common(
#             weight_value, ivs_value, *value_values)
#
#         values_selected = values + ivs_value if self._ivs and with_ivs else values
#         values_weighted = values_selected + weight_value
#
#         log_sum = utils.reduce_log_sum(values_weighted)
#         weight_gradients = gradients * tf.exp(values_weighted - log_sum)
#
#         output_gradients = weight_gradients
#
#         # Split the output_gradients to value inputs
#         _, _, *value_sizes = self.get_input_sizes(None, None, *value_values)
#         output_gradients_split = utils.split_maybe(output_gradients, value_sizes, 1)
#
#         return self._scatter_to_input_tensors(
#             (weight_gradients, weight_value),  # Weights
#             (weight_gradients, ivs_value),  # IVs
#             *[(t, v) for t, v in zip(output_gradients_split, value_values)])  # Values
#
#     @utils.lru_cache
#     def sum_exponents(self, values_weighted):
#         log_max = tf.reduce_max(values_weighted, 1, keep_dims=True)
#         log_rebased = tf.subtract(values_weighted, log_max)
#         return tf.reduce_sum(tf.exp(log_rebased), 1, keep_dims=True)
#
#     @utils.lru_cache
#     def _compute_log_gradient_log(self, gradients, weight_value, ivs_value,
#                                   *value_values, with_ivs=True):
#         weight_value, ivs_value, values = self._compute_value_common(
#             weight_value, ivs_value, *value_values)
#
#         values_selected = values + ivs_value if self._ivs and with_ivs else values
#         values_weighted = values_selected + weight_value
#
#         log_max = tf.reduce_max(values_weighted, axis=1, keep_dims=True)
#         log_rebased = tf.subtract(values_weighted, log_max)
#         expo_logs = tf.exp(log_rebased)
#         summed_exponents = tf.reduce_sum(expo_logs, axis=1, keep_dims=True)
#
#         max_indices = tf.argmax(values_weighted, axis=1)
#         expos_excl_max = tf.one_hot(max_indices, on_value=0.0, off_value=1.0,
#                                     depth=values_weighted.get_shape()[1],
#                                     dtype=conf.dtype) * tf.truediv(expo_logs,
#                                                                    summed_exponents)
#         summed_expos_excl_max = tf.reduce_sum(expos_excl_max, axis=1, keep_dims=True)
#         max_weight_gradient = 1.0 - summed_expos_excl_max
#         max_weight_gradient_scattered = \
#             tf.one_hot(max_indices, depth=values_weighted.get_shape()[1],
#                        dtype=conf.dtype) * max_weight_gradient
#         weight_gradients = gradients * (expos_excl_max + max_weight_gradient_scattered)
#
#         output_gradients = weight_gradients
#
#         # Split the output_gradients to value inputs
#         _, _, *value_sizes = self.get_input_sizes(None, None, *value_values)
#         output_gradients_split = utils.split_maybe(output_gradients, value_sizes, 1)
#
#         return self._scatter_to_input_tensors(
#             (weight_gradients, weight_value),  # Weights
#             (weight_gradients, ivs_value),  # IVs
#             *[(t, v) for t, v in zip(output_gradients_split, value_values)])  # Values
