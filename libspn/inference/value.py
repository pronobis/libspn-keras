# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

import tensorflow as tf
from types import MappingProxyType
from libspn.graph.algorithms import compute_graph_up
from libspn.inference.type import InferenceType
from libspn.graph.basesum import BaseSum


class Value:
    """Assembles TF operations computing the values of nodes of the SPN during
    an upwards pass. The value can be either an SPN value (marginal inference)
    or an MPN value (MPE inference) or a mixture of both.

    Args:
        inference_type (InferenceType): Determines the type of inference that
            should be used. If set to ``None``, the inference type is specified
            by the ``inference_type`` flag of the node. If set to ``MARGINAL``,
            marginal inference will be used for all nodes. If set to ``MPE``,
            MPE inference will be used for all nodes.
    """

    def __init__(self, inference_type=None, dropconnect_keep_prob=None, name="Value"):
        self._inference_type = inference_type
        self._values = {}
        self._dropconnect_keep_prob = dropconnect_keep_prob
        self._name = name

    @property
    def values(self):
        """dict: A dictionary of ``Tensor`` indexed by the SPN node containing
        operations computing value for each node."""
        return MappingProxyType(self._values)

    def log(self):
        return False

    def get_value(self, root):
        """Assemble a TF operation computing the values of nodes of the SPN
        rooted in ``root``.

        Returns the operation computing the value for the ``root``. Operations
        computing values for other nodes can be obtained using :obj:`values`.

        Args:
            root (Node): The root node of the SPN graph.

        Returns:
            Tensor: A tensor of shape ``[None, num_outputs]``, where the first
            dimension corresponds to the batch size.
        """
        def fun(node, *args):
            if self._dropconnect_keep_prob and isinstance(node, BaseSum):
                kwargs = dict(dropconnect_keep_prob=self._dropconnect_keep_prob)
            else:
                kwargs = dict()
            with tf.name_scope(node.name):
                if (self._inference_type == InferenceType.MARGINAL
                    or (self._inference_type is None and
                        node.inference_type == InferenceType.MARGINAL)):
                    return node._compute_value(*args, **kwargs)
                else:
                    return node._compute_mpe_value(*args, **kwargs)

        self._values = {}
        with tf.name_scope(self._name):
            return compute_graph_up(root, val_fun=fun,
                                    all_values=self._values)


class LogValue:
    """Assembles a TF operation computing the log values of nodes of the SPN
    during an upwards pass. The value can be either an SPN value (marginal
    inference) or an MPN value (MPE inference) or a mixture of both.

    Args:
        inference_type (InferenceType): Determines the type of inference that
            should be used. If set to ``None``, the inference type is specified
            by the ``inference_type`` flag of the node. If set to ``MARGINAL``,
            marginal inference will be used for all nodes. If set to ``MPE``,
            MPE inference will be used for all nodes.
    """

    def __init__(self, inference_type=None, dropconnect_keep_prob=None, name="LogValue"):
        self._inference_type = inference_type
        self._values = {}
        self._dropconnect_keep_prob = dropconnect_keep_prob
        self._name = name

    @property
    def values(self):
        """dict: A dictionary of ``Tensor`` indexed by the SPN node containing
        operations computing log value for each node."""
        return MappingProxyType(self._values)

    def log(self):
        return True

    def get_value(self, root):
        """Assemble TF operations computing the log values of nodes of the SPN
        rooted in ``root``.

        Returns the operation computing the log value for the ``root``.
        Operations computing log values for other nodes can be obtained using
        :obj:`values`.

        Args:
            root: Root node of the SPN.

        Returns:
            Tensor: A tensor of shape ``[None, num_outputs]``, where the first
            dimension corresponds to the batch size.
        """
        def fun(node, *args):
            if self._dropconnect_keep_prob and isinstance(node, BaseSum):
                kwargs = dict(
                    dropconnect_keep_prob=self._dropconnect_keep_prob)
            else:
                kwargs = dict()
            with tf.name_scope(node.name):
                if (self._inference_type == InferenceType.MARGINAL
                    or (self._inference_type is None and
                        node.inference_type == InferenceType.MARGINAL)):
                    return node._compute_log_value(*args, **kwargs)
                else:
                    return node._compute_log_mpe_value(*args, **kwargs)

        self._values = {}
        with tf.name_scope(self._name):
            return compute_graph_up(root, val_fun=fun, all_values=self._values)
