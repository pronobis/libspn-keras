# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

from libspn.inference.type import InferenceType
from libspn.graph.basesum import BaseSum
import libspn.utils as utils


@utils.register_serializable
class Sum(BaseSum):
    """A node representing a single sum in an SPN.

    Args:
        *values (input_like): Inputs providing input values to this node.
            See :meth:`~libspn.Input.as_input` for possible values.
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

    def __init__(self, *values, weights=None, ivs=None, inference_type=InferenceType.MARGINAL,
                 sample_prob=None, dropconnect_keep_prob=None, name="Sum"):
        super().__init__(
            *values, num_sums=1, weights=weights, ivs=ivs, inference_type=inference_type,
            sample_prob=sample_prob, dropconnect_keep_prob=dropconnect_keep_prob, name=name)
