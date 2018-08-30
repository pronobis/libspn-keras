# ------------------------------------------------------------------------
# Copyright (C) 2016 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

from libspn.inference.type import InferenceType
from libspn.graph.basesum import BaseSum
from libspn.learning.type import GradientType
import libspn.utils as utils


@utils.register_serializable
class ParSums(BaseSum):
    """A node representing multiple par-sums (which share the same input) in an SPN.

    Args:
        *values (input_like): Inputs providing input values to this node.
            See :meth:`~libspn.Input.as_input` for possible values.
        num_sums (int): Number of Sum ops modelled by this node.
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

    def __init__(self, *values, num_sums=1, weights=None, ivs=None,
                 inference_type=InferenceType.MARGINAL, sample_prob=None, 
                 dropconnect_keep_prob=None, gradient_type=GradientType.SOFT, name="ParallelSums"):
        super().__init__( 
            *values, num_sums=num_sums, weights=weights, ivs=ivs,
            inference_type=inference_type, sample_prob=sample_prob, 
            dropconnect_keep_prob=dropconnect_keep_prob,
            gradient_type=gradient_type, name=name)
