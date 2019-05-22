from libspn.inference.type import InferenceType
from libspn.graph.op.base_sum import BaseSum
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
        latent_indicators (input_like): Input providing IndicatorLeaf of an explicit latent variable
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

    def __init__(self, *values, weights=None, latent_indicators=None,
                 inference_type=InferenceType.MARGINAL,
                 sample_prob=None, name="Sum"):
        super().__init__(
            *values, num_sums=1, weights=weights, latent_indicators=latent_indicators,
            inference_type=inference_type, sample_prob=sample_prob, name=name)
