from libspn.graph.op.block_sum import BlockSum
from libspn.graph.op.block_random_decompositions import BlockRandomDecompositions
from libspn.graph.op.block_permute_product import BlockPermuteProduct
from libspn.inference.type import InferenceType
import libspn.utils as utils
from libspn.log import get_logger
from libspn.graph.algorithms import traverse_graph


@utils.register_serializable
class BlockRootSum(BlockSum):

    """
    This node can be used for putting a root node on top of an SPN built with ``BlockNode`` s. This
    simply assumes there's only a single decomposition and a single scope in the preceding layer,
    so that the root can join all nodes by means of a sum. This node will also ensure correct
    generation of decompositions if there exists a ``BlockRandomDecompositions`` descendant.

    Args:
        child (input_like): Child for this node.
            See :meth:`~libspn.Input.as_input` for possible values.
        weights (input_like): Input providing weights node to this sum node.
            See :meth:`~libspn.Input.as_input` for possible values. If set
            to ``None``, the input is disconnected.
        sample_prob (float): Probability for sampling on MPE path computation.
        latent_indicators (IndicatorLeaf): Latent indicators (can be used for classification tasks)
        name (str): Name of the node.

    Attributes:
        inference_type(InferenceType): Flag indicating the preferred inference
                                       type for this node that will be used
                                       during value calculation and learning.
                                       Can be changed at any time and will be
                                       used during the next inference/learning
                                       op generation.
    """

    def __init__(self, child, weights=None, latent_indicators=None,
                 inference_type=InferenceType.MARGINAL, sample_prob=None,
                 name="BlockRootSum"):
        super().__init__(
            child=child, num_sums_per_block=1, weights=weights, latent_indicators=latent_indicators,
            inference_type=inference_type, sample_prob=sample_prob, name=name)

        # Take care of generating the random decompositions
        randomize_node = traverse_graph(
            root=self, fun=lambda node: isinstance(node, BlockRandomDecompositions))

        if randomize_node is not None:
            factors = []
            traverse_graph(
                self, lambda n: factors.append(n.num_factors)
                if isinstance(n, BlockPermuteProduct) else None)
            randomize_node.generate_permutations(factors)
