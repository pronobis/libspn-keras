from libspn.graph.op.block_sum import BlockSum
from libspn.graph.op.block_random_decompositions import BlockRandomDecompositions
from libspn.graph.op.block_permute_product import BlockPermuteProduct
from libspn.inference.type import InferenceType
import libspn.utils as utils
from libspn.log import get_logger
from libspn.graph.algorithms import traverse_graph


@utils.register_serializable
class BlockRootSum(BlockSum):

    logger = get_logger()
    info = logger.info

    """An abstract node representing sums in an SPN.

    Args:
        *values (input_like): Inputs providing input values to this node.
            See :meth:`~libspn.Input.as_input` for possible values.
        num_sums (int): Number of Sum ops modelled by this node.
        sum_sizes (list): A list of ints corresponding to the sizes of each sum. If both num_sums
                          and sum_sizes are given, we should have len(sum_sizes) == num_sums.
        batch_axis (int): The index of the batch axis.
        op_axis (int): The index of the op axis that contains the individual sums being modeled.
        reduce_axis (int): The axis over which to perform summing (or max for MPE)
        weights (input_like): Input providing weights node to this sum node.
            See :meth:`~libspn.Input.as_input` for possible values. If set
            to ``None``, the input is disconnected.
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
                 inference_type=InferenceType.MARGINAL, masked=False, sample_prob=None,
                 name="RootSum", input_format="SDBN",
                 output_format="SDBN"):
        super().__init__(
            child=child, num_sums_per_block=1, weights=weights, latent_indicators=latent_indicators,
            inference_type=inference_type, masked=masked, sample_prob=sample_prob,
            name=name, input_format=input_format, output_format=output_format)

        # Take care of generating the random decompositions
        randomize_node = traverse_graph(
            root=self, fun=lambda node: isinstance(node, BlockRandomDecompositions))

        if randomize_node is not None:
            factors = []
            traverse_graph(
                self, lambda n: factors.append(n.num_factors)
                if isinstance(n, BlockPermuteProduct) else None)
            randomize_node.generate_permutations(factors)
