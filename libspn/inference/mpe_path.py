from types import MappingProxyType
import tensorflow as tf
from libspn.inference.value import Value, LogValue
from libspn.graph.algorithms import compute_graph_up_down
from libspn.graph.op.base_sum import BaseSum
from libspn import utils


class MPEPath:
    """Assembles TF operations computing the branch counts for the MPE downward
    path through the SPN. It computes the number of times each branch was
    traveled by a complete subcircuit determined by the MPE value of the latent
    variables in the model.

    Args:
        value (Value or LogValue): Pre-computed SPN values.
        value_inference_type (InferenceType): The inference type used during the
            upwards pass through the SPN. Ignored if ``value`` is given.
        log (bool): If ``True``, calculate the value in the log space. Ignored
                    if ``value`` is given.
    """

    def __init__(self, value=None, value_inference_type=None, log=True, use_unweighted=False,
                 sample=False, sample_prob=None, matmul_or_conv=False):
        self._true_counts = {}
        self._actual_counts = {}
        self._log = log
        self._use_unweighted = use_unweighted
        self._sample = sample
        self._sample_prob = sample_prob
        # Create internal value generator
        self._value = value or LogValue(value_inference_type, matmul_or_conv=matmul_or_conv)

    @property
    def value(self):
        """Value or LogValue: Computed SPN values."""
        return self._value

    @property
    def counts(self):
        """dict: Dictionary indexed by node, where each value is a list of tensors
        computing the branch counts, based on the true value of the SPN's latent
        variable, for the inputs of the node."""
        return MappingProxyType(self._true_counts)

    @property
    def log(self):
        return self._log

    def get_mpe_path(self, root):
        """Assemble TF operations computing the true branch counts for the MPE
        downward path through the SPN rooted in ``root``.

        Args:
            root (Node): The root node of the SPN graph.
        """
        def down_fun(node, parent_vals):
            self._true_counts[node] = summed = self._accumulate_parents(*parent_vals)
            basesum_kwargs = dict(
                use_unweighted=self._use_unweighted, sample=self._sample,
                sample_prob=self._sample_prob)
            if node.is_op:
                kwargs = basesum_kwargs if isinstance(node, BaseSum) else dict()
                # Compute for inputs
                with tf.name_scope(node.name):
                    return node._compute_log_mpe_path(
                        summed, *[self._value.values[i.node] if i else None
                                  for i in node.inputs], **kwargs)

        # Generate values if not yet generated
        if not self._value.values:
            self._value.get_value(root)

        with tf.name_scope("TrueMPEPath"):
            # Compute the tensor to feed to the root node
            graph_input = self._graph_input(self._value.values[root])

            # Traverse the graph computing counts
            self._true_counts = {}
            compute_graph_up_down(root, down_fun=down_fun, graph_input=graph_input)

    @staticmethod
    @utils.lru_cache
    def _accumulate_parents(*parent_vals):
        # Sum up all parent vals
        return tf.add_n([pv for pv in parent_vals if pv is not None], name="AccumulateParents")

    @staticmethod
    @utils.lru_cache
    def _graph_input(root_value):
        return tf.ones_like(root_value)
