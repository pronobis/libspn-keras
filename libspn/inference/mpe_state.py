import tensorflow as tf
from libspn.inference.mpe_path import MPEPath


class MPEState:
    """Assembles TF operations computing MPE state for an SPN.

    Args:
        mpe_path (MPEPath): Pre-computed MPE_path.
        value_inference_type (InferenceType): The inference type used during the
            upwards pass through the SPN. Ignored if ``mpe_path`` is given.
        log (bool): If ``True``, calculate the value in the log space. Ignored
                    if ``mpe_path`` is given.
    """

    def __init__(self, mpe_path=None, log=True, value_inference_type=None,
                 matmul_or_conv=False):
        # Create internal MPE path generator
        if mpe_path is None:
            self._mpe_path = MPEPath(log=log, value_inference_type=value_inference_type,
                                     use_unweighted=False, matmul_or_conv=matmul_or_conv)
        else:
            self._mpe_path = mpe_path

    @property
    def mpe_path(self):
        """MPEPath: Computed MPE path."""
        return self._mpe_path

    def get_state(self, root, *var_nodes):
        """Assemble TF operations computing the MPE state of the given SPN
        variables for the SPN rooted in ``root``.

        Args:
            root (Node): The root node of the SPN graph.
            *var_nodes (VarNode): Variable nodes for which the state should
                                  be computed.

        Returns:
            list of Tensor: A list of tensors containing the MPE state for the
            variable nodes.
        """
        # Generate path if not yet generated
        if not self._mpe_path.counts:
            self._mpe_path.get_mpe_path(root)

        with tf.name_scope("MPEState"):
            return tuple(var_node._compute_mpe_state(
                self._mpe_path.counts[var_node])
                for var_node in var_nodes)
