import tensorflow as tf
from libspn.graph.scope import Scope
from libspn.graph.node import VarNode
from libspn import conf
from libspn import utils
from libspn.utils.serialization import register_serializable


@register_serializable
class RawLeaf(VarNode):
    """A node representing a vector of continuous random variables as raw inputs. The inputs are
    not transformed to probabilities, and should therefore be probabilities themselves if the
    SPN has to compute a (log) PDF. It is mainly used for testing purposes and should generally
    be avoided.

    Args:
        feed (Tensor): Tensor feeding this node or ``None``. If ``None``,
                       an internal placeholder will be used to feed this node.
        num_vars (int): Number of variables in the vector.
        name (str): Name of the node
    """

    def __init__(self, feed=None, num_vars=1, name="RawLeaf"):
        if not isinstance(num_vars, int) or num_vars < 1:
            raise ValueError("num_vars must be a positive integer")
        self._num_vars = num_vars
        super().__init__(feed, name)

    def attach_feed(self, feed):
        super().attach_feed(feed)
        self._evidence = self._evidence_placeholder()

    def serialize(self):
        data = super().serialize()
        data['num_vars'] = self._num_vars
        return data

    def deserialize(self, data):
        self._num_vars = data['num_vars']
        super().deserialize(data)

    @property
    def evidence(self):
        return self._evidence

    @property
    def num_vars(self):
        return self._num_vars

    def _create_placeholder(self):
        """Create a placeholder that will be used to feed this variable when
        no other feed is available.

        Returns:
            Tensor: A TF placeholder of shape ``[None, num_vars]``, where the
            first dimension corresponds to the batch size.
        """
        return tf.placeholder(conf.dtype, [None, self._num_vars])

    def _evidence_placeholder(self):
        return tf.placeholder_with_default(
            tf.cast(tf.ones_like(self._feed, dtype=conf.dtype), tf.bool), [None, self._num_vars])

    def _compute_out_size(self):
        return self._num_vars

    def _compute_scope(self):
        return [Scope(self, i) for i in range(self._num_vars)]

    @utils.lru_cache
    def _compute_log_value(self):
        # We used identity, since this way we can feed and fetch this node
        # and there is an operation in TensorBoard even if the internal
        # placeholder is not used for feeding.
        return tf.log(self._feed)

    def _compute_mpe_state(self, counts):
        return counts
