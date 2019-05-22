import tensorflow as tf
from libspn.graph.scope import Scope
from libspn.graph.node import VarNode
from libspn import conf
from libspn import utils
from libspn.utils.serialization import register_serializable


@register_serializable
class IndicatorLeaf(VarNode):
    """A node representing multiple random variables in the form of indicator
    variables. Each random variable is assumed to take the same number of
    possible values ``[0, 1, ..., num_vals-1]``. If the value of the random
    variable is negative (e.g. -1), all indicators will be set to 1. For a
    Bernoulli random variable, use ``num_vals=2``.

    Args:
        feed (Tensor): Tensor feeding this node or ``None``. If ``None``,
                       an internal placeholder will be used to feed this node.
        num_vars (int): Number of random variables.
        num_vals (int): Number of values of each random variable.
        name (str): Name of the node
    """

    def __init__(self, feed=None, num_vars=1, num_vals=2, name="IndicatorLeaf"):
        if not isinstance(num_vars, int) or num_vars < 1:
            raise ValueError("num_vars must be a positive integer")
        if not isinstance(num_vals, int) or num_vals < 2:
            raise ValueError("num_vals must be >1")
        self._num_vars = num_vars
        self._num_vals = num_vals
        super().__init__(feed, name)

    @property
    def num_vars(self):
        """int: Number of random variables of the IndicatorLeaf."""
        return self._num_vars

    @property
    def num_vals(self):
        """int: Number of values of each random variable."""
        return self._num_vals

    def serialize(self):
        data = super().serialize()
        data['num_vars'] = self._num_vars
        data['num_vals'] = self._num_vals
        return data

    def deserialize(self, data):
        self._num_vars = data['num_vars']
        self._num_vals = data['num_vals']
        super().deserialize(data)

    def _create_placeholder(self):
        """Create a placeholder that will be used to feed this variable when
        no other feed is available.

        Returns:
            Tensor: An integer TF placeholder of shape ``[None, num_vars]``,
            where the first dimension corresponds to the batch size.
        """
        return tf.placeholder(tf.int32, [None, self._num_vars])

    def _compute_out_size(self):
        return self._num_vars * self._num_vals

    def _compute_scope(self):
        return [Scope(self, i)
                for i in range(self._num_vars)
                for _ in range(self._num_vals)]
    
    @utils.lru_cache
    def _compute_log_value(self):
        """Assemble the TF operations computing the output value of the node
        for a normal upwards pass.

        This function converts the integer inputs to indicators.

        Returns:
            Tensor: A tensor of shape ``[None, num_vars*num_vals]``, where the
            first dimension corresponds to the batch size.
        """
        # The output type has to be conf.dtype otherwise MatMul will
        # complain about being unable to mix types
        oh = tf.one_hot(self._feed, self._num_vals, dtype=conf.dtype)
        # Detect negative input values and convert them to all IndicatorLeaf equal to 1
        neg = tf.expand_dims(tf.cast(tf.less(self._feed, 0), dtype=conf.dtype), axis=-1)
        oh = tf.add(oh, neg)
        # Reshape
        return tf.log(tf.reshape(oh, [-1, self._num_vars * self._num_vals]))

    @utils.lru_cache
    def _compute_mpe_state(self, counts):
        r = tf.reshape(counts, (-1, self._num_vars, self._num_vals))
        return tf.argmax(r, axis=2)
