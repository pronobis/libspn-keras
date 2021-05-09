from libspn_keras.sum_ops.base import SumOpBase
from libspn_keras.sum_ops.grad_backprop import SumOpGradBackprop

_DEFAULT_SUM_OP: SumOpBase = SumOpGradBackprop()


def get_default_sum_op() -> SumOpBase:
    """
    Obtain default sum op.

    Returns:
        The default sum op.
    """
    return _DEFAULT_SUM_OP


def set_default_sum_op(op: SumOpBase) -> None:
    """
    Set default sum op to conveniently use it throughout an SPN architecture.

    Args:
        op: Implementation of sum op with corresponding backward pass definitions
    """
    global _DEFAULT_SUM_OP
    _DEFAULT_SUM_OP = op
