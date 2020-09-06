from libspn_keras.sum_ops import SumOpBase, SumOpGradBackprop

_DEFAULT_SUM_OP = None


def get_default_sum_op() -> SumOpBase:
    """
    Get default sum op.

    Returns:
        A ``SumOpBase`` instance that was set with ``set_default_sum_op``
    """
    return SumOpGradBackprop() if _DEFAULT_SUM_OP is None else _DEFAULT_SUM_OP


def set_default_sum_op(op: SumOpBase) -> None:
    """
    Set default sum op to conveniently use it throughout an SPN architecture.

    Args:
        op (SumOpBase): Implementation of sum op with corresponding backward pass definitions
    """
    global _DEFAULT_SUM_OP
    _DEFAULT_SUM_OP = op
