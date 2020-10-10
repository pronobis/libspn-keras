from tensorflow.keras.constraints import Constraint

from libspn_keras.constraints.log_normalized import LogNormalized

_DEFAULT_LOGSPACE_ACCUMULATOR_CONSTRAINT = None


def get_default_logspace_accumulators_constraint() -> Constraint:
    """
    Get default logspace accumulator constraint.

    Returns:
        A ``Constraint`` instance that was set with ``set_default_logspace_accumulators_constraint``
    """
    return (
        LogNormalized()
        if _DEFAULT_LOGSPACE_ACCUMULATOR_CONSTRAINT is None
        else _DEFAULT_LOGSPACE_ACCUMULATOR_CONSTRAINT
    )


def set_default_logspace_accumulators_constraint(op: Constraint) -> None:
    """
    Set default sum op to conveniently use it throughout an SPN architecture.

    Args:
        op: A constraint applied to logspace accumulators after updates
    """
    global _DEFAULT_LOGSPACE_ACCUMULATOR_CONSTRAINT
    _DEFAULT_LOGSPACE_ACCUMULATOR_CONSTRAINT = op
