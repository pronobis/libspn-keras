from tensorflow.keras.constraints import Constraint

from libspn_keras.constraints.greater_equal_epsilon_normalized import (
    GreaterEqualEpsilonNormalized,
)

_DEFAULT_LINEAR_ACCUMULATOR_CONSTRAINT = None


def get_default_linear_accumulators_constraint() -> Constraint:
    """
    Get default linear accumulator constraint.

    Returns:
        A ``Constraint`` instance that was set with ``set_default_linear_accumulators_constraint``
    """
    return (
        GreaterEqualEpsilonNormalized()
        if _DEFAULT_LINEAR_ACCUMULATOR_CONSTRAINT is None
        else _DEFAULT_LINEAR_ACCUMULATOR_CONSTRAINT
    )


def set_default_linear_accumulators_constraint(op: Constraint) -> None:
    """
    Set default sum op to conveniently use it throughout an SPN architecture.

    Args:
        op: A constraint applied to linear accumulators after updates
    """
    global _DEFAULT_LINEAR_ACCUMULATOR_CONSTRAINT
    _DEFAULT_LINEAR_ACCUMULATOR_CONSTRAINT = op
