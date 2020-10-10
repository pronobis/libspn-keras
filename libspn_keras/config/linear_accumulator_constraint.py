from typing import Optional

from tensorflow.keras.constraints import Constraint

from libspn_keras.constraints.greater_equal_epsilon_normalized import (
    GreaterEqualEpsilonNormalized,
)


_DEFAULT_LINEAR_ACCUMULATOR_CONSTRAINT: Optional[
    Constraint
] = GreaterEqualEpsilonNormalized()


def get_default_linear_accumulators_constraint() -> Optional[Constraint]:
    """
    Get default linear accumulator constraint.

    Returns:
        A ``Constraint`` instance that was set with ``set_default_linear_accumulators_constraint``
    """
    return _DEFAULT_LINEAR_ACCUMULATOR_CONSTRAINT


def set_default_linear_accumulators_constraint(op: Constraint) -> None:
    """
    Set default sum op to conveniently use it throughout an SPN architecture.

    Args:
        op: A constraint applied to linear accumulators after updates
    """
    global _DEFAULT_LINEAR_ACCUMULATOR_CONSTRAINT
    _DEFAULT_LINEAR_ACCUMULATOR_CONSTRAINT = op
