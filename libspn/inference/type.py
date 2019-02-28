from enum import Enum


class InferenceType(Enum):
    """Types of inference."""
    MARGINAL = 0
    """Use marginal inference."""

    MPE = 1
    """Use MPE inference."""
