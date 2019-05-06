from enum import Enum


class LearningTaskType(Enum):
    """Learning task to complete (supervised or unsupervised)."""
    UNSUPERVISED = 0
    """Use Unsupervised Learning."""

    SUPERVISED = 1
    """Use Supervised Learning."""


class LearningMethodType(Enum):
    """Learning methods (generative or discriminative)."""
    DISCRIMINATIVE = 0
    """Use Discriminative Learning Objective (e.g. cross-entropy for ``GDLearning``)."""

    GENERATIVE = 1
    """Use Generative Learning Objective (e.g. NLL for ``GDLearning``)."""
