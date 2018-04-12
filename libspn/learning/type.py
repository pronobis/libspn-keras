# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

from enum import Enum


class LearningType(Enum):
    """Types of learning."""
    DISCRIMINATIVE = 0
    """Use Discriminative Learning."""

    GENERATIVE = 1
    """Use Generative Learning."""


class LearningInferenceType(Enum):
    """Types of learning inference."""
    SOFT = 0
    """Use Soft inference."""

    HARD = 1
    """Use Hard inference."""
