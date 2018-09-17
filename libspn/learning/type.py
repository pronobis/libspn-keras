# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

from enum import Enum


class LearningTaskType(Enum):
    """Types of learning."""
    UNSUPERVISED = 0
    """Use Unsupervised Learning."""

    SUPERVISED = 1
    """Use Supervised Learning."""

class LearningMethodType(Enum):
    """Learning methods."""
    DISCRIMINATIVE = 0
    """Use Discriminative Learning."""

    GENERATIVE = 1
    """Use Generative Learning."""

class GradientType(Enum):
    """Types of gradient."""
    SOFT = 0
    """Compute Soft gradient."""

    HARD = 1
    """Compute Hard gradient."""
