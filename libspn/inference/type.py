# ------------------------------------------------------------------------
# Copyright (C) 2016 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

from enum import Enum


class InferenceType(Enum):
    """Types of inference."""
    MARGINAL = 0
    """Use marginal inference."""

    MPE = 1
    """Use MPE inference."""
