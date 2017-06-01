# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

from abc import ABC, abstractmethod


class Model(ABC):
    """An abstract class defining the interface of a model."""

    def __init__(self):
        self._root = None

    def root(self):
        """OpNode: Root node of the model."""
        return self._root

    @abstractmethod
    def build():
        """Build the SPN graph of the model."""
