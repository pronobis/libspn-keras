# ------------------------------------------------------------------------
# Copyright (C) 2016 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

from enum import Enum
import numpy as np
from libspn.graph.ivs import IVs
from libspn.graph.sum import Sum
from libspn.graph.product import Product
from libspn import conf


class TestSPNGenerator:
    """Generates various simple test SPNs with initial weight values set."""

    class Type(Enum):

        POON11_NAIVE_MIXTURE = 0
        """A simple naive Bayes mixture from the Poon&Domingos'11 paper."""

    def __init__(self, spn_type):
        self._spn_type = spn_type

    def generate(self):
        if self._spn_type == TestSPNGenerator.Type.POON11_NAIVE_MIXTURE:
            return self.generate_poon11_naive_mixture()

    @property
    def true_mpe_state(self):
        """The true MPE state for the SPN."""
        if self._spn_type == TestSPNGenerator.Type.POON11_NAIVE_MIXTURE:
            return np.array([1, 0])

    @property
    def true_values(self):
        """The true values of the SPN for the :meth:`feed`."""
        if self._spn_type == TestSPNGenerator.Type.POON11_NAIVE_MIXTURE:
            return np.array([[1.0],
                             [0.75],
                             [0.25],
                             [0.31],
                             [0.228],
                             [0.082],
                             [0.69],
                             [0.522],
                             [0.168]], dtype=conf.dtype.as_numpy_dtype)

    @property
    def true_mpe_values(self):
        """The true MPE values of the SPN for the :meth:`feed`."""
        if self._spn_type == TestSPNGenerator.Type.POON11_NAIVE_MIXTURE:
            return np.array([[0.216],
                             [0.216],
                             [0.09],
                             [0.14],
                             [0.14],
                             [0.06],
                             [0.216],
                             [0.216],
                             [0.09]], dtype=conf.dtype.as_numpy_dtype)

    @property
    def feed(self):
        """Feed containing all possible values of the input variables."""
        if self._spn_type == TestSPNGenerator.Type.POON11_NAIVE_MIXTURE:
            values = np.arange(-1, 2)
            points = np.array(np.meshgrid(*[values for i in range(2)])).T
            return points.reshape(-1, points.shape[-1])

    def generate_poon11_naive_mixture(self):
        """Generates a simple naive Bayes mixture model shown in Fig1a of the
        Poon&Domingos'11 paper.

        Returns:
            A tuple ``(ivs, root)`` with the IVs node providing inputs to the
            SPN and the sum root node.
        """
        # Inputs
        ivs = IVs(num_vars=2, num_vals=2, name="IVs")
        # Input mixtures
        s11 = Sum((ivs, [0, 1]), name="Sum1.1")
        s11.generate_weights([0.4, 0.6])
        s12 = Sum((ivs, [0, 1]), name="Sum1.2")
        s12.generate_weights([0.1, 0.9])
        s21 = Sum((ivs, [2, 3]), name="Sum2.1")
        s21.generate_weights([0.7, 0.3])
        s22 = Sum((ivs, [2, 3]), name="Sum2.2")
        s22.generate_weights([0.8, 0.2])
        # Components
        p1 = Product(s11, s21, name="Comp1")
        p2 = Product(s11, s22, name="Comp2")
        p3 = Product(s12, s22, name="Comp3")
        # Mixing components
        root = Sum(p1, p2, p3, name="Mixture")
        root.generate_weights([0.5, 0.2, 0.3])
        return ivs, root
