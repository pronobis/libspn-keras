import logging
from typing import List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import initializers

from libspn_keras.layers.permute_and_pad_scopes import PermuteAndPadScopes


logger = logging.getLogger("libspn-keras")


class PermuteAndPadScopesRandom(PermuteAndPadScopes):
    """
    Permutes scopes, usually applied after a ``FlatToRegions`` and a ``BaseLeaf`` layer.

    Args:
        factors: Number of factors in preceding product layers. Needed to compute
            the effective number of scopes, including padded nodes. Can be applied at later stage
            through ``generate_factors``.
        **kwargs: kwargs to pass on to the ``keras.Layer`` superclass.
    """

    def __init__(self, factors: Optional[List[int]] = None, **kwargs):
        super(PermuteAndPadScopesRandom, self).__init__(None, **kwargs)
        self.factors = factors

    def set_factors(self, factors: List[int]) -> None:
        """
        Set the factors.

        These factors determine the amount of padding which is required to for uniform dimension sizes
        across scope, decomp and node axes.

        Args:
            factors: The factors of products in the SPN graph per layer (DenseProduct or ReduceProduct).
        """
        self.factors = factors

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:  # noqa: C901
        """
        Build the internal components for this layer.

        Args:
            input_shape: Shape of the input Tensor.

        Raises:
            ValueError: If shape cannot be determined
        """
        _, num_scopes, num_decomps, num_nodes_in = input_shape
        if num_decomps is None:
            raise ValueError(
                "Cannot build PermuteAndPadScopesRandom with an unknown decomposition dimension"
            )
        if num_scopes is None:
            raise ValueError(
                "Cannot build PermuteAndPadScopesRandom with an unknown scopes dimension"
            )
        if num_nodes_in is None:
            raise ValueError(
                "Cannot build PermuteAndPadScopesRandom with an unknown number of nodes per "
                "scope/decomposition"
            )

        if self.factors is None or self.factors == []:
            raise ValueError("Factors needs to be a non-empty sequence.")
        factor_cumprod = np.cumprod(self.factors)
        factor_prod = factor_cumprod[-1]
        if factor_prod < num_scopes:
            raise ValueError(
                "{}: not enough factors to cover all variables ({} vs. {}).".format(
                    self, factor_prod, num_scopes
                )
            )
        for i, fc in enumerate(factor_cumprod[:-1]):
            if fc >= num_scopes:
                raise ValueError(
                    "{}: too many factors, taking out the bottom {} products still "
                    "results in {} factors while {} are needed.".format(
                        self, len(self.factors) - i - 1, fc, num_scopes
                    )
                )

        # Now we generate the random index permutations
        perms: List[List[int]] = [
            list(np.random.permutation(num_scopes).astype(int))
            for _ in range(num_decomps)
        ]

        num_m1 = factor_prod - num_scopes
        if num_m1 > 0:
            # e.g. num_m1 == 2 and factor_prod = 32. Then rate_m1 is 16, so once every 16 values
            # we should leave a variable slot empty
            rate_m1 = int(np.floor(factor_prod / num_m1))

            for p in perms:
                for i in range(num_m1):
                    p.insert(i * rate_m1, -1)
        self.permutations = self.add_weight(
            name="permutations",
            initializer=initializers.Constant(perms),
            trainable=False,
            shape=[num_decomps, factor_prod],
            dtype=tf.int32,
        )

    def get_config(self) -> dict:
        """
        Obtain a key-value representation of the layer config.

        Returns:
            A dict holding the configuration of the layer.
        """
        config = dict(num_vars_input=self.num_vars_spn_input, factors=self.factors)
        base_config = super(PermuteAndPadScopesRandom, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
