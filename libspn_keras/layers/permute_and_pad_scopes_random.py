import tensorflow as tf
import numpy as np
from tensorflow.keras import initializers
import logging

from libspn_keras.layers import PermuteAndPadScopes

logger = logging.getLogger('libspn-keras')


class PermuteAndPadScopesRandom(PermuteAndPadScopes):
    """
    Permutes scopes, usually applied after a ``FlatToRegions`` and a ``BaseLeaf`` layer.

    Args:
        factors (list of ints): Number of factors in preceding product layers. Needed to compute
            the effective number of scopes, including padded nodes. Can be applied at later stage
            through ``generate_factors``.
        **kwargs: kwargs to pass on to the ``keras.Layer`` superclass.
    """
    def __init__(self, factors=None, **kwargs):
        super(PermuteAndPadScopesRandom, self).__init__(None, **kwargs)
        self.factors = factors

    def set_factors(self, factors):
        self.factors = factors

    def build(self, input_shape):
        _, num_scopes, num_decomps, num_nodes_in = input_shape

        if self.factors is None or self.factors == []:
            raise ValueError("Factors needs to be a non-empty sequence.")
        factor_cumprod = np.cumprod(self.factors)
        factor_prod = factor_cumprod[-1]
        if factor_prod < num_scopes:
            raise ValueError("{}: not enough factors to cover all variables ({} vs. {})."
                             .format(self, factor_prod, num_scopes))
        for i, fc in enumerate(factor_cumprod[:-1]):
            if fc >= num_scopes:
                raise ValueError(
                    "{}: too many factors, taking out the bottom {} products still "
                    "results in {} factors while {} are needed.".format(
                        self, len(self.factors) - i - 1, fc, num_scopes))

        # Now we generate the random index permutations
        perms = [np.random.permutation(num_scopes).astype(int).tolist()
                 for _ in range(num_decomps)]

        num_m1 = factor_prod - num_scopes
        if num_m1 > 0:
            # e.g. num_m1 == 2 and factor_prod = 32. Then rate_m1 is 16, so once every 16 values
            # we should leave a variable slot empty
            rate_m1 = int(np.floor(factor_prod / num_m1))

            for p in perms:
                for i in range(num_m1):
                    p.insert(i * rate_m1, -1)
        perms = np.asarray(perms, dtype=np.int)
        self.permutations = self.add_weight(
            initializer=initializers.Constant(perms), trainable=False,
            shape=perms.shape, dtype=tf.int32
        )
        return perms

    def get_config(self):
        config = dict(
            num_vars_input=self.num_vars_spn_input,
            factors=self.factors
        )
        base_config = super(PermuteAndPadScopesRandom, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
