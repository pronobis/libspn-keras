import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import initializers


class RandomDecompositions(keras.layers.Layer):

    def __init__(self, num_decomps, permutations=None, **kwargs):
        """
        Creates random decompositions of input variables by generating random permutations. This
        results in adjacent scopes that can easily be 'joined' by computing outer products later on
        in the SPN.

        Args:
            num_decomps: Number of decompositions
            permutations: If not None, will override random permutations
            **kwargs: kwargs to pass on to the keras.Layer superclass.
        """
        super(RandomDecompositions, self).__init__(**kwargs)
        self.num_decomps = num_decomps
        self.permutations = None
        self._num_nodes = self._num_scopes = self.permutations = None
        if permutations is not None:
            self.permutations = self.add_weight(
                initializers.Constant(permutations), trainable=False)

    def generate_permutations(self, factors, num_vars_spn_input):
        if not factors:
            raise ValueError("{}: factors needs to be a non-empty sequence.")
        factor_cumprod = np.cumprod(factors)
        factor_prod = factor_cumprod[-1]
        if factor_prod < num_vars_spn_input:
            raise ValueError("{}: not enough factors to cover all variables ({} vs. {})."
                             .format(self, factor_prod, num_vars_spn_input))
        for i, fc in enumerate(factor_cumprod[:-1]):
            if fc >= num_vars_spn_input:
                raise ValueError(
                    "{}: too many factors, taking out the bottom {} products still "
                    "results in {} factors while {} are needed.".format(
                        self, len(factors) - i - 1, fc, num_vars_spn_input))

        # Now we generate the random index permutations
        perms = [np.random.permutation(num_vars_spn_input).astype(int).tolist()
                 for _ in range(self.num_decomps)]

        num_m1 = factor_prod - num_vars_spn_input
        if num_m1 > 0:
            # e.g. num_m1 == 2 and factor_prod = 32. Then rate_m1 is 16, so once every 16 values
            # we should leave a variable slot empty
            rate_m1 = int(np.floor(factor_prod / num_m1))

            for p in perms:
                for i in range(num_m1):
                    p.insert(i * rate_m1, -1)
        self.permutations = self.add_weight(
            initializer=initializers.Constant(perms), trainable=False)
        return perms

    def call(self, x):

        if self.permutations is None:
            raise ValueError("First need to determine permutations")
        zero_padded = tf.pad(x, [[0, 0], [1, 0]])
        gather_indices = self.permutations + 1
        return tf.expand_dims(tf.transpose(
            tf.gather(zero_padded, gather_indices, axis=1),
            (2, 1, 0)
        ), axis=-1)

    def compute_output_shape(self, input_shape):

        num_batch, num_in_scopes = input_shape
        if self._num_scopes is None or self.num_decomps is None:
            raise ValueError("Number of scopes or decomps is yet unknown")

        return num_in_scopes, self.num_decomps, num_batch, self._num_sums

    def get_config(self):
        config = dict(
            num_decomps=self.num_decomps
        )
        base_config = super(RandomDecompositions, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
