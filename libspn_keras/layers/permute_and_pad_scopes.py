import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import initializers


class PermuteAndPadScopes(keras.layers.Layer):

    def __init__(self, num_decomps, permutations=None, **kwargs):
        """
        Permutes scopes, usually applied after a ReshapeFlatToScopeDecompFirst and a BaseLeaf layer.

        Args:
            permutations: If not None, will override random permutations
            **kwargs: kwargs to pass on to the keras.Layer superclass.
        """
        super(PermuteAndPadScopes, self).__init__(**kwargs)
        self.num_decomps = num_decomps
        self._num_nodes = self._num_scopes = self.permutations = None
        if permutations is not None:
            self.permutations = self.add_weight(
                initializer=initializers.Constant(permutations), trainable=False,
                shape=permutations.shape, dtype=tf.int32)

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
        perms = np.asarray(perms, dtype=np.int)
        self.permutations = self.add_weight(
            initializer=initializers.Constant(perms), trainable=False,
            shape=perms.shape, dtype=tf.int32
        )
        return perms

    def call(self, x):

        decomps_first = tf.transpose(x, (1, 0, 2, 3))
        decomps_first_padded = tf.pad(decomps_first, [[0, 0], [1, 0], [0, 0], [0, 0]])
        gather_indices = self.permutations + 1

        if self.permutations is None:
            raise ValueError("First need to determine permutations")
        permuted = tf.gather(decomps_first_padded, gather_indices, axis=1, batch_dims=1)
        return tf.transpose(permuted, (1, 0, 2, 3))

    def compute_output_shape(self, input_shape):
        return input_shape
