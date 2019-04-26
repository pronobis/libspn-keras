import tensorflow as tf
from tensorflow.python.ops.init_ops import Initializer
from libspn.utils.serialization import register_serializable
from libspn import conf


def _tf_init_serialize(self):
    return self.get_config()


def _tf_init_deserialize(self, data):
    self.__init__(**{k: v for k, v in data.items() if k != "__type__"})


initializer_names = [
    'random_uniform', 'random_normal', 'constant', 'ones', 'glorot_normal', 'glorot_uniform',
    'identity', 'orthogonal', 'truncated_normal', 'uniform_unit_scaling', 'variance_scaling',
    'zeros'
]

for name in initializer_names:
    if not hasattr(tf.initializers, name):
        continue
    initializer = getattr(tf.initializers, name)
    initializer.deserialize = _tf_init_deserialize
    initializer.serialize = _tf_init_serialize
    register_serializable(initializer)


class Equidistant(Initializer):
    """Initializer that generates tensors where the last axis is initialized with 'equidistant'
    values.

    Args:
      minval: A python scalar or a scalar tensor. Lower bound of the range
        of random values to generate.
      maxval: A python scalar or a scalar tensor. Upper bound of the range
        of random values to generate.  Defaults to 1 for float types.
      seed: A Python integer. Used to create random seeds. See
        @{tf.set_random_seed}
        for behavior.
      dtype: The data type.
    """

    def __init__(self, minval=0.0, maxval=1.0, dtype=conf.dtype):
        self.minval = minval
        self.maxval = maxval
        self.dtype = dtype

    def __call__(self, shape, dtype=None, partition_info=None):
        if dtype is None:
            dtype = self.dtype

        rank = len(shape)
        last_dim = shape[-1]
        linspace = tf.reshape(
            tf.linspace(self.minval, self.maxval, num=last_dim), [1] * (rank - 1) + [last_dim])
        return tf.cast(tf.tile(linspace, tf.concat([shape[:-1], [1]], axis=0)), dtype=dtype)

    def get_config(self):
        return {
            "minval": self.minval,
            "maxval": self.maxval,
            "dtype": self.dtype.name
        }