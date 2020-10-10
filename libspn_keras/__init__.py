try:
    from importlib.metadata import version, PackageNotFoundError  # type: ignore
except ImportError:  # pragma: no cover
    from importlib_metadata import version, PackageNotFoundError  # type: ignore

try:
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

from libspn_keras import config
from libspn_keras import constraints
from libspn_keras import initializers
from libspn_keras import layers
from libspn_keras import losses
from libspn_keras import metrics
from libspn_keras import models
from libspn_keras import optimizers
from libspn_keras.config.accumulator_initializer import (
    get_default_accumulator_initializer,
    set_default_accumulator_initializer,
)
from libspn_keras.config.linear_accumulator_constraint import (
    get_default_linear_accumulators_constraint,
    set_default_linear_accumulators_constraint,
)
from libspn_keras.config.logspace_accumulator_constraint import (
    get_default_logspace_accumulators_constraint,
    set_default_logspace_accumulators_constraint,
)
from libspn_keras.config.sum_op import get_default_sum_op, set_default_sum_op
from libspn_keras.logspace import logspace_wrapper_initializer
from libspn_keras.region import region_graph_to_dense_spn
from libspn_keras.region import RegionNode
from libspn_keras.region import RegionVariable
from libspn_keras.sum_ops import (
    SumOpEMBackprop,
    SumOpGradBackprop,
    SumOpHardEMBackprop,
    SumOpUnweightedHardEMBackprop,
)
from libspn_keras.visualize import visualize_dense_spn


__all__ = [
    "config",
    "get_default_accumulator_initializer",
    "set_default_accumulator_initializer",
    "set_default_logspace_accumulators_constraint",
    "set_default_linear_accumulators_constraint",
    "get_default_linear_accumulators_constraint",
    "get_default_logspace_accumulators_constraint",
    "get_default_sum_op",
    "set_default_sum_op",
    "logspace_wrapper_initializer",
    "optimizers",
    "metrics",
    "losses",
    "layers",
    "constraints",
    "RegionNode",
    "RegionVariable",
    "region_graph_to_dense_spn",
    "visualize_dense_spn",
    "initializers",
    "models",
    "SumOpUnweightedHardEMBackprop",
    "SumOpEMBackprop",
    "SumOpGradBackprop",
    "SumOpHardEMBackprop",
]
