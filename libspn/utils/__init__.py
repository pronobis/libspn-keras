# Import public interface of the library

from .utils import decode_bytes_array
from .lrucache import lru_cache
from .math import gather_cols_3d
from .math import scatter_cols
from .math import scatter_values
from .math import scatter_values_nd
from .math import cwise_add
from .math import logmatmul
from .math import logconv_1x1
from .math import multinomial_sample
from .math import argmax_breaking_ties
from .partition import StirlingNumber
from .partition import StirlingRatio
from .partition import Stirling
from .partition import random_partition
from .partition import all_partitions
from .partition import random_partitions_by_sampling
from .partition import random_partitions_by_enumeration
from .partition import random_partitions
from .doc import docinherit
from .serialization import register_serializable
from .serialization import json_dumps, json_loads
from .serialization import json_dump, json_load
from .serialization import str2type, type2str
from .utils import maybe_first
from .enum import Enum
from .spngraphkeys import SPNGraphKeys

# All
# TODO newlines here to make git merging easier
__all__ = ['decode_bytes_array', 'scatter_cols', 'scatter_values',
           'logmatmul', 'logconv_1x1',
           'multinomial_sample', 'argmax_breaking_ties',
           'gather_cols_3d', 'scatter_values_nd',
           'StirlingNumber', 'StirlingRatio', 'Stirling', 'random_partition',
           'all_partitions', 'random_partitions_by_sampling',
           'random_partitions_by_enumeration',
           'random_partitions',
           'docinherit',
           'register_serializable',
           'json_dumps', 'json_loads', 'json_dump', 'json_load',
           'str2type', 'type2str',
           'maybe_first',
           'Enum',
           'SPNGraphKeys']
