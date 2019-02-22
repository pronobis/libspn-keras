Utilities
=========

Various utility functions and classes.


Math
----

Various LibSPN math functions.

.. autofunction:: libspn.utils.gather_cols_3d
.. autofunction:: libspn.utils.scatter_cols
.. autofunction:: libspn.utils.scatter_values
.. autofunction:: libspn.utils.cwise_add

Set Partitions
--------------

Functions for generating a random set of `partitions of a set
<https://en.wikipedia.org/wiki/Partition_of_a_set>`_.

.. autoclass:: libspn.utils.StirlingNumber
.. autoclass:: libspn.utils.StirlingRatio
.. autoclass:: libspn.utils.Stirling
.. autofunction:: libspn.utils.random_partition
.. autofunction:: libspn.utils.all_partitions
.. autofunction:: libspn.utils.random_partitions_by_sampling
.. autofunction:: libspn.utils.random_partitions_by_enumeration
.. autofunction:: libspn.utils.random_partitions

Serialization
-------------

LibSPN serialization tools.

.. autofunction:: libspn.utils.register_serializable
.. autofunction:: libspn.utils.json_dump
.. autofunction:: libspn.utils.json_load
.. autofunction:: libspn.utils.str2type
.. autofunction:: libspn.utils.type2str


Documentation
-------------

.. autofunction:: libspn.utils.docinherit
