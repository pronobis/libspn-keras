Region graphs
=============

The region graph utilities can be made to create SPNs by explicitly defining the region structure.
This can be used to e.g. conveniently express some learned structure.

For some examples on how to use region graphs check out
`this tutorial <https://colab.research.google.com/drive/1QMEFEjb7jZdOtuo5OT5J2HVhNOE_3xmc>`_.

.. autoclass:: libspn_keras.RegionVariable
.. autoclass:: libspn_keras.RegionNode
.. autofunction:: libspn_keras.region_graph_to_dense_spn
