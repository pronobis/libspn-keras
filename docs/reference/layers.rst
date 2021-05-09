Layers
======

This submodule contains Keras layers for building SPNs. Besides leaf layers and regularization
layers, there are two main groups of layers:

- Region layers for arbitrary decompositions of variables. Must be preceded with a
  :class:`~libspn_keras.layers.FlatToRegions` - :class:`~libspn_keras.layers.BaseLeaf` - :class:`~libspn_keras.layers.PermuteAndPadScopes` block. Regions are arbitrary sets of variables. A
  region graph describes how these sets of variables hierarchically define a probability
  distribution.
- Spatial layers for `Deep Generalized Convolutional Sum Product Networks <https://arxiv.org/abs/1902.06155>`_

All layers propagate **log probabilities** in the forward pass. So in case you want to know about the
'raw' probability in linear space, you simply pass the output of a layer through :math:`\exp`.

Leaf layers
-----------
Leaf layers transform raw observations to probabilities.

- :class:`~libspn_keras.layers.NormalLeaf`, :class:`~libspn_keras.layers.CauchyLeaf` and :class:`~libspn_keras.layers.LaplaceLeaf` can be used for continuous inputs.
- :class:`~libspn_keras.layers.IndicatorLeaf` should be used for discrete inputs.

If a variable is not part of the
evidence, that means that variable should be marginalized out. This can be done by replacing
the output of the corresponding components with 0 since that corresponds with 1 in log-space.

Continuous leaf layers
^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: libspn_keras.layers.NormalLeaf
.. autoclass:: libspn_keras.layers.CauchyLeaf
.. autoclass:: libspn_keras.layers.LaplaceLeaf

Discrete leaf layers
^^^^^^^^^^^^^^^^^^^^
.. autoclass:: libspn_keras.layers.IndicatorLeaf

Region layers
-------------
Region layers assume the tensors that are passed between them are of the shape
``[num_scopes, num_decomps, num_batch, num_nodes]``. One region is given by the scope index + the
decomposition (so it is indexed on the first two axes). This shape is chosen so that ``matmul``
operations done in ``DenseSum`` layers don't always require transposing first.

.. autoclass:: libspn_keras.layers.FlatToRegions

Permutation layers
^^^^^^^^^^^^^^^^^^
These layers permute variables so that this only has to be done once at the bottom of the network
.. autoclass:: libspn_keras.layers.PermuteAndPadScopes
.. autoclass:: libspn_keras.layers.PermuteAndPadScopesRandom

Sum Layers
^^^^^^^^^^
.. autoclass:: libspn_keras.layers.DenseSum
.. autoclass:: libspn_keras.layers.RootSum

Product Layers
^^^^^^^^^^^^^^
.. autoclass:: libspn_keras.layers.DenseProduct
.. autoclass:: libspn_keras.layers.ReduceProduct

Spatial layers
--------------
Spatial layers are layers needed for building `DGC-SPNs <https://arxiv.org/abs/1902.06155>`_. The
final layer of an SPN should still be a ``RootSum``. Use ``SpatialToRegions`` to convert the output
from a spatial SPN to a region SPN.

.. autoclass:: libspn_keras.layers.Local2DSum
.. autoclass:: libspn_keras.layers.Conv2DSum
.. autoclass:: libspn_keras.layers.Conv2DProduct
.. autoclass:: libspn_keras.layers.SpatialToRegions

Dynamic SPN layers
------------------
For reusing SPN structures along the temporal dimension one can implement dynamic SPNs. These rely
on *template SPNs*, *top SPNs* and an *interface*. The interface of the previous timestep and the
template at the current timestep can be combined through ``TemporalDenseProduct``.

.. autoclass:: libspn_keras.layers.TemporalDenseProduct

Regularization layers
---------------------
.. autoclass:: libspn_keras.layers.LogDropout

Normalization
-------------

Normalize axes
^^^^^^^^^^^^^^
.. autoclass:: libspn_keras.layers.NormalizeAxes

    .. autoattribute:: SAMPLE_WISE
    .. autoattribute:: VARIABLE_WISE
    .. autoattribute:: GLOBAL

Normalize layers
^^^^^^^^^^^^^^^^
.. autoclass:: libspn_keras.layers.NormalizeStandardScore
    :members: adapt
