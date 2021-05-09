Initializers
============

In addition to initializers in ``tensorflow.keras.initializers``, ``libspn-keras`` implements a few
more useful initialization schemes for both leaf layers as well as sum weights.

Setting Defaults
----------------
Since accumulator initializers are often the same for all layers in an SPN, ``libspn-keras`` provides the following
functions to get and set default accumulator initializers. These can still be overridden by providing the initializers
explicitly at initialization of a layer.

.. autofunction:: libspn_keras.set_default_accumulator_initializer
.. autofunction:: libspn_keras.get_default_accumulator_initializer

Location initializers
---------------------
For a leaf distribution of the location scale family, the following initializers can be used for initializing the
location parameters

.. autoclass:: libspn_keras.initializers.PoonDomingosMeanOfQuantileSplit
.. autoclass:: libspn_keras.initializers.KMeans
.. autoclass:: libspn_keras.initializers.Equidistant

Scale initializers
---------------------
For a leaf distribution of the location scale family, the following initializers can be used for initializing the
scale parameters

.. autoclass:: libspn_keras.initializers.PoonDomingosStddevOfQuantileSplit

Weight initializers
-------------------

.. autoclass:: libspn_keras.initializers.Dirichlet

**Note**
Initializer for discrete EM (:class:`~libspn_keras.SumOpHardEMBackprop` and
:class:`~libspn_keras.SumOpUnweightedHardEMBackprop`).

.. autoclass:: libspn_keras.initializers.EpsilonInverseFanIn
