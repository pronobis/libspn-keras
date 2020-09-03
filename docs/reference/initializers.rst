Initializers
============

In addition to initializers in ``tensorflow.keras.initializers``, ``libspn-keras`` implements a few
more useful initialization schemes for both leaf layers as well as sum weights.

Location initializers
---------------------
For a leaf distribution of the location scale family, the following initializers are useful

.. autoclass:: libspn_keras.initializers.PoonDomingosMeanOfQuantileSplit
.. autoclass:: libspn_keras.initializers.KMeans
.. autoclass:: libspn_keras.initializers.Equidistant

Weight initializers
-------------------
When training with either ``HARD_EM`` or ``HARD_EM_UNWEIGHTED``, you can use the
``EpsilonInverseFanIn`` initializer.

.. autoclass:: libspn_keras.initializers.EpsilonInverseFanIn
