Initializers
============

In addition to initializers in ``tensorflow.keras.initializers``, ``libspn-keras`` implements a few
more useful initialization schemes for both leaf layers as well as sum weights.

Setting Defaults
----------------
Since accumulator initializers are often the same for all layers in an SPN, ``libspn-keras`` provides the following
functions to get and set default accumulator initializers. These can still be overridden by providing the initializers
explicitly at initialization of a layer.

.. autofunction:: libspn_keras.config.accumulator_initializer.set_default_accumulator_initializer
.. autofunction:: libspn_keras.config.accumulator_initializer.get_default_accumulator_initializer

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
