Constraints
============

Setting Defaults
----------------
Since constraints are often the same for all layers in an SPN, ``libspn-keras`` provides the following
functions to get and set default constraints. These can still be overridden by providing the initializers
explicitly at initialization of a layer.

Linear accumulator constraints
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, the logspace accumulator constraint is set to :class:`~libspn_keras.constraints.GreaterEqualEpsilonNormalized`.

.. autofunction:: libspn_keras.set_default_linear_accumulators_constraint
.. autofunction:: libspn_keras.get_default_linear_accumulators_constraint

Logspace accumulator constraints
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, the logspace accumulator constraint is set to :class:`~libspn_keras.constraints.LogNormalized`.

.. autofunction:: libspn_keras.set_default_logspace_accumulators_constraint
.. autofunction:: libspn_keras.get_default_logspace_accumulators_constraint

Linear Weight Constraints
-------------------------
These should be used for linear accumulators.

.. autoclass:: libspn_keras.constraints.GreaterEqualEpsilonNormalized
.. autoclass:: libspn_keras.constraints.GreaterEqualEpsilon

Log Weight Constraints
----------------------
These should be used for log accumulators.

.. autoclass:: libspn_keras.constraints.LogNormalized

Scale Constraints
-----------------
The following constraint is useful for ensuring stable scale parameters in location-scale leaf layers.

.. autoclass:: libspn_keras.constraints.Clip
