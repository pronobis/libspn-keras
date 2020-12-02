Constraints
============

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
