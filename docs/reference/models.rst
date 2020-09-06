Models
======

This submodule provides some out-of-the box model analogues of ``tensorflow.keras.Model``. They can be used to train
SPNs for e.g. generative scenarios, where there is no label for an input. There's also a ``DynamicSumProductNetwork``
that can be used for

Feedforward models
------------------
.. autoclass:: libspn_keras.models.SumProductNetwork
.. autoclass:: libspn_keras.models.SequentialSumProductNetwork

Temporal models
---------------
.. autoclass:: libspn_keras.models.DynamicSumProductNetwork
