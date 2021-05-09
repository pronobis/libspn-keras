Models
======

This submodule provides some out-of-the box model analogues of :class:`~tensorflow.keras.Model`. They can be used to train
SPNs for e.g. generative scenarios, where there is no label for an input. There's also a :class:`~libspn_keras.models.DynamicSumProductNetwork`
that can be used for

Feedforward models
------------------
.. autoclass:: libspn_keras.models.SumProductNetwork
.. autoclass:: libspn_keras.models.SequentialSumProductNetwork
    :members: zero_evidence_inference

Temporal models
---------------
.. autoclass:: libspn_keras.models.DynamicSumProductNetwork
