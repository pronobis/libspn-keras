Sum Operations
==============

LibSPN-Keras offers a convenient way to control the backward pass for sum operations used in the SPNs you build.
Internally, LibSPN-Keras defines a couple of sum operations with different backward passes, for gradient as well as
EM learning. All of these operations inherit from the ``SumOpBase`` class in ``libspn_keras/sum_ops.py``.

**NOTE**

By default, LibSPN-Keras uses ``SumOpGradBackprop``.

Getting And Setting a Sum Op
----------------------------
These methods allow for setting and getting the current default ``SumOpBase``. By setting a default all sum layers
(``DenseSum``, ``Conv2DSum``, ``Local2DSum`` and ``RootSum``) will use that sum op, unless you explicitly provide a
``SumOpBase`` instance to any of those classes when initializing them.

.. autofunction:: libspn_keras.config.sum_op.set_default_sum_op
.. autofunction:: libspn_keras.config.sum_op.get_default_sum_op

Sum Operation With Gradients In Backward Pass
---------------------------------------------
.. autoclass:: libspn_keras.sum_ops.SumOpGradBackprop


Sum Operations With EM Signals In Backward Pass
-----------------------------------------------
.. autoclass:: libspn_keras.sum_ops.SumOpEMBackprop
.. autoclass:: libspn_keras.sum_ops.SumOpHardEMBackprop
.. autoclass:: libspn_keras.sum_ops.SumOpUnweightedHardEMBackprop
