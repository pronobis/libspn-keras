Sum Operations
==============

LibSPN-Keras offers a convenient way to control the backward pass for sum operations used in the SPNs you build.
Internally, LibSPN-Keras defines a couple of sum operations with different backward passes, for gradient as well as
EM learning. All of these operations inherit from the :class:`~libspn_keras.sum_ops.SumOpBase`.

**NOTE**

By default, LibSPN-Keras uses :class:`~libspn_keras.SumOpGradBackprop`.

Getting And Setting a Sum Op
----------------------------
These methods allow for setting and getting the current default :class:`~libspn_keras.sum_ops.SumOpBase`. By setting a default all sum layers
(:class:`~libspn_keras.layers.DenseSum`, :class:`~libspn_keras.layers.Conv2DSum`, :class:`~libspn_keras.layers.Local2DSum` and :class:`~libspn_keras.layers.RootSum`)
will use that sum op, unless you explicitly provide a :class:`~libspn_keras.sum_ops.SumOpBase` instance to any of those classes when initializing them.

.. autofunction:: libspn_keras.set_default_sum_op
.. autofunction:: libspn_keras.get_default_sum_op

Sum Operation With Gradients In Backward Pass
---------------------------------------------
.. autoclass:: libspn_keras.SumOpGradBackprop


Sum Operations With EM Signals In Backward Pass
-----------------------------------------------
.. autoclass:: libspn_keras.SumOpEMBackprop
.. autoclass:: libspn_keras.SumOpHardEMBackprop
.. autoclass:: libspn_keras.SumOpUnweightedHardEMBackprop


Sum Operations With Sample Signals In Backward Pass
---------------------------------------------------
.. autoclass:: libspn_keras.SumOpSampleBackprop
