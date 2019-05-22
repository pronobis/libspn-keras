Graph
=====

The SPN graph consists of multiple connected nodes. These nodes can be
categorized as:

* Variable nodes - represent random variables or inputs to the SPN graph (e.g.
  indicator variables)
* Operation nodes - represent operations performing computations in the graph
  (e.g. sums or products)
* Parameter nodes - represent parameters of the graph that can be learned (e.g.
  sum weights)

Each node has a single output, but the output can produces multiple values (a
tensor). Operation nodes can have one or multiple inputs and each input can be
connected to a single output of another node. Therefore, each input also accepts
a tensor and nodes pass tensors between each other. The number of inputs does
not have to be fixed, and nodes can accept a varying number of inputs (the
number of inputs is then determined by the final structure of the graph). In
other words, an operation node can be seen as a function with the following
signature ``output_tensor = op_node(input_tensor1, input_tensor2, ...,
*input_tensors)``. The number and interpretation of inputs depends on the
operation node. An input can be disconnected if the graph is not yet complete or
the input is optional.

Since in SPN, shuffling of inputs happens often, when connecting an output to an
input of an operation node, it is possible to indicate a subset and order of
tensor elements that should be passed to that input. This is done by specifying
a list of indices of elements of the tensor when connecting a node to an input.
See the :class:`~libspn.Input` and :meth:`~libspn.Input.as_input` for the ways a
connection to an input is specified.

Nodes process data by batches, and the first dimension of a tensor passed
between nodes corresponds to batch samples. It is the second dimension that
corresponds to sample values and it is that dimension that is indexed by the
indices associated with inputs.

Nodes are attached to inputs of an operation node using specific methods defined
in the operation node that correspond to the semantics of the inputs. For
instance, to attach a ``Weights`` node to a ``Sum`` node, use the
``set_weights`` method of the ``Sum`` node. These methods accept values which
:meth:`~libspn.Input.as_input` accepts.


Node Interface
--------------

The interface of a node is specified using abstract classes. Custom nodes should
inherit from these classes.

.. autoclass:: libspn.Scope
.. autoclass:: libspn.Input
.. autoclass:: libspn.Node
.. autoclass:: libspn.VarNode
.. autoclass:: libspn.OpNode
.. autoclass:: libspn.ParamNode


Variable Nodes
--------------

.. autoclass:: libspn.RawLeaf
.. autoclass:: libspn.IndicatorLeaf
.. autoclass:: libspn.NormalLeaf
.. autoclass:: libspn.CauchyLeaf
.. autoclass:: libspn.LaplaceLeaf
.. autoclass:: libspn.TruncatedNormalLeaf
.. autoclass:: libspn.StudentTLeaf
.. autoclass:: libspn.MultivariateNormalDiagLeaf
.. autoclass:: libspn.MultivariateCauchyDiagLeaf


Operation Nodes
---------------

.. autoclass:: libspn.Sum
.. autoclass:: libspn.ParallelSums
.. autoclass:: libspn.SumsLayer
.. autoclass:: libspn.Product
.. autoclass:: libspn.PermuteProducts
.. autoclass:: libspn.ProductsLayer
.. autoclass:: libspn.Concat
.. autoclass:: libspn.BlockSum
.. autoclass:: libspn.BlockRootSum
.. autoclass:: libspn.BlockReduceProduct
.. autoclass:: libspn.BlockPermuteProduct
.. autoclass:: libspn.BlockMergeDecompositions
.. autoclass:: libspn.BlockRandomDecompositions


Convolutional Nodes
-------------------

.. autoclass:: libspn.ConvSums
.. autoclass:: libspn.LocalSums
.. autoclass:: libspn.ConvProducts
.. autoclass:: libspn.ConvProductsDepthwise


Parameter Nodes
---------------

.. autoclass:: libspn.Weights

Helper functions related to parameter nodes operating on the graph:

.. autofunction:: libspn.assign_weights
.. autofunction:: libspn.initialize_weights


Saving and Loading
------------------

Saver/loader classes are used to save/load SPN graph to disk and implement an
interface specified by :class:`~libspn.Saver` and :class:`~libspn.Loader`.

.. autoclass:: libspn.Saver
.. autoclass:: libspn.Loader

Currently, JSON saver and loader are implemented:

.. autoclass:: libspn.JSONSaver
.. autoclass:: libspn.JSONLoader


Algorithms
----------

Several graph traversal algorithms are provided. These are mostly for internal
use, but can be run on the graph by external code if necessary.

.. autofunction:: libspn.compute_graph_up
.. autofunction:: libspn.compute_graph_up_down
.. autofunction:: libspn.traverse_graph
