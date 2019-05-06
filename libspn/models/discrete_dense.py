from libspn.models.model import Model
from libspn.log import get_logger
from libspn.graph.leaf.indicator import IndicatorLeaf
from libspn.generation.dense import DenseSPNGenerator
from libspn.generation.weights import generate_weights
from libspn.graph.node import Input
from libspn.graph.serialization import serialize_graph, deserialize_graph
from libspn.graph.op.sum import Sum
from libspn import utils
from libspn.utils.serialization import register_serializable
import random
import tensorflow as tf


@register_serializable
class DiscreteDenseModel(Model):

    """Basic dense SPN model operating on discrete data.

    If `num_classes` is greater than 1, a multi-class model is created by
    generating multiple parallel dense models (one for each class) and combining
    them with a sum node with an explicit latent class variable.

    Args:
        num_vars (int): Number of discrete random variables representing data
                        samples.
        num_vals (int): Number of values of each random variable.
        num_classes (int): Number of classes assumed by the model.
        num_decomps (int): Number of decompositions at each level of dense SPN.
        num_subsets (int): Number of variable sub-sets for each decomposition.
        num_mixtures (int): Number of mixtures (sums) for each variable subset.
        input_dist (InputDist): Determines how IndicatorLeaf of the discrete variables for
                                data samples are connected to the model.
        num_input_mixtures (int): Number of mixtures used representing each
                                  discrete data variable (mixing the data variable
                                  IndicatorLeaf) when ``input_dist`` is set to ``MIXTURE``.
                                  If set to ``None``, ``num_mixtures`` is used.
        weight_init_value: Initial value of the weights.
    """

    __logger = get_logger()
    __info = __logger.info
    __debug1 = __logger.debug1
    __is_debug1 = __logger.is_debug1
    __debug2 = __logger.debug2
    __is_debug2 = __logger.is_debug2

    def __init__(self, num_classes,
                 num_decomps, num_subsets, num_mixtures,
                 input_dist=DenseSPNGenerator.InputDist.MIXTURE,
                 num_input_mixtures=None,
                 weight_initializer=tf.initializers.random_uniform(0.0, 1.0)):
        super().__init__()
        if not isinstance(num_classes, int):
            raise ValueError("num_classes must be an integer")
        self._num_classes = num_classes
        self._num_decomps = num_decomps
        self._num_subsets = num_subsets
        self._num_mixtures = num_mixtures
        self._input_dist = input_dist
        self._num_input_mixtures = num_input_mixtures
        self._weight_initializer = weight_initializer
        self._class_latent_indicators = None
        self._sample_latent_indicators = None
        self._class_input = None
        self._sample_inputs = None

    def __repr__(self):
        return (type(self).__qualname__ + "(" +
                ("num_classes=" + str(self._num_classes)) + ", " +
                ("num_decomps=" + str(self._num_decomps)) + ", " +
                ("num_subsets=" + str(self._num_subsets)) + ", " +
                ("num_mixtures=" + str(self._num_mixtures)) + ", " +
                ("input_dist=" + str(self._input_dist)) + ", " +
                ("num_input_mixtures=" + str(self._num_input_mixtures)) + ", " +
                ("weight_init_value=" + str(self._weight_initializer))
                + ")")

    @utils.docinherit(Model)
    def serialize(self, save_param_vals=True, sess=None):
        # Serialize the graph first
        data = serialize_graph(self._root, save_param_vals=save_param_vals,
                               sess=sess)
        # Add model specific information
        # Inputs
        if self._sample_latent_indicators is not None:
            data['sample_latent_indicators'] = self._sample_latent_indicators.name
        data['sample_inputs'] = [(i.node.name, i.indices)
                                 for i in self._sample_inputs]
        if self._class_latent_indicators is not None:
            data['class_latent_indicators'] = self._class_latent_indicators.name
        if self._class_input:
            data['class_input'] = (self._class_input.node.name,
                                   self._class_input.indices)
        # Model params
        data['num_classes'] = self._num_classes
        data['num_decomps'] = self._num_decomps
        data['num_subsets'] = self._num_subsets
        data['num_mixtures'] = self._num_mixtures
        data['input_dist'] = self._input_dist
        data['num_input_mixtures'] = self._num_input_mixtures
        data['weight_init_value'] = self._weight_initializer
        return data

    @utils.docinherit(Model)
    def deserialize(self, data, load_param_vals=True, sess=None):
        # Deserialize the graph first
        nodes_by_name = {}
        self._root = deserialize_graph(data, load_param_vals=load_param_vals,
                                       sess=sess,
                                       nodes_by_name=nodes_by_name)
        # Model specific information
        # Inputs
        sample_latent_indicators = data.get('sample_latent_indicators', None)
        if sample_latent_indicators:
            self._sample_latent_indicators = nodes_by_name[sample_latent_indicators]
        else:
            self._sample_latent_indicators = None
        self._sample_inputs = tuple(Input(nodes_by_name[nn], i)
                                    for nn, i in data['sample_inputs'])
        class_latent_indicators = data.get('class_latent_indicators', None)
        if class_latent_indicators:
            self._class_latent_indicators = nodes_by_name[class_latent_indicators]
        else:
            self._class_latent_indicators = None
        class_input = data.get('class_input', None)
        if class_input:
            self._class_input = Input(nodes_by_name[class_input[0]], class_input[1])
        else:
            self._class_input = None
        # Model params
        self._num_classes = data['num_classes']
        self._num_decomps = data['num_decomps']
        self._num_subsets = data['num_subsets']
        self._num_mixtures = data['num_mixtures']
        self._input_dist = data['input_dist']
        self._num_input_mixtures = data['num_input_mixtures']
        self._weight_initializer = data['weight_init_value']

    @property
    def sample_latent_indicators(self):
        """IndicatorLeaf: IndicatorLeaf with input data sample."""
        return self._sample_latent_indicators

    @property
    def class_latent_indicators(self):
        """IndicatorLeaf: Class indicator variables."""
        return self._class_latent_indicators

    @property
    def sample_inputs(self):
        """list of Input: Inputs to the model providing data samples."""
        return self._sample_inputs

    @property
    def class_input(self):
        """Input: Input providing class indicators.."""
        return self._class_input

    def build(self, *sample_inputs, class_input=None, num_vars=None,
              num_vals=None, seed=None):
        """Build the SPN graph of the model.

        The model can be built on top of any ``sample_inputs``. Otherwise, if no
        sample inputs are provided, the model will internally crate a single IndicatorLeaf
        node to represent the input data samples. In such case, ``num_vars`` and
        ``num_vals`` must be specified.

        Similarly, if ``class_input`` is provided, it is used as a source of
        class indicators of the root sum node combining sub-SPNs modeling
        particular classes. Otherwise, an internal IndicatorLeaf node is created for this
        purpose.

        Args:
            *sample_inputs (input_like): Optional. Inputs to the model
                                         providing data samples.
            class_input (input_like): Optional. Input providing class indicators.
            num_vars (int): Optional. Number of variables in each sample. Must
                            only be provided if ``sample_inputs`` are not given.
            num_vals (int or list of int): Optional. Number of values of each
                variable. Can be a single value or a list of values, one for
                each of ``num_vars`` variables. Must only be provided if
                ``sample_inputs`` are not given.
            seed (int): Optional. Seed used for the dense SPN generator.

        Returns:
           Sum: Root node of the generated model.
        """
        if not sample_inputs:
            if num_vars is None:
                raise ValueError("num_vars must be given when sample_inputs are not")
            if num_vals is None:
                raise ValueError("num_vals must be given when sample_inputs are not")
            if not isinstance(num_vars, int) or num_vars < 1:
                raise ValueError("num_vars must be an integer > 0")
            if not isinstance(num_vals, int) or num_vals < 1:
                raise ValueError("num_vals must be an integer > 0")
        if self._num_classes > 1:
            self.__info("Building a discrete dense model with %d classes" %
                        self._num_classes)
        else:
            self.__info("Building a 1-class discrete dense model")

        # Create IndicatorLeaf if inputs not given
        if not sample_inputs:
            self._sample_latent_indicators = IndicatorLeaf(num_vars=num_vars, num_vals=num_vals,
                                             name="SampleIndicatorLeaf")
            self._sample_inputs = [Input(self._sample_latent_indicators)]
        else:
            self._sample_inputs = tuple(Input.as_input(i) for i in sample_inputs)
        if self._num_classes > 1:
            if class_input is None:
                self._class_latent_indicators = IndicatorLeaf(num_vars=1, num_vals=self._num_classes,
                                                name="ClassIndicatorLeaf")
                self._class_input = Input(self._class_latent_indicators)
            else:
                self._class_input = Input.as_input(class_input)

        # Generate structure
        dense_gen = DenseSPNGenerator(num_decomps=self._num_decomps,
                                      num_subsets=self._num_subsets,
                                      num_mixtures=self._num_mixtures,
                                      input_dist=self._input_dist,
                                      num_input_mixtures=self._num_input_mixtures,
                                      balanced=True)
        rnd = random.Random(seed)
        if self._num_classes == 1:
            # One-class
            self._root = dense_gen.generate(*self._sample_inputs, rnd=rnd,
                                            root_name='Root')
        else:
            # Multi-class: create sub-SPNs
            sub_spns = []
            for c in range(self._num_classes):
                rnd_copy = random.Random()
                rnd_copy.setstate(rnd.getstate())
                with tf.name_scope("Class%d" % c):
                    sub_root = dense_gen.generate(*self._sample_inputs,
                                                  rnd=rnd_copy)
                if self.__is_debug1():
                    self.__debug1("sub-SPN %d has %d nodes" %
                                  (c, sub_root.get_num_nodes()))
                sub_spns.append(sub_root)
            # Create root
            self._root = Sum(*sub_spns, latent_indicators=self._class_input, name="Root")

        if self.__is_debug1():
            self.__debug1("SPN graph has %d nodes" % self._root.get_num_nodes())

        # Generate weight nodes
        self.__debug1("Generating weight nodes")
        generate_weights(self._root, initializer=self._weight_initializer)
        if self.__is_debug1():
            self.__debug1("SPN graph has %d nodes and %d TF ops" % (
                self._root.get_num_nodes(), self._root.get_tf_graph_size()))

        return self._root
