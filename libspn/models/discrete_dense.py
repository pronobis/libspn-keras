# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

from libspn.models.model import Model
from libspn.log import get_logger
from libspn.graph.ivs import IVs
from libspn.generation.dense import DenseSPNGenerator
from libspn.utils.math import ValueType
from libspn.generation.weights import generate_weights
from libspn import conf
from libspn.learning.em import EMLearning
from libspn.graph.weights import initialize_weights
from libspn.inference.type import InferenceType
import tensorflow as tf
from libspn import utils


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
        input_dist (InputDist): Determines how IVs of the discrete variables for
                                data samples are connected to the model.
        num_input_mixtures (int): Number of mixtures used representing each
                                  discrete data variable (mixing the data variable
                                  IVs) when ``input_dist`` is set to ``MIXTURE``.
                                  If set to ``None``, ``num_mixtures`` is used.
        weight_init_value: Initial value of the weights. For possible values,
                           see :meth:`~libspn.utils.broadcast_value`.
    """

    __logger = get_logger()
    __info = __logger.info
    __debug1 = __logger.debug1
    __is_debug1 = __logger.is_debug1

    def __init__(self, num_vars, num_vals, num_classes,
                 num_decomps, num_subsets, num_mixtures,
                 input_dist=DenseSPNGenerator.InputDist.MIXTURE,
                 num_input_mixtures=None,
                 weight_init_value=ValueType.RANDOM_UNIFORM(0, 1)):
        super().__init__()
        if not isinstance(num_vars, int):
            raise ValueError("num_vars must be an integer")
        self._num_vars = num_vars
        if not isinstance(num_vals, int):
            raise ValueError("num_vals must be an integer")
        self._num_vals = num_vals
        if not isinstance(num_classes, int):
            raise ValueError("num_classes must be an integer")
        self._num_classes = num_classes
        self._num_decomps = num_decomps
        self._num_subsets = num_subsets
        self._num_mixtures = num_mixtures
        self._input_dist = input_dist
        self._num_input_mixtures = num_input_mixtures
        self._weight_init_value = weight_init_value
        self._ivs = None

    def build(self, *inputs, class_ivs=None, dataset=None):
        """Build the SPN graph of the model.

        The model can be built on top of any ``inputs``. Otherwise, if no inputs
        are provided, the model will internally crate an IVs node to represent
        the inputs. Similarly, if ``class_ivs`` is provided, it is used as a
        source of indicators of the root sum node combining sub-SPNs modeling
        partic ular classes. Othewise, an internal IVs node is created for this
        purpose.

        Furthermore, if ``dataset`` is provided, it will be connected to the
        inputs of the model automatically.

        Args:
            inputs (input_like): Optional. Inputs to the model.
            class_ivs (input_like): Optional. Inputs providing class indicators.
            dataset (Dataset): Optional. Dataset providing data to the model.

        Returns:
           Sum: Root node of the generated model.
        """
        self.__info("Building discrete dense model with %d classes for "
                    "%d variables with %d values" %
                    (self._num_classes, self._num_vars, self._num_vals))

        # Get data from dataset
        samples = None
        labels = None
        if dataset is not None:
            data = dataset.get_data()
            if isinstance(data, list):
                samples, labels = data
            else:
                samples = data

        # Create inputs if not given
        if inputs is None:
            inputs = [IVs(feed=samples,
                          num_vars=self._num_vars,
                          num_vals=self._num_vals)]

        # # TODO: Check if data is of type int

        # Get data from dataset
        data = dataset.get_data()
        if isinstance(data, list):
            samples = data[0]
        else:
            samples = data

        # Get number of variables
        num_vars = int(samples.shape[1])
        num_vals = 2

        # IVs
        self._ivs = IVs(num_vars=num_vars, num_vals=num_vals)

        # Generate structure
        dense_gen = DenseSPNGenerator(num_decomps=num_decomps,
                                      num_subsets=num_subsets,
                                      num_mixtures=num_mixtures,
                                      input_dist=input_dist,
                                      num_input_mixtures=num_input_mixtures,
                                      balanced=True)
        self._root = dense_gen.generate(self._ivs)
        if self.__is_debug1():
            self.__debug1("SPN graph with %d nodes" % self._root.get_num_nodes())

        # Generate weights
        self.__debug1("Generating weights")
        generate_weights(self._root, init_value=weight_init_value)
        if self.__is_debug1():
            self.__debug1("SPN graph with %d nodes and %d TF ops" % (
                self._root.get_num_nodes(), self._root.get_tf_graph_size()))
