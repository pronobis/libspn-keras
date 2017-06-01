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

    def __init__(self, num_classes,
                 num_decomps, num_subsets, num_mixtures,
                 input_dist=DenseSPNGenerator.InputDist.MIXTURE,
                 num_input_mixtures=None,
                 weight_init_value=ValueType.RANDOM_UNIFORM(0, 1)):
        super().__init__()
        if not isinstance(num_classes, int):
            raise ValueError("num_classes must be an integer")
        self._num_classes = num_classes
        self._num_decomps = num_decomps
        self._num_subsets = num_subsets
        self._num_mixtures = num_mixtures
        self._input_dist = input_dist
        self._num_input_mixtures = num_input_mixtures
        self._weight_init_value = weight_init_value
        self._class_ivs = None

    @property
    def sample_ivs(self):
        """IVs: IVs with input data sample."""
        return self._sample_ivs

    @property
    def class_ivs(self):
        """IVs: Class indicator variables."""
        return self._class_ivs

    def build(self, *sample_inputs, class_input=None, num_vars=None, num_vals=None):
        """Build the SPN graph of the model.

        The model can be built on top of any ``sample_inputs``. Otherwise, if no
        sample inputs are provided, the model will internally crate a single IVs
        node to represent the input data samples. In such case, ``num_vars`` and
        ``num_vals`` must be specified.

        Similarly, if ``class_input`` is provided, it is used as a source of
        class indicators of the root sum node combining sub-SPNs modeling
        particular classes. Otherwise, an internal IVs node is created for this
        purpose.

        Args:
            *sample_inputs (input_like): Optional. Inputs to the model
                                         providing data samples.
            class_input (input_like): Optional. Input providing class indicators.
            num_vars (int): Optional. Number of input variables of the model.
                            Must only be provided if ``inputs`` are not given.
            num_vals (int): Optional. Number of values of each input variable.
                            Must only be provided if ``inputs`` are not given.

        Returns:
           Sum: Root node of the generated model.
        """
        if not sample_inputs:
            if not isinstance(num_vars, int) or num_vars < 1:
                raise ValueError("num_vars must be an integer > 0")
            if not isinstance(num_vals, int) or num_vals < 1:
                raise ValueError("num_vals must be an integer > 0")
        if self._num_classes > 1:
            self.__info("Building a discrete dense model with %d classes" %
                        self._num_classes)
        else:
            self.__info("Building a 1-class discrete dense model")

        # Create IVs if inputs not given
        if not sample_inputs:
            self._sample_ivs = IVs(num_vars=self._num_vars,
                                   num_vals=self._num_vals)
            sample_inputs = [self._sample_ivs]
        if self._num_classes > 1 and class_input is None:
            self._class_ivs = IVs(num_vars=1, num_vals=self._num_classes)
            class_input = self._class_ivs

        # Generate structure
        dense_gen = DenseSPNGenerator(num_decomps=self._num_decomps,
                                      num_subsets=self._num_subsets,
                                      num_mixtures=self._num_mixtures,
                                      input_dist=self._input_dist,
                                      num_input_mixtures=self._num_input_mixtures,
                                      balanced=True)
        self._root = dense_gen.generate(self._ivs)
        if self.__is_debug1():
            self.__debug1("SPN graph has %d nodes" % self._root.get_num_nodes())

        # Generate weights
        self.__debug1("Generating weight nodes")
        generate_weights(self._root, init_value=self._weight_init_value)
        if self.__is_debug1():
            self.__debug1("SPN graph has %d nodes and %d TF ops" % (
                self._root.get_num_nodes(), self._root.get_tf_graph_size()))
