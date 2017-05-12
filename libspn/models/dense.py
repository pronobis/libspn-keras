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


class DenseModel(Model):

    logger = get_logger()
    info = logger.info
    debug1 = logger.debug1
    is_debug1 = logger.is_debug1

    def __init__(self, num_decomps, num_subsets, num_mixtures,
                 input_dist=DenseSPNGenerator.InputDist.MIXTURE,
                 num_input_mixtures=None):
        super().__init__()
        self._num_decomps = num_decomps
        self._num_subsets = num_subsets
        self._num_mixtures = num_mixtures
        self._input_dist = input_dist
        self._num_input_mixtures = num_input_mixtures
        self._ivs = None

    def build_spn(self, dataset):
        self.info("Building SPN")

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
        dense_gen = DenseSPNGenerator(num_decomps=self._num_decomps,
                                      num_subsets=self._num_subsets,
                                      num_mixtures=self._num_mixtures,
                                      input_dist=self._input_dist,
                                      num_input_mixtures=self._num_input_mixtures,
                                      balanced=True)
        self._root = dense_gen.generate(self._ivs)
        if self.is_debug1():
            self.debug1("SPN graph with %d nodes" % self._root.get_num_nodes())

        # Generate weights
        self.debug1("Generating weights")
        weight_init_value = ValueType.RANDOM_UNIFORM(0, 1)
        generate_weights(self._root, init_value=weight_init_value)
        if self.is_debug1():
            self.debug1("SPN graph with %d nodes and %d TF ops" % (
                self._root.get_num_nodes(), self._root.get_tf_graph_size()))

    def train(self):
        pass

    def test(self):
        pass
