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
from enum import Enum


class DenseModel(Model):

    __logger = get_logger()
    __info = __logger.info
    __debug1 = __logger.debug1
    __is_debug1 = __logger.is_debug1

    def __init__(self):
        super().__init__()
        self._ivs = None
        self._root = None

    def build_spn(self, dataset, num_decomps, num_subsets, num_mixtures,
                  input_dist=DenseSPNGenerator.InputDist.MIXTURE,
                  num_input_mixtures=None,
                  weight_init_value=ValueType.RANDOM_UNIFORM(0, 1)):
        self.__info("Building SPN")

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

    def train(self, value_inference_type=InferenceType.MARGINAL,
              init_accum_value=20, additive_smoothing_value=0.0,
              additive_smoothing_decay=0.2, additive_smoothing_min=0.0,
              stop_condition=0.0):
        self.__info("Adding EM learning ops")
        additive_smoothing_var = tf.Variable(additive_smoothing_value,
                                             dtype=conf.dtype)
        em_learning = EMLearning(
            self._root, log=True,
            value_inference_type=value_inference_type,
            additive_smoothing=additive_smoothing_var,
            add_random=None,
            initial_accum_value=init_accum_value,
            use_unweighted=True)
        reset_accumulators = em_learning.reset_accumulators()
        accumulate_updates = em_learning.accumulate_updates()
        update_spn = em_learning.update_spn()
        train_likelihood = em_learning.value.values[self._root]
        avg_train_likelihood = tf.reduce_mean(train_likelihood)
        self.__info("Adding weight initialization ops")
        init_weights = initialize_weights(self._root)

        self.__info("Learning...")

    def save_graph(self):
        self.__info("Saving TensorFlow graph")
        writer = tf.summary.FileWriter("./logs",
                                       self._root.tf_graph)
        writer.flush()

    def test(self):
        pass
