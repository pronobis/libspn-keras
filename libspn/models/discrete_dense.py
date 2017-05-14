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


class DiscreteDenseModel(Model):

    __logger = get_logger()
    __info = __logger.info
    __debug1 = __logger.debug1
    __is_debug1 = __logger.is_debug1

    def __init__(self, num_decomps, num_subsets, num_mixtures,
                 input_dist=DenseSPNGenerator.InputDist.MIXTURE,
                 num_input_mixtures=None,
                 weight_init_value=ValueType.RANDOM_UNIFORM(0, 1)):
        super().__init__()
        self._num_decomps = num_decomps
        self._num_subsets = num_subsets
        self._num_mixtures = num_mixtures
        self._input_dist = input_dist
        self._num_input_mixtures = num_input_mixtures
        self._weight_init_value = weight_init_value
        self._ivs = None

    def build(self, train_data):
        """Builds the SPN graph of the model. """
        # # TODO: Check if data is of type int
        # self.__info("Building discrete dense model")

        # # Get data from dataset
        # data = dataset.get_data()
        # if isinstance(data, list):
        #     samples = data[0]
        # else:
        #     samples = data

        # # Get number of variables
        # num_vars = int(samples.shape[1])
        # num_vals = 2

        # # IVs
        # self._ivs = IVs(num_vars=num_vars, num_vals=num_vals)

        # # Generate structure
        # dense_gen = DenseSPNGenerator(num_decomps=num_decomps,
        #                               num_subsets=num_subsets,
        #                               num_mixtures=num_mixtures,
        #                               input_dist=input_dist,
        #                               num_input_mixtures=num_input_mixtures,
        #                               balanced=True)
        # self._root = dense_gen.generate(self._ivs)
        # if self.__is_debug1():
        #     self.__debug1("SPN graph with %d nodes" % self._root.get_num_nodes())

        # # Generate weights
        # self.__debug1("Generating weights")
        # generate_weights(self._root, init_value=weight_init_value)
        # if self.__is_debug1():
        #     self.__debug1("SPN graph with %d nodes and %d TF ops" % (
        #         self._root.get_num_nodes(), self._root.get_tf_graph_size()))

    # def learn(self, value_inference_type=InferenceType.MARGINAL,
    #           init_accum_value=20, additive_smoothing_value=0.0,
    #           additive_smoothing_decay=0.2, additive_smoothing_min=0.0,
    #           stop_condition=0.0):
    #     self.__info("Adding EM learning ops")
    #     additive_smoothing_var = tf.Variable(additive_smoothing_value,
    #                                          dtype=conf.dtype,
    #                                          name="AdditiveSmoothing")
    #     em_learning = EMLearning(
    #         self._root, log=True,
    #         value_inference_type=value_inference_type,
    #         additive_smoothing=additive_smoothing_var,
    #         add_random=None,
    #         initial_accum_value=init_accum_value,
    #         use_unweighted=True)
    #     reset_accumulators = em_learning.reset_accumulators()
    #     accumulate_updates = em_learning.accumulate_updates()
    #     update_spn = em_learning.update_spn()
    #     train_likelihood = em_learning.value.values[self._root]
    #     avg_train_likelihood = tf.reduce_mean(train_likelihood,
    #                                           name="AverageTrainLikelihood")
    #     self.__info("Adding weight initialization ops")
    #     init_weights = initialize_weights(self._root)

    #     self.__info("Initializing")
    #     self._sess.run(init_weights)
    #     self._sess.run(reset_accumulators)

    #     # self.__info("Learning")
    #     # num_batches = 1
    #     # batch_size = self._data.training_scans.shape[0] // num_batches
    #     # prev_likelihood = 100
    #     # likelihood = 0
    #     # epoch = 0
    #     # # Print weights
    #     # print(self._sess.run(self._root.weights.node.variable))
    #     # print(self._sess.run(self._em_learning.root_accum()))

    #     # while abs(prev_likelihood - likelihood) > stop_condition:
    #     #     prev_likelihood = likelihood
    #     #     likelihoods = []
    #     #     for batch in range(num_batches):
    #     #         start = (batch) * batch_size
    #     #         stop = (batch + 1) * batch_size
    #     #         print("- EPOCH", epoch, "BATCH", batch, "SAMPLES", start, stop)
    #     #         # Adjust smoothing
    #     #         ads = max(np.exp(-epoch * additive_smoothing_decay) *
    #     #                   self._additive_smoothing_value,
    #     #                   additive_smoothing_min)
    #     #         self._sess.run(self._additive_smoothing_var.assign(ads))
    #     #         print("  Smoothing: ", self._sess.run(self._additive_smoothing_var))
    #     #         # Run accumulate_updates
    #     #         train_likelihoods_arr, avg_train_likelihood_val, _, = \
    #     #             self._sess.run([self._train_likelihood,
    #     #                             self._avg_train_likelihood,
    #     #                             self._accumulate_updates],
    #     #                            feed_dict={self._ivs:
    #     #                                       self._data.training_scans[start:stop]})
    #     #         # Print avg likelihood of this batch data on previous batch weights
    #     #         print("  Avg likelihood (this batch data on previous weights): %s" %
    #     #               (avg_train_likelihood_val))
    #     #         likelihoods.append(avg_train_likelihood_val)
    #     #         # Update weights
    #     #         self._sess.run(self._update_spn)
    #     #         # Print weights
    #     #         print(self._sess.run(self._root.weights.node.variable))
    #     #         print(self._sess.run(self._em_learning.root_accum()))
    #     #     likelihood = sum(likelihoods) / len(likelihoods)
    #     #     print("- Batch avg likelihood: %s" % (likelihood))
    #     #     epoch += 1
