# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

from collections import namedtuple
import tensorflow as tf

from libspn.graph.distribution import GaussianLeaf
from libspn.inference.mpe_path import MPEPath
from libspn.graph.algorithms import traverse_graph
from libspn import conf


class EMLearning():
    """Assembles TF operations performing EM learning of an SPN.

    Args:
        mpe_path (MPEPath): Pre-computed MPE_path.
        value_inference_type (InferenceType): The inference type used during the
            upwards pass through the SPN. Ignored if ``mpe_path`` is given.
        log (bool): If ``True``, calculate the value in the log space. Ignored
                    if ``mpe_path`` is given.
    """

    ParamNode = namedtuple("ParamNode", ["node", "name_scope", "accum"])
    GaussianLeafNode = namedtuple(
        "GaussianLeafNode", ["node", "name_scope", "accum", "sum_data", "sum_data_squared"])

    def __init__(self, root, mpe_path=None, log=True, value_inference_type=None,
                 additive_smoothing=None, add_random=None, initial_accum_value=None,
                 use_unweighted=False, sample=False, sample_prob=None,
                 dropconnect_keep_prob=None):
        self._root = root
        self._log = log
        self._additive_smoothing = additive_smoothing
        self._initial_accum_value = initial_accum_value
        self._sample = sample
        # Create internal MPE path generator
        if mpe_path is None:
            self._mpe_path = MPEPath(
                log=log, value_inference_type=value_inference_type, add_random=add_random,
                use_unweighted=use_unweighted, sample=sample, sample_prob=sample_prob,
                dropconnect_keep_prob=dropconnect_keep_prob)
        else:
            self._mpe_path = mpe_path
        # Create a name scope
        with tf.name_scope("EMLearning") as self._name_scope:
            pass
        # Create accumulators
        self._create_accumulators()

    @property
    def mpe_path(self):
        """MPEPath: Computed MPE path."""
        return self._mpe_path

    @property
    def value(self):
        """Value or LogValue: Computed SPN values."""
        return self._mpe_path.value

    # TODO: For testing only
    def root_accum(self):
        for pn in self._param_nodes:
            if pn.node == self._root.weights.node:
                return pn.accum
        return None

    def reset_accumulators(self):
        with tf.name_scope(self._name_scope):
            return tf.group(*(
                    [pn.accum.initializer for pn in self._param_nodes] +
                    [dn.accum.initializer for dn in self._gaussian_leaf_nodes] +
                    [dn.sum_data.initializer for dn in self._gaussian_leaf_nodes] +
                    [dn.sum_data_squared.initializer for dn in self._gaussian_leaf_nodes] +
                    [dn.node._total_count_variable.initializer
                     for dn in self._gaussian_leaf_nodes]),
                            name="reset_accumulators")

    def accumulate_updates(self):
        # Generate path if not yet generated
        if not self._mpe_path.counts:
            self._mpe_path.get_mpe_path(self._root)

        # Generate all accumulate operations
        with tf.name_scope(self._name_scope):
            assign_ops = []
            for pn in self._param_nodes:
                with tf.name_scope(pn.name_scope):
                    # counts = self._mpe_path.counts[pn.node]
                    # update_value = pn.node._compute_hard_em_update(counts)
                    # with tf.control_dependencies([update_value]):
                    # op = tf.assign_add(pn.accum, self._mpe_path.counts[pn.node])
                    counts_summed_batch = pn.node._compute_hard_em_update(
                        self._mpe_path.counts[pn.node])
                    assign_ops.append(tf.assign_add(pn.accum, counts_summed_batch))

            for dn in self._gaussian_leaf_nodes:
                with tf.name_scope(dn.name_scope):
                    counts = self._mpe_path.counts[dn.node]
                    update_value = dn.node._compute_hard_em_update(counts)
                    with tf.control_dependencies(update_value.values()):
                        assign_ops.append(tf.assign_add(dn.accum, update_value['accum']))
                        assign_ops.append(tf.assign_add(dn.sum_data, update_value['sum_data']))
                        assign_ops.append(tf.assign_add(
                            dn.sum_data_squared, update_value['sum_data_squared']))

            return tf.group(*assign_ops, name="accumulate_updates")

    def update_spn(self):
        # Generate all update operations
        with tf.name_scope(self._name_scope):
            assign_ops = []
            for pn in self._param_nodes:
                with tf.name_scope(pn.name_scope):
                    accum = pn.accum
                    if self._additive_smoothing is not None:
                        accum = tf.add(accum, self._additive_smoothing)
                    if pn.node.log:
                        assign_ops.append(pn.node.assign_log(tf.log(accum)))
                    else:
                        assign_ops.append(pn.node.assign(accum))

            for dn in self._gaussian_leaf_nodes:
                with tf.name_scope(dn.name_scope):
                    assign_ops.extend(dn.node.assign(dn.accum, dn.sum_data, dn.sum_data_squared))

            return tf.group(*assign_ops, name="update_spn")

    def learn(self):
        """Assemble TF operations performing EM learning of the SPN."""
        return None

    def _create_accumulators(self):
        def fun(node):
            if node.is_param:
                with tf.name_scope(node.name) as scope:
                    if self._initial_accum_value is not None:
                        if node.mask and not all(node.mask):
                            accum = tf.Variable(tf.cast(tf.reshape(node.mask,
                                                node.variable.shape),
                                                dtype=conf.dtype) *
                                                self._initial_accum_value,
                                                dtype=conf.dtype,
                                                collections=['em_accumulators'])
                        else:
                            accum = tf.Variable(tf.ones_like(node.variable,
                                                             dtype=conf.dtype) *
                                                self._initial_accum_value,
                                                dtype=conf.dtype,
                                                collections=['em_accumulators'])
                    else:
                        accum = tf.Variable(tf.zeros_like(node.variable,
                                                          dtype=conf.dtype),
                                            dtype=conf.dtype,
                                            collections=['em_accumulators'])
                    param_node = EMLearning.ParamNode(node=node, accum=accum,
                                                      name_scope=scope)
                    self._param_nodes.append(param_node)
            if isinstance(node, GaussianLeaf) and node.learn_distribution_parameters:
                with tf.name_scope(node.name) as scope:
                    if self._initial_accum_value is not None:
                        accum = tf.Variable(tf.ones_like(node.loc_variable, dtype=conf.dtype) *
                                            self._initial_accum_value,
                                            dtype=conf.dtype,
                                            collections=['em_accumulators'])
                        sum_x = tf.Variable(node.loc_variable * self._initial_accum_value,
                                            dtype=conf.dtype, collections=['em_accumulators'])
                        sum_x2 = tf.Variable(tf.square(node.loc_variable) *
                                             self._initial_accum_value,
                                             dtype=conf.dtype, collections=['em_accumulators'])
                    else:
                        accum = tf.Variable(tf.zeros_like(node.loc_variable, dtype=conf.dtype),
                                            dtype=conf.dtype,
                                            collections=['em_accumulators'])
                        sum_x = tf.Variable(tf.zeros_like(node.loc_variable), dtype=conf.dtype,
                                            collections=['em_accumulators'])
                        sum_x2 = tf.Variable(tf.zeros_like(node.loc_variable), dtype=conf.dtype,
                                             collections=['em_accumulators'])
                    gaussian_node = EMLearning.GaussianLeafNode(
                        node=node, accum=accum, sum_data=sum_x, sum_data_squared=sum_x2,
                        name_scope=scope)
                    self._gaussian_leaf_nodes.append(gaussian_node)

        self._gaussian_leaf_nodes = []
        self._param_nodes = []
        with tf.name_scope(self._name_scope):
            traverse_graph(self._root, fun=fun)

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
