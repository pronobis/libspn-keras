#!/usr/bin/env python3

import libspn as spn
from libspn.tests.test import argsprod
import tensorflow as tf
import numpy as np
import scipy.stats as stats


# Batch size is pretty large to obtain good approximations
BATCH_SIZE = int(1e4)


class TestGaussianQuantile(tf.test.TestCase):

    def test_split_in_quantiles(self):
        quantiles = [np.random.rand(32, 32) + i * 2 for i in range(4)]
        data = np.concatenate(quantiles, axis=0)
        np.random.shuffle(data)
        gq = spn.NormalLeaf(num_vars=32, num_components=4)

        values_per_quantile = gq._split_in_quantiles(data, 4)

        for val, q in zip(values_per_quantile, quantiles):
            self.assertAllClose(np.sort(q, axis=0), val)

    def test_compute_scope(self):
        gl = spn.NormalLeaf(num_vars=32, num_components=4)
        scope = gl._compute_scope()
        for b in range(0, len(scope), 4):
            [self.assertEqual(scope[b], scope[b + i]) for i in range(1, 4)]
            [self.assertNotEqual(scope[b], scope[b + i]) for i in range(4, len(scope) - b, 4)]

    @argsprod([False, True])
    def test_learn_from_data(self, softplus):
        quantiles = [np.random.rand(32, 32) + i * 2 for i in range(4)]
        data = np.concatenate(quantiles, axis=0)
        np.random.shuffle(data)
        gq = spn.NormalLeaf(
            num_vars=32, num_components=4, initialization_data=data, softplus_scale=softplus)
        true_vars = np.stack([np.var(q, axis=0) for q in quantiles], axis=-1)
        true_means = np.stack([np.mean(q, axis=0) for q in quantiles], axis=-1)

        if softplus:
            self.assertAllClose(np.log(1 + np.exp(gq._scale_init)), np.sqrt(true_vars))
        else:
            self.assertAllClose(gq._scale_init, np.sqrt(true_vars))
        self.assertAllClose(gq._loc_init, true_means)

    def test_learn_from_data_prior(self):
        prior_beta = 3.0
        prior_alpha = 2.0
        N = 32
        quantiles = [np.random.rand(N, 32) + i * 2 for i in range(4)]
        data = np.concatenate(quantiles, axis=0)
        np.random.shuffle(data)
        gq = spn.NormalLeaf(
            num_vars=32, num_components=4, initialization_data=data,
            prior_alpha=prior_alpha, prior_beta=prior_beta, use_prior=True, softplus_scale=False)

        mus = [np.mean(q, axis=0, keepdims=True) for q in quantiles]
        ssq = np.stack([np.sum((x - mu) ** 2, axis=0) for x, mu in zip(quantiles, mus)], axis=-1)
        true_vars = (2 * prior_beta + ssq) / (2 * prior_alpha + 2 + N)

        self.assertAllClose(gq._scale_init, np.sqrt(true_vars))

    # def test_sum_update_1(self):
    #     child1 = spn.NormalLeaf(num_vars=1, num_components=1, total_counts_init=3.0,
    #                             loc_init=0.0, scale_init=1.0)
    #     child2 = spn.NormalLeaf(num_vars=1, num_components=1, total_counts_init=7.0,
    #                             loc_init=1.0, scale_init=4.0)
    #     root = spn.Sum(child1, child2)
    #     root.generate_weights()
    #
    #     value_inference_type = spn.InferenceType.MARGINAL
    #     init_weights = spn.initialize_weights(root)
    #     learning = spn.HardEMLearning(root, log=True, value_inference_type=value_inference_type,
    #                               use_unweighted=True)
    #     reset_accumulators = learning.reset_accumulators()
    #     accumulate_updates = learning.accumulate_updates()
    #     update_spn = learning.update_spn()
    #     train_likelihood = learning.value.values[root]
    #
    #     with self.test_session() as sess:
    #         sess.run(init_weights)
    #         sess.run(reset_accumulators)
    #         sess.run(accumulate_updates, {child1: [[0.0]], child2: [[0.0]]})
    #         sess.run(update_spn)
    #
    #         child1_n = sess.run(child1._total_count_variable)
    #         child2_n = sess.run(child2._total_count_variable)
    #
    #     # equalWeight is true, so update passes the data point to the component
    #     # with highest likelihood without considering the weight of each component.
    #     # In this case, N(0|0,1) > N(0|1,4), so child1 is picked.
    #     # If component weights are taken into account, then child2 will be picked
    #     # since 0.3*N(0|0,1) < 0.7*N(0|1,4).
    #     # self.assertEqual(root.n, 11)
    #     self.assertEqual(child1_n, 4)
    #     self.assertEqual(child2_n, 7)

    # @argsprod([True])
    # def test_param_learning(self, softplus_scale):
    #     spn.conf.argmax_zero = True
    #     num_vars = 2
    #     num_components = 2
    #     batch_size = 32
    #     count_init = 100.0
    #
    #     # Create means and variances
    #     means = np.array([[0, 1],
    #                       [10, 15]]).astype(np.float32)
    #     vars = np.array([[0.25, 0.5],
    #                      [0.33, 0.67]])
    #
    #     # Sample some data
    #     data0 = [stats.norm(loc=m, scale=np.sqrt(v)).rvs(batch_size//2).astype(np.float32)
    #              for m, v in zip(means[0], vars[0])]
    #     data1 = [stats.norm(loc=m, scale=np.sqrt(v)).rvs(batch_size//2).astype(np.float32)
    #              for m, v in zip(means[1], vars[1])]
    #     data = np.stack([np.concatenate(data0), np.concatenate(data1)], axis=-1)
    #
    #     # Set up SPN
    #     gq = spn.NormalLeaf(num_vars=num_vars, num_components=num_components,
    #                         initialization_data=data, total_counts_init=count_init,
    #                         softplus_scale=softplus_scale, trainable_scale=True)
    #
    #     mixture00 = spn.Sum((gq, [0, 1]), name="Mixture00")
    #     weights00 = spn.Weights(initializer=tf.initializers.constant([0.25, 0.75]), num_weights=2)
    #     mixture00.set_weights(weights00)
    #     mixture01 = spn.Sum((gq, [0, 1]), name="Mixture01")
    #     weights01 = spn.Weights(initializer=tf.initializers.constant([0.75, 0.25]), num_weights=2)
    #     mixture01.set_weights(weights01)
    #
    #     mixture10 = spn.Sum((gq, [2, 3]), name="Mixture10")
    #     weights10 = spn.Weights(initializer=tf.initializers.constant([2/3, 1/3]), num_weights=2)
    #     mixture10.set_weights(weights10)
    #     mixture11 = spn.Sum((gq, [2, 3]), name="Mixture11")
    #     weights11 = spn.Weights(initializer=tf.initializers.constant([1/3, 2/3]), num_weights=2)
    #     mixture11.set_weights(weights11)
    #
    #     prod0 = spn.Product(mixture00, mixture10, name="Prod0")
    #     prod1 = spn.Product(mixture01, mixture11, name="Prod1")
    #
    #     root = spn.Sum(prod0, prod1, name="Root")
    #     root_weights = spn.Weights(initializer=tf.initializers.constant([1/2, 1/2]), num_weights=2)
    #     root.set_weights(root_weights)
    #
    #     # Generate new data from slightly shifted Gaussians
    #     data0 = np.concatenate(
    #         [stats.norm(loc=m, scale=np.sqrt(v)).rvs(batch_size//2).astype(np.float32)
    #          for m, v in zip(means[0] + 0.2, vars[0])])
    #     data1 = np.concatenate(
    #         [stats.norm(loc=m, scale=np.sqrt(v)).rvs(batch_size//2).astype(np.float32)
    #          for m, v in zip(means[1] + 1.0, vars[1])])
    #
    #     # Compute actual log probabilities of roots
    #     empirical_means = gq._loc_init
    #     empirical_vars = np.square(gq._scale_init) if not softplus_scale else np.square(
    #         np.log(np.exp(gq._scale_init) + 1))
    #     log_probs0 = [stats.norm(loc=m, scale=np.sqrt(v)).logpdf(data0)
    #                   for m, v in zip(empirical_means[0], empirical_vars[0])]
    #     log_probs1 = [stats.norm(loc=m, scale=np.sqrt(v)).logpdf(data1)
    #                   for m, v in zip(empirical_means[1], empirical_vars[1])]
    #
    #     # Compute actual log probabilities of mixtures
    #     mixture00_val = np.logaddexp(log_probs0[0] + np.log(1/4), log_probs0[1] + np.log(3/4))
    #     mixture01_val = np.logaddexp(log_probs0[0] + np.log(3/4), log_probs0[1] + np.log(1/4))
    #
    #     mixture10_val = np.logaddexp(log_probs1[0] + np.log(2/3), log_probs1[1] + np.log(1/3))
    #     mixture11_val = np.logaddexp(log_probs1[0] + np.log(1/3), log_probs1[1] + np.log(2/3))
    #
    #     # Compute actual log probabilities of products
    #     prod0_val = mixture00_val + mixture10_val
    #     prod1_val = mixture01_val + mixture11_val
    #
    #     # Compute the index of the max probability at the products layer
    #     prod_winner = np.argmax(np.stack([prod0_val, prod1_val], axis=-1), axis=-1)
    #
    #     # Compute the indices of the max component per mixture
    #     component_winner00 = np.argmax(
    #         np.stack([log_probs0[0] + np.log(1/4), log_probs0[1] + np.log(3/4)], axis=-1), axis=-1)
    #     component_winner01 = np.argmax(
    #         np.stack([log_probs0[0] + np.log(3/4), log_probs0[1] + np.log(1/4)], axis=-1), axis=-1)
    #     component_winner10 = np.argmax(
    #         np.stack([log_probs1[0] + np.log(2/3), log_probs1[1] + np.log(1/3)], axis=-1), axis=-1)
    #     component_winner11 = np.argmax(
    #         np.stack([log_probs1[0] + np.log(1/3), log_probs1[1] + np.log(2/3)], axis=-1), axis=-1)
    #
    #     # Initialize true counts
    #     counts_per_component = np.zeros((2, 2))
    #     sum_data_val = np.zeros((2, 2))
    #     sum_data_squared_val = np.zeros((2, 2))
    #
    #     data00 = []
    #     data01 = []
    #     data10 = []
    #     data11 = []
    #
    #     # Compute true counts
    #     counts_per_step = np.zeros((batch_size, num_vars, num_components))
    #     for i, (prod_ind, d0, d1) in enumerate(zip(prod_winner, data0, data1)):
    #         if prod_ind == 0:
    #             # mixture 00 and mixture 10
    #             counts_per_step[i, 0, component_winner00[i]] = 1
    #             counts_per_component[0, component_winner00[i]] += 1
    #             sum_data_val[0, component_winner00[i]] += data0[i]
    #             sum_data_squared_val[0, component_winner00[i]] += data0[i] * data0[i]
    #             (data00 if component_winner00[i] == 0 else data01).append(data0[i])
    #
    #             counts_per_step[i, 1, component_winner10[i]] = 1
    #             counts_per_component[1, component_winner10[i]] += 1
    #             sum_data_val[1, component_winner10[i]] += data1[i]
    #             sum_data_squared_val[1, component_winner10[i]] += data1[i] * data1[i]
    #             (data10 if component_winner10[i] == 0 else data11).append(data1[i])
    #         else:
    #             counts_per_step[i, 0, component_winner01[i]] = 1
    #             counts_per_component[0, component_winner01[i]] += 1
    #             sum_data_val[0, component_winner01[i]] += data0[i]
    #             sum_data_squared_val[0, component_winner01[i]] += data0[i] * data0[i]
    #             (data00 if component_winner01[i] == 0 else data01).append(data0[i])
    #
    #             counts_per_step[i, 1, component_winner11[i]] = 1
    #             counts_per_component[1, component_winner11[i]] += 1
    #             sum_data_val[1, component_winner11[i]] += data1[i]
    #             sum_data_squared_val[1, component_winner11[i]] += data1[i] * data1[i]
    #             (data10 if component_winner11[i] == 0 else data11).append(data1[i])
    #
    #     # Setup learning Ops
    #     value_inference_type = spn.InferenceType.MARGINAL
    #     init_weights = spn.initialize_weights(root)
    #     learning = spn.HardEMLearning(root, log=True, value_inference_type=value_inference_type)
    #     reset_accumulators = learning.reset_accumulators()
    #     accumulate_updates = learning.accumulate_updates()
    #     update_spn = learning.update_spn()
    #     train_likelihood = learning.value.values[root]
    #     avg_train_likelihood = tf.reduce_mean(train_likelihood)
    #
    #     # Setup feed dict and update ops
    #     fd = {gq: np.stack([data0, data1], axis=-1)}
    #     update_ops = gq._compute_hard_em_update(learning._mpe_path.counts[gq])
    #
    #     print("sess begin")
    #     with self.test_session() as sess:
    #         print("R")
    #         sess.run(tf.global_variables_initializer())
    #
    #         # Get log probabilities of Gaussian leaf
    #         print("Get log prob gauss")
    #         log_probs = sess.run(learning.value.values[gq], fd)
    #
    #         # Get log probabilities of mixtures
    #         print("Get prob mixtures")
    #         mixture00_graph, mixture01_graph, mixture10_graph, mixture11_graph = sess.run([
    #             learning.value.values[mixture00],
    #             learning.value.values[mixture01],
    #             learning.value.values[mixture10],
    #             learning.value.values[mixture11]], fd)
    #
    #         # Get log probabilities of products
    #         print("Get log prob prod")
    #         prod0_graph, prod1_graph = sess.run(
    #             [learning.value.values[prod0], learning.value.values[prod1]], fd)
    #
    #         # Get counts for graph
    #         print("Get counts")
    #         counts = sess.run(tf.reduce_sum(learning._mpe_path.counts[gq], axis=0), fd)
    #         counts_per_sample = sess.run(learning._mpe_path.counts[gq], fd)
    #
    #         accum, sum_data_graph, sum_data_squared_graph = sess.run([
    #             update_ops['accum'], update_ops['sum_data'], update_ops['sum_data_squared']], fd)
    #
    #         sess.run(init_weights)
    #         sess.run(reset_accumulators)
    #
    #         # data_per_component_op = graph.get_tensor_by_name(
    #         #     "HardEMLearning/NormalLeaf/DataPerComponent:0")
    #         # squared_data_per_component_op = graph.get_tensor_by_name(
    #         #     "HardEMLearning/NormalLeaf/SquaredDataPerComponent:0")
    #         #
    #         # update_vals, data_per_component_out, squared_data_per_component_out = sess.run(
    #         #     [accumulate_updates, data_per_component_op, squared_data_per_component_op], fd)
    #
    #         # Get likelihood before update
    #         lh_before = sess.run(avg_train_likelihood, fd)
    #         sess.run(update_spn)
    #
    #         # Get likelihood after update
    #         lh_after = sess.run(avg_train_likelihood, fd)
    #
    #         # Get variables after update
    #         total_counts_graph, scale_graph, mean_graph = sess.run([
    #             gq._total_count_variable, gq.scale_variable, gq.loc_variable])
    #
    #     self.assertAllClose(prod0_val, prod0_graph.ravel())
    #     self.assertAllClose(prod1_val, prod1_graph.ravel())
    #
    #     self.assertAllClose(log_probs[:, 0], log_probs0[0])
    #     self.assertAllClose(log_probs[:, 1], log_probs0[1])
    #     self.assertAllClose(log_probs[:, 2], log_probs1[0])
    #     self.assertAllClose(log_probs[:, 3], log_probs1[1])
    #
    #     self.assertAllClose(mixture00_val, mixture00_graph.ravel())
    #     self.assertAllClose(mixture01_val, mixture01_graph.ravel())
    #     self.assertAllClose(mixture10_val, mixture10_graph.ravel())
    #     self.assertAllClose(mixture11_val, mixture11_graph.ravel())
    #
    #     self.assertAllEqual(counts, counts_per_component.ravel())
    #     self.assertAllEqual(accum, counts_per_component)
    #
    #     self.assertAllClose(counts_per_step, counts_per_sample.reshape(
    #         (batch_size, num_vars, num_components)))
    #
    #     self.assertAllClose(sum_data_val, sum_data_graph)
    #     self.assertAllClose(sum_data_squared_val, sum_data_squared_graph)
    #
    #     self.assertAllClose(total_counts_graph, count_init + counts_per_component)
    #     self.assertTrue(np.all(np.not_equal(mean_graph, gq._loc_init)))
    #     self.assertTrue(np.all(np.not_equal(scale_graph, gq._scale_init)))
    #
    #     mean_new_vals = []
    #     variance_new_vals = []
    #     variance_left, variance_right = [], []
    #     for i, obs in enumerate([data00, data01, data10, data11]):
    #         # Note that this does not depend on accumulating anything!
    #         # It actually is copied (more-or-less) from
    #         # https://github.com/whsu/spn/blob/master/spn/normal_leaf_node.py
    #         x = np.asarray(obs).astype(np.float32)
    #         n = count_init
    #         k = len(obs)
    #         if softplus_scale:
    #             var_old = np.square(np.log(np.exp(
    #                 gq._scale_init.astype(np.float32)).ravel()[i] + 1))
    #         else:
    #             var_old = np.square(gq._scale_init.astype(np.float32)).ravel()[i]
    #         mean = (n * gq._loc_init.astype(np.float32).ravel()[i] + np.sum(obs)) / (n + k)
    #         dx = x - gq._loc_init.astype(np.float32).ravel()[i]
    #         dm = mean - gq._loc_init.astype(np.float32).ravel()[i]
    #         var = (n * var_old + dx.dot(dx)) / (n + k) - dm * dm
    #
    #         mean_new_vals.append(mean)
    #         variance_new_vals.append(var)
    #         variance_left.append((n * var_old + dx.dot(dx)) / (n + k))
    #         variance_right.append(dm * dm)
    #
    #     mean_new_vals = np.asarray(mean_new_vals).reshape((2, 2))
    #     variance_new_vals = np.asarray(variance_new_vals).reshape((2, 2))
    #
    #     def assert_non_zero_at_ij_equal(arr, i, j, truth):
    #         # Select i-th variable and j-th component
    #         arr = arr[:, i, j]
    #         self.assertAllClose(arr[arr != 0.0], truth)
    #
    #     # assert_non_zero_at_ij_equal(data_per_component_out, 0, 0, data00)
    #     # assert_non_zero_at_ij_equal(data_per_component_out, 0, 1, data01)
    #     # assert_non_zero_at_ij_equal(data_per_component_out, 1, 0, data10)
    #     # assert_non_zero_at_ij_equal(data_per_component_out, 1, 1, data11)
    #     #
    #     # assert_non_zero_at_ij_equal(squared_data_per_component_out, 0, 0, np.square(data00))
    #     # assert_non_zero_at_ij_equal(squared_data_per_component_out, 0, 1, np.square(data01))
    #     # assert_non_zero_at_ij_equal(squared_data_per_component_out, 1, 0, np.square(data10))
    #     # assert_non_zero_at_ij_equal(squared_data_per_component_out, 1, 1, np.square(data11))
    #
    #     self.assertAllClose(mean_new_vals, mean_graph)
    #     # self.assertAllClose(np.asarray(variance_left).reshape((2, 2)), var_graph_left)
    #     self.assertAllClose(variance_new_vals, np.square(scale_graph if not softplus_scale else
    #         np.log(np.exp(scale_graph) + 1)), atol=1e-5, rtol=1e-5)
    #
    #     self.assertGreater(lh_after, lh_before)

    def test_value(self):
        num_vars = 8
        data = np.stack(
            [np.random.normal(a, size=BATCH_SIZE) for a in range(num_vars)], axis=1)

        data = np.concatenate([data, np.stack(
            [np.random.normal(a, size=BATCH_SIZE) + num_vars for a in range(num_vars)], axis=1)],
                              axis=0).astype(np.float32)

        gq = spn.NormalLeaf(num_vars=num_vars, num_components=2, initialization_data=data)

        value_op = gq._compute_value()
        log_value_op = gq._compute_log_value()

        modes = np.stack([np.arange(num_vars) for _ in range(BATCH_SIZE)] +
                         [np.arange(num_vars) + num_vars for _ in range(BATCH_SIZE)], axis=0)
        val_at_mode = stats.norm.pdf(0)

        with self.test_session() as sess:
            sess.run([gq.loc_variable.initializer, gq.scale_variable.initializer])
            value_out, log_value_out = sess.run(
                [value_op, log_value_op], feed_dict={gq.feed: modes})

        value_out = value_out.reshape((BATCH_SIZE * 2, num_vars, 2))
        log_value_out = log_value_out.reshape((BATCH_SIZE * 2, num_vars, 2))

        # We'll be quite tolerant for the error, as our output is really just an empirical mean
        self.assertAllClose(
            value_out[:BATCH_SIZE, :, 0], np.ones([BATCH_SIZE, num_vars]) * val_at_mode,
            rtol=1e-2, atol=1e-2)

        self.assertAllClose(
            value_out[BATCH_SIZE:, :, 1], np.ones([BATCH_SIZE, num_vars]) * val_at_mode,
            rtol=1e-2, atol=1e-2)

        self.assertAllClose(
            np.exp(log_value_out[:BATCH_SIZE, :, 0]), np.ones([BATCH_SIZE, num_vars]) * val_at_mode,
            rtol=1e-2, atol=1e-2)

        self.assertAllClose(
            np.exp(log_value_out[BATCH_SIZE:, :, 1]), np.ones([BATCH_SIZE, num_vars]) * val_at_mode,
            rtol=1e-2, atol=1e-2)


if __name__ == '__main__':
    tf.test.main()
