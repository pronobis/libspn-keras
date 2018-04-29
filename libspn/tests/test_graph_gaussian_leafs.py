#!/usr/bin/env python3

# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

from context import libspn as spn
from test import TestCase
import tensorflow as tf
import numpy as np
import scipy.stats as stats

# Batch size is pretty large to obtain good approximations
BATCH_SIZE = int(1e5)


class TestGaussianQuantile(TestCase):

    def test_values_per_quantile(self):
        quantiles = [np.random.rand(32, 32) + i * 2 for i in range(4)]
        data = np.concatenate(quantiles, axis=0)
        np.random.shuffle(data)
        gq = spn.GaussianLeaf(num_vars=32, num_components=4, learn_dist_params=False)

        values_per_quantile = gq._values_per_quantile(data)

        for val, q in zip(values_per_quantile, quantiles):
            self.assertAllClose(np.sort(q, axis=0), val)

    def test_compute_scope(self):
        gl = spn.GaussianLeaf(num_vars=32, num_components=4)
        scope = gl._compute_scope()
        for b in range(0, len(scope), 4):
            [self.assertEqual(scope[b], scope[b+i]) for i in range(1, 4)]
            [self.assertNotEqual(scope[b], scope[b + i]) for i in range(4, len(scope) - b, 4)]

    def test_learn_from_data(self):
        quantiles = [np.random.rand(32, 32) + i * 2 for i in range(4)]
        data = np.concatenate(quantiles, axis=0)
        np.random.shuffle(data)
        gq = spn.GaussianLeaf(
            num_vars=32, num_components=4, learn_dist_params=False, data=data)
        true_vars = np.stack([np.var(q, axis=0) for q in quantiles], axis=-1)
        true_means = np.stack([np.mean(q, axis=0) for q in quantiles], axis=-1)

        self.assertAllClose(gq._variance_init, true_vars)
        self.assertAllClose(gq._mean_init, true_means)

    def test_learn_from_data_prior(self):
        prior_beta = 3.0
        prior_alpha = 2.0
        N = 32
        quantiles = [np.random.rand(N, 32) + i * 2 for i in range(4)]
        data = np.concatenate(quantiles, axis=0)
        np.random.shuffle(data)
        gq = spn.GaussianLeaf(
            num_vars=32, num_components=4, learn_dist_params=False, data=data,
            prior_alpha=prior_alpha, prior_beta=prior_beta, use_prior=True)

        mus = [np.mean(q, axis=0, keepdims=True) for q in quantiles]
        ssq = np.stack([np.sum((x - mu) ** 2, axis=0) for x, mu in zip(quantiles, mus)], axis=-1)
        true_vars = (2 * prior_beta + ssq) / (2 * prior_alpha + 2 + N)

        self.assertAllClose(gq._variance_init, true_vars)



    def test_sum_update_1(self):
        child1 = spn.GaussianLeaf(num_vars=1, num_components=1, total_counts_init=3,
                                  mean_init=0.0, variance_init=1.0, learn_dist_params=True)
        child2 = spn.GaussianLeaf(num_vars=1, num_components=1, total_counts_init=7,
                                  mean_init=1.0, variance_init=4.0, learn_dist_params=True)
        root = spn.Sum(child1, child2)
        root.generate_weights()

        value_inference_type = spn.InferenceType.MARGINAL
        init_weights = spn.initialize_weights(root)
        learning = spn.EMLearning(root, log=True, value_inference_type=value_inference_type,
                                  use_unweighted=True)
        reset_accumulators = learning.reset_accumulators()
        accumulate_updates = learning.accumulate_updates()
        update_spn = learning.update_spn()
        train_likelihood = learning.value.values[root]

        with self.test_session() as sess:
            sess.run(init_weights)
            sess.run(reset_accumulators)
            sess.run(accumulate_updates, {child1: [[0.0]], child2: [[0.0]]})
            sess.run(update_spn)

            child1_n = sess.run(child1._total_count_variable)
            child2_n = sess.run(child2._total_count_variable)

        # equalWeight is true, so update passes the data point to the component
        # with highest likelihood without considering the weight of each component.
        # In this case, N(0|0,1) > N(0|1,4), so child1 is picked.
        # If component weights are taken into account, then child2 will be picked
        # since 0.3*N(0|0,1) < 0.7*N(0|1,4).
        # self.assertEqual(root.n, 11)
        self.assertEqual(child1_n, 4)
        self.assertEqual(child2_n, 7)

    def test_param_learning(self):
        num_vars = 2
        num_components = 2
        batch_size = 32
        count_init = 100

        # Shape is
        means = np.array([[0, 1],
                          [10, 15]])
        vars = np.array([[0.25, 0.5],
                         [0.33, 0.67]])
        data0 = [stats.norm(loc=m, scale=np.sqrt(v)).rvs(batch_size//2).astype(np.float32)
                 for m, v in zip(means[0], vars[0])]
        data1 = [stats.norm(loc=m, scale=np.sqrt(v)).rvs(batch_size//2).astype(np.float32)
                 for m, v in zip(means[1], vars[1])]
        data = np.stack([np.concatenate(data0), np.concatenate(data1)], axis=-1)

        gq = spn.GaussianLeaf(num_vars=num_vars, num_components=num_components, data=data,
                              total_counts_init=count_init, learn_dist_params=True)

        self.assertAllClose(means, gq._mean_init, rtol=3e-1, atol=3e-1)
        self.assertAllClose(vars, gq._variance_init, rtol=3e-1, atol=3e-1)

        mixture00 = spn.Sum((gq, [0, 1]), name="Mixture00")
        weights00 = spn.Weights(init_value=[0.25, 0.75], num_weights=2)
        mixture00.set_weights(weights00)
        mixture01 = spn.Sum((gq, [0, 1]), name="Mixture01")
        weights01 = spn.Weights(init_value=[0.75, 0.25], num_weights=2)
        mixture01.set_weights(weights01)

        mixture10 = spn.Sum((gq, [2, 3]), name="Mixture10")
        weights10 = spn.Weights(init_value=[2/3, 1/3], num_weights=2)
        mixture10.set_weights(weights10)
        mixture11 = spn.Sum((gq, [2, 3]), name="Mixture11")
        weights11 = spn.Weights(init_value=[1/3, 2/3], num_weights=2)
        mixture11.set_weights(weights11)

        prod0 = spn.Product(mixture00, mixture10, name="Prod0")
        prod1 = spn.Product(mixture01, mixture11, name="Prod1")

        root = spn.Sum(prod0, prod1, name="Root")
        root_weights = spn.Weights(init_value=[1/2, 1/2], num_weights=2)
        root.set_weights(root_weights)
        # root.generate_weights()

        data0 = np.concatenate(
            [stats.norm(loc=m, scale=np.sqrt(v)).rvs(batch_size//2).astype(np.float32)
             for m, v in zip(means[0] + 0.2, vars[0])])
        data1 = np.concatenate(
            [stats.norm(loc=m, scale=np.sqrt(v)).rvs(batch_size//2).astype(np.float32)
             for m, v in zip(means[1] + 1.0, vars[1])])

        empirical_means = gq._mean_init
        empirical_vars = gq._variance_init
        log_probs0 = [stats.norm(loc=m, scale=np.maximum(np.sqrt(v), gq._min_stddev)).logpdf(data0)
                      for m, v in zip(empirical_means[0], empirical_vars[0])]
        log_probs1 = [stats.norm(loc=m, scale=np.maximum(np.sqrt(v), gq._min_stddev)).logpdf(data1)
                      for m, v in zip(empirical_means[1], empirical_vars[1])]

        mixture00_val = np.logaddexp(log_probs0[0] + np.log(1/4), log_probs0[1] + np.log(3/4))
        mixture01_val = np.logaddexp(log_probs0[0] + np.log(3/4), log_probs0[1] + np.log(1/4))

        mixture10_val = np.logaddexp(log_probs1[0] + np.log(2/3), log_probs1[1] + np.log(1/3))
        mixture11_val = np.logaddexp(log_probs1[0] + np.log(1/3), log_probs1[1] + np.log(2/3))

        prod0_val = mixture00_val + mixture10_val
        prod1_val = mixture01_val + mixture11_val

        prod_winner = np.argmax(np.stack([prod0_val, prod1_val], axis=-1), axis=-1)

        component_winner00 = np.argmax(
            np.stack([log_probs0[0] + np.log(1/4), log_probs0[1] + np.log(3/4)], axis=-1), axis=-1)
        component_winner01 = np.argmax(
            np.stack([log_probs0[0] + np.log(3/4), log_probs0[1] + np.log(1/4)], axis=-1), axis=-1)
        component_winner10 = np.argmax(
            np.stack([log_probs1[0] + np.log(2/3), log_probs1[1] + np.log(1/3)], axis=-1), axis=-1)
        component_winner11 = np.argmax(
            np.stack([log_probs1[0] + np.log(1/3), log_probs1[1] + np.log(2/3)], axis=-1), axis=-1)

        counts_per_component = np.zeros((2, 2))
        sum_data_val = np.zeros((2, 2))
        sum_data_squared_val = np.zeros((2, 2))

        data00 = []
        data01 = []
        data10 = []
        data11 = []

        counts_per_step = np.zeros((batch_size, num_vars, num_components))
        for i, (prod_ind, d0, d1) in enumerate(zip(prod_winner, data0, data1)):
            if prod_ind == 0:
                # mixture 00 and mixture 10
                counts_per_step[i, 0, component_winner00[i]] = 1
                counts_per_component[0, component_winner00[i]] += 1
                sum_data_val[0, component_winner00[i]] += data0[i]
                sum_data_squared_val[0, component_winner00[i]] += data0[i] * data0[i]
                (data00 if component_winner00[i] == 0 else data01).append(data0[i])

                counts_per_step[i, 1, component_winner10[i]] = 1
                counts_per_component[1, component_winner10[i]] += 1
                sum_data_val[1, component_winner10[i]] += data1[i]
                sum_data_squared_val[1, component_winner10[i]] += data1[i] * data1[i]
                (data10 if component_winner10[i] == 0 else data11).append(data1[i])
            else:
                counts_per_step[i, 0, component_winner01[i]] = 1
                counts_per_component[0, component_winner01[i]] += 1
                sum_data_val[0, component_winner01[i]] += data0[i]
                sum_data_squared_val[0, component_winner01[i]] += data0[i] * data0[i]
                (data00 if component_winner01[i] == 0 else data01).append(data0[i])

                counts_per_step[i, 1, component_winner11[i]] = 1
                counts_per_component[1, component_winner11[i]] += 1
                sum_data_val[1, component_winner11[i]] += data1[i]
                sum_data_squared_val[1, component_winner11[i]] += data1[i] * data1[i]
                (data10 if component_winner11[i] == 0 else data11).append(data1[i])

        self.assertTrue(root.is_valid)
        value_inference_type = spn.InferenceType.MARGINAL
        init_weights = spn.initialize_weights(root)
        learning = spn.EMLearning(root, log=True, value_inference_type=value_inference_type)
        reset_accumulators = learning.reset_accumulators()
        accumulate_updates = learning.accumulate_updates()
        update_spn = learning.update_spn()
        train_likelihood = learning.value.values[root]
        avg_train_likelihood = tf.reduce_mean(train_likelihood)

        fd = {gq: np.stack([data0, data1], axis=-1)}
        update_ops = gq._compute_hard_em_update(learning._mpe_path.counts[gq])

        with self.test_session() as sess:
            sess.run(init_weights)

            log_probs = sess.run(learning.value.values[gq], fd)

            mixture00_graph, mixture01_graph, mixture10_graph, mixture11_graph = sess.run([
                learning.value.values[mixture00],
                learning.value.values[mixture01],
                learning.value.values[mixture10],
                learning.value.values[mixture11]], fd)

            prod0_graph, prod1_graph = sess.run(
                [learning.value.values[prod0], learning.value.values[prod1]], fd)

            counts = sess.run(tf.reduce_sum(learning._mpe_path.counts[gq], axis=0), fd)
            counts_per_step_graph = sess.run(learning._mpe_path.counts[gq], fd)

            accum, sum_data_graph, sum_data_squared_graph = sess.run([
                update_ops['accum'], update_ops['sum_data'], update_ops['sum_data_squared']], fd)

        with self.test_session() as sess:
            sess.run(init_weights)
            sess.run(reset_accumulators)
            # all_ops = sess.graph.get_operations()
            data_per_component_op = sess.graph.get_tensor_by_name("DataPerComponent:0")
            squared_data_per_component_op = sess.graph.get_tensor_by_name(
                "SquaredDataPerComponent:0")

            update_vals, data_per_component_out, squared_data_per_component_out = sess.run(
                [accumulate_updates, data_per_component_op, squared_data_per_component_op], fd)

            lh_before = sess.run(avg_train_likelihood, fd)
            sess.run(update_spn)

            lh_after = sess.run(avg_train_likelihood, fd)
            total_counts_graph, variance_graph, mean_graph = sess.run([
                gq._total_count_variable, gq.variance_variable, gq.mean_variable])

        self.assertAllClose(counts_per_step, counts_per_step_graph.reshape(
            (batch_size, num_vars, num_components)))

        self.assertAllClose(prod0_val, prod0_graph.ravel())
        self.assertAllClose(prod1_val, prod1_graph.ravel())

        self.assertAllClose(log_probs[:, 0], log_probs0[0])
        self.assertAllClose(log_probs[:, 1], log_probs0[1])
        self.assertAllClose(log_probs[:, 2], log_probs1[0])
        self.assertAllClose(log_probs[:, 3], log_probs1[1])

        self.assertAllClose(mixture00_val, mixture00_graph.ravel())
        self.assertAllClose(mixture01_val, mixture01_graph.ravel())
        self.assertAllClose(mixture10_val, mixture10_graph.ravel())
        self.assertAllClose(mixture11_val, mixture11_graph.ravel())

        self.assertAllEqual(counts, counts_per_component.ravel())
        self.assertAllEqual(accum, counts_per_component)

        self.assertAllClose(sum_data_val, sum_data_graph)
        self.assertAllClose(sum_data_squared_val, sum_data_squared_graph)

        self.assertTrue(np.all(np.not_equal(variance_graph, gq._variance_init)))
        self.assertTrue(np.all(np.not_equal(mean_graph, gq._mean_init)))
        self.assertAllClose(total_counts_graph, count_init + counts_per_component)

        mean_new_vals = []
        variance_new_vals = []
        variance_left, variance_right = [], []
        for i, obs in enumerate([data00, data01, data10, data11]):
            # Note that this does no depend on accumulating anything!
            # It actually is copied (more or less) from
            # https://github.com/whsu/spn/blob/master/spn/normal_leaf_node.py
            x = np.asarray(obs).astype(np.float32)
            n = count_init
            k = len(obs)
            mean = (n * gq._mean_init.astype(np.float32).ravel()[i] + np.sum(obs)) / (n + k)
            dx = x - gq._mean_init.astype(np.float32).ravel()[i]
            dm = mean - gq._mean_init.astype(np.float32).ravel()[i]
            var = (n * gq._variance_init.astype(np.float32).ravel()[i] +
                   dx.dot(dx)) / (n + k) - dm * dm

            # print(mean.shape)
            # print(var.shape)
            mean_new_vals.append(mean)
            variance_new_vals.append(var)
            variance_left.append((n * gq._variance_init.ravel()[i] + dx.dot(dx)) / (n + k))
            variance_right.append(dm * dm)

        mean_new_vals = np.asarray(mean_new_vals).reshape((2, 2))
        variance_new_vals = np.asarray(variance_new_vals).reshape((2, 2))

        def assert_non_zero_at_equal(arr, i, j, truth):
            arr = arr[:, i, j]
            self.assertAllClose(arr[arr != 0.0], truth)

        assert_non_zero_at_equal(data_per_component_out, 0, 0, data00)
        assert_non_zero_at_equal(data_per_component_out, 0, 1, data01)
        assert_non_zero_at_equal(data_per_component_out, 1, 0, data10)
        assert_non_zero_at_equal(data_per_component_out, 1, 1, data11)

        assert_non_zero_at_equal(squared_data_per_component_out, 0, 0, np.square(data00))
        assert_non_zero_at_equal(squared_data_per_component_out, 0, 1, np.square(data01))
        assert_non_zero_at_equal(squared_data_per_component_out, 1, 0, np.square(data10))
        assert_non_zero_at_equal(squared_data_per_component_out, 1, 1, np.square(data11))

        self.assertAllClose(mean_new_vals, mean_graph)
        self.assertAllClose(variance_new_vals, variance_graph)

        self.assertGreater(lh_after, lh_before)

    def test_value(self):
        num_vars = 8
        data = np.stack(
            [np.random.normal(a, size=BATCH_SIZE) for a in range(num_vars)], axis=1)

        data = np.concatenate([data, np.stack(
            [np.random.normal(a, size=BATCH_SIZE) + num_vars for a in range(num_vars)], axis=1)],
                              axis=0).astype(np.float32)

        gq = spn.GaussianLeaf(num_vars=num_vars, num_components=2, learn_dist_params=False,
                              data=data)

        value_op = gq._compute_value()
        log_value_op = gq._compute_log_value()

        modes = np.stack([np.arange(num_vars) for _ in range(BATCH_SIZE)] +
                         [np.arange(num_vars) + num_vars for _ in range(BATCH_SIZE)], axis=0)
        val_at_mode = stats.norm.pdf(0)

        with self.test_session() as sess:
            sess.run([gq.mean_variable.initializer, gq.variance_variable.initializer])
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

    def test_mpe_state(self):
        num_vars = 4
        data = np.stack(
            [np.random.normal(a, size=BATCH_SIZE) for a in range(num_vars)], axis=1)

        data = np.concatenate([data, np.stack(
            [np.random.normal(a, size=BATCH_SIZE) + num_vars for a in range(num_vars)], axis=1)],
                              axis=0).astype(np.float32)

        gq = spn.GaussianLeaf(num_vars=num_vars, num_components=2, data=data,
                              learn_dist_params=False)

        batch_size = 3
        left = np.random.randint(2, size=batch_size * num_vars).reshape((-1, num_vars))
        counts = np.stack((left, 1 - left), axis=-1)

        mpe_truth = []
        for vars in left:
            for i, val in enumerate(vars):
                mpe_truth.append(i if val == 1 else i + num_vars)

        mpe_truth = np.reshape(mpe_truth, (-1, num_vars))

        mpe_state = gq._compute_mpe_state(counts)

        with self.test_session() as sess:
            sess.run([gq.initialize()])
            mpe_state_out = sess.run(mpe_state)
        # Again we must be quite tolerant, but that's ok, the predictions are 1.0 apart
        self.assertAllClose(mpe_truth, mpe_state_out, atol=1e-1, rtol=1e-1)


if __name__ == '__main__':
    tf.test.main()
