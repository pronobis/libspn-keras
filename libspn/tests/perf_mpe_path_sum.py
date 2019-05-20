#!/usr/bin/env python3

import itertools
import random
import numpy as np
import tensorflow as tf
from context import libspn as spn
from collections import namedtuple
import abc

from libspn.tests.abstract_performance_profiling import AbstractPerformanceTest, \
    AbstractPerformanceUnit, PerformanceTestArgs, ConfigGenerator
from libspn.tests.perf_sum_value_varying_sizes import sums_layer_numpy_common

MPEPathPerformanceInput = namedtuple("LogMatMulInput",
                                     ["values", "indices", "num_parallel", "sum_sizes", "num_sums",
                                      "weights", "latent_indicators"])


def _repeat_elements(arr, n):
    """
    Repeats the elements int the input array, e.g.
    [1, 2, 3] -> [1, 1, 1, 2, 2, 2, 3, 3, 3]
    """
    ret = list(itertools.chain(*[list(itertools.repeat(elem, n)) for elem in arr]))
    return ret


class AbstractSumUnit(AbstractPerformanceUnit, abc.ABC):

    def true_out(self, inputs, conf):
        """ Computes the output of _compute_mpe_path with numpy """
        weights = inputs.weights
        latent_indicators = inputs.latent_indicators
        sums_sizes = inputs.sum_sizes
        values = _repeat_elements([(inputs.values[0][:, ind], None) for ind in
                                   inputs.indices], inputs.num_parallel)
        sums_sizes = _repeat_elements(sums_sizes, inputs.num_parallel)
        inputs_to_reduce, iv_mask_per_sum, weights_per_sum = sums_layer_numpy_common(
            values, latent_indicators, sums_sizes, weights)
        # Get max index for sum node
        if conf.inf_type == spn.InferenceType.MPE:
            # We don't have to think about the max individual sum outcome in this case, it will just
            # be the max of
            # the weighted inputs at the bottom layer
            weighted_sums = [x * np.reshape(w / np.sum(w), (1, -1)) * iv
                             for x, w, iv in
                             zip(inputs_to_reduce, weights_per_sum, iv_mask_per_sum)]
            weighted_sums_concat = np.concatenate(weighted_sums, axis=1)
            max_indices = np.argmax(weighted_sums_concat, axis=1)

            # Compute the concatenated counts
            out = np.zeros_like(weighted_sums_concat)
            out[np.arange(out.shape[0]), max_indices] = np.ones(out.shape[0])

            # Split to obtain counts per sum
            splits = np.cumsum(sums_sizes)[:-1]
            return np.split(out, splits, axis=1)
        else:
            # In this case, we first have to consider the sum with the largest outcome
            weighted_sums = [x * np.reshape(w / np.sum(w), (1, -1)) * iv
                             for x, w, iv in
                             zip(inputs_to_reduce, weights_per_sum, iv_mask_per_sum)]
            sum_outcomes = np.concatenate([np.sum(s, axis=1, keepdims=True) for s in weighted_sums],
                                          axis=1)

            sum_indices = np.argmax(sum_outcomes, axis=1)

            # Now we compute the max per 'winning' sum
            out = [np.zeros_like(w) for w in weighted_sums]
            for i, ind in enumerate(sum_indices):
                max_ind = np.argmax(weighted_sums[ind][i])
                out[ind][i, max_ind] = 1

            return out


class SumsLayerUnit(AbstractSumUnit):

    def __init__(self, name, dtype, sumslayer_count_strategy):
        super(SumsLayerUnit, self).__init__(name, dtype)
        self.sum_count_strategy = sumslayer_count_strategy

    def _build_placeholders(self, inputs):
        total_inputs = sum([inp.shape[1] for inp in inputs.values])
        return [spn.RawLeaf(num_vars=total_inputs)]

    def _build_op(self, inputs, placeholders, conf):
        # TODO make sure the latent_indicators are correct
        sum_indices, weights, latent_indicators = inputs.indices, inputs.weights, None
        log, inf_type = conf.log, conf.inf_type
        repeated_inputs = []
        repeated_sum_sizes = []
        offset = 0
        for ind in sum_indices:
            # Indices are given by looking at the sizes of the sums
            size = len(ind)
            repeated_inputs.extend([(placeholders[0], ind) for _ in range(inputs.num_parallel)])
            repeated_sum_sizes.extend([size for _ in range(inputs.num_parallel)])
            offset += size

        # Globally configure to add up the sums before passing on the values to children
        spn.conf.sumslayer_count_sum_strategy = self.sum_count_strategy
        sums_layer = spn.SumsLayer(*repeated_inputs, num_or_size_sums=repeated_sum_sizes)
        weight_node = self._generate_weights(sums_layer, weights)
        if latent_indicators:
            sums_layer.set_latent_indicators(*latent_indicators)
        # Connect a single sum to group outcomes
        root = spn.Sum(sums_layer)

        self._generate_weights(root)
        # Then build MPE path Ops
        mpe_path_gen = spn.MPEPath(value_inference_type=inf_type, log=log)
        mpe_path_gen.get_mpe_path(root)
        path_op = [tf.tuple([mpe_path_gen.counts[weight_node],
                             mpe_path_gen.counts[placeholders[0]]])[0]]

        return path_op, self._initialize_from(root)

    def feed_dict(self, inputs):
        return {self._placeholders[0]: np.concatenate(inputs.values, axis=1)}

    def true_out(self, inputs, conf):
        true_out = super(SumsLayerUnit, self).true_out(inputs, conf)
        max_size = max(inputs.sum_sizes)
        padded = [np.concatenate(
            (o, np.zeros((o.shape[0], max_size - o.shape[1]))), axis=1) for o in true_out]
        true_out = [np.concatenate(padded, axis=1)
                    .reshape((-1, len(inputs.sum_sizes) * inputs.num_parallel, max_size))]
        return true_out


class ParallelSumsUnit(AbstractSumUnit):

    def __init__(self, name, dtype):
        super(ParallelSumsUnit, self).__init__(name, dtype)

    def _build_placeholders(self, inputs):
        return [spn.RawLeaf(num_vars=inputs.values[0].shape[1])]

    def _build_op(self, inputs, placeholders, conf):
        """ Creates the graph using only ParSum nodes """
        # TODO make sure the latent_indicators are correct
        sum_indices, weights, latent_indicators = inputs.indices, inputs.weights, None
        log, inf_type = conf.log, conf.inf_type
        weights = np.split(weights, np.cumsum([len(ind) * inputs.num_parallel for ind in
                                               sum_indices])[:-1])

        parallel_sum_nodes = []
        for ind in sum_indices:
            parallel_sum_nodes.append(spn.ParallelSums((placeholders[0], ind),
                                                       num_sums=inputs.num_parallel))

        weight_nodes = [self._generate_weights(node, w.tolist()) for node, w in
                        zip(parallel_sum_nodes, weights)]
        if latent_indicators:
            [s.set_latent_indicators(iv) for s, iv in zip(parallel_sum_nodes, latent_indicators)]
        root = spn.Sum(*parallel_sum_nodes)
        self._generate_weights(root)

        mpe_path_gen = spn.MPEPath(value_inference_type=inf_type, log=log)
        mpe_path_gen.get_mpe_path(root)
        path_op = [mpe_path_gen.counts[w] for w in weight_nodes]
        input_counts = [mpe_path_gen.counts[inp] for inp in placeholders]

        return tf.tuple(path_op + input_counts)[:len(path_op)], self._initialize_from(root)

    def true_out(self, inputs, conf):
        true_out = super(ParallelSumsUnit, self).true_out(inputs, conf)
        true_out = [np.concatenate(true_out[i:i + inputs.num_parallel], axis=1)
                    .reshape((true_out[0].shape[0], inputs.num_parallel, -1))
                    for i in range(0, len(true_out), inputs.num_parallel)]
        return true_out


class MPEPathPerformanceTest(AbstractPerformanceTest):

    def __init__(self, name, performance_units, test_args, config_generator):
        super().__init__(name, performance_units, test_args, config_generator)
        self.sum_sizes = test_args.sum_sizes
        self.num_parallel = test_args.num_parallel
        self.num_sums = test_args.num_sums

    def description(self):
        return "MPEPathSumNodes"

    def generate_input(self):
        values = self.random_numpy_tensor(self._shape)
        indices = [random.sample(range(self._shape[1]), size) for size in self.sum_sizes]
        num_params = sum([size * self.num_parallel for size in self.sum_sizes])
        weights = self.random_numpy_tensor((num_params,))
        return MPEPathPerformanceInput(
            values=[values], indices=indices, num_parallel=self.num_parallel,
            num_sums=self.num_sums, sum_sizes=self.sum_sizes, weights=weights, latent_indicators=None
        )


def main():
    parser = PerformanceTestArgs()
    parser.add_argument("--sum-sizes", default=[70, 80, 90, 100], type=int, nargs='+',
                        help="The size of each sum being modeled. Will be repeated "
                             "num-repetitions times, e.g. "
                             "[1, 2, 3] --> [1, 1, 1, 2, 2, 2, 3, 3, 3]")
    parser.add_argument('--num-sums', default=100, type=int,
                        help="Num of sums modelled in a single layer")
    parser.add_argument('--num-parallel', default=10, type=int,
                        help="Num repetitions for each input and sum size")
    args = parser.parse_args()

    units = [
        SumsLayerUnit("LayerCountMatmul", tf.float32, "matmul"),
        SumsLayerUnit("LayerCountGather", tf.float32, "gather"),
        SumsLayerUnit("LayerCountSegmented", tf.float32, "segmented"),
        ParallelSumsUnit("ParallelSums", tf.float32)
    ]
    # Select the device
    gpu = [False, True]
    if args.without_cpu:
        gpu.remove(False)
    if args.without_gpu:
        gpu.remove(True)

    # Make a config generator and run the test
    config_generator = ConfigGenerator(gpu=gpu)
    performance_test = MPEPathPerformanceTest(
        name="MPEPathPerformanceSumNodes", performance_units=units, test_args=args,
        config_generator=config_generator)

    performance_test.run()


if __name__ == '__main__':
    main()
