#!/usr/bin/env python3

from context import libspn as spn
from test import TestCase
import numpy as np
import random
import tensorflow as tf


def assert_list_elements_equal(list1, list2):
    """Check if lists have the same elements."""
    for l1 in list1:
        if l1 not in list2:
            raise AssertionError("List elements differ: %s != %s" % (list1, list2))


class TestPartition(TestCase):

    @classmethod
    def setUpClass(cls):
        super(TestPartition, cls).setUpClass()
        cls.test_set = [1, 2, 3, 4, 5]
        # Partitions for num_subsets=[1; 4]
        cls.possible_partitions = [None] * len(cls.test_set)
        cls.possible_partitions[0] = [[{1, 2, 3, 4, 5}]]
        cls.possible_partitions[1] = [[{1}, {2, 3, 4, 5}],
                                      [{1, 5}, {2, 3, 4}],
                                      [{1, 2, 3, 4}, {5}],
                                      [{1, 3, 5}, {2, 4}],
                                      [{1, 4}, {2, 3, 5}],
                                      [{1, 2, 3}, {4, 5}],
                                      [{1, 3, 4}, {2, 5}],
                                      [{1, 2, 4}, {3, 5}],
                                      [{1, 4, 5}, {2, 3}],
                                      [{1, 2, 4, 5}, {3}],
                                      [{1, 3}, {2, 4, 5}],
                                      [{1, 3, 4, 5}, {2}],
                                      [{1, 2}, {3, 4, 5}],
                                      [{1, 2, 3, 5}, {4}],
                                      [{1, 2, 5}, {3, 4}]]
        cls.possible_partitions[2] = [[{1}, {2, 4}, {3, 5}],
                                      [{1, 4}, {2, 5}, {3}],
                                      [{1, 2}, {3, 4}, {5}],
                                      [{1, 3}, {2}, {4, 5}],
                                      [{1, 2, 5}, {3}, {4}],
                                      [{1, 2, 4}, {3}, {5}],
                                      [{1, 4}, {2}, {3, 5}],
                                      [{1, 2, 3}, {4}, {5}],
                                      [{1, 2}, {3, 5}, {4}],
                                      [{1, 5}, {2, 4}, {3}],
                                      [{1, 3, 4}, {2}, {5}],
                                      [{1}, {2, 5}, {3, 4}],
                                      [{1}, {2}, {3, 4, 5}],
                                      [{1, 4}, {2, 3}, {5}],
                                      [{1, 3}, {2, 4}, {5}],
                                      [{1, 5}, {2, 3}, {4}],
                                      [{1}, {2, 4, 5}, {3}],
                                      [{1, 4, 5}, {2}, {3}],
                                      [{1}, {2, 3}, {4, 5}],
                                      [{1}, {2, 3, 5}, {4}],
                                      [{1, 3, 5}, {2}, {4}],
                                      [{1, 3}, {2, 5}, {4}],
                                      [{1}, {2, 3, 4}, {5}],
                                      [{1, 2}, {3}, {4, 5}],
                                      [{1, 5}, {2}, {3, 4}]]
        cls.possible_partitions[3] = [[{1, 2}, {3}, {4}, {5}],
                                      [{1}, {2}, {3, 5}, {4}],
                                      [{1, 5}, {2}, {3}, {4}],
                                      [{1}, {2, 4}, {3}, {5}],
                                      [{1}, {2}, {3}, {4, 5}],
                                      [{1}, {2}, {3, 4}, {5}],
                                      [{1}, {2, 3}, {4}, {5}],
                                      [{1, 4}, {2}, {3}, {5}],
                                      [{1, 3}, {2}, {4}, {5}],
                                      [{1}, {2, 5}, {3}, {4}]]
        cls.possible_partitions[4] = [[{1}, {2}, {3}, {4}, {5}]]
        # Balanced partitions for num_subsets=[1; 4]
        cls.possible_balanced_partitions = [None] * len(cls.test_set)
        cls.possible_balanced_partitions[0] = [[{1, 2, 3, 4, 5}]]
        cls.possible_balanced_partitions[1] = [[{1, 5}, {2, 3, 4}],
                                               [{1, 3, 5}, {2, 4}],
                                               [{1, 4}, {2, 3, 5}],
                                               [{1, 2, 3}, {4, 5}],
                                               [{1, 3, 4}, {2, 5}],
                                               [{1, 2, 4}, {3, 5}],
                                               [{1, 4, 5}, {2, 3}],
                                               [{1, 3}, {2, 4, 5}],
                                               [{1, 2}, {3, 4, 5}],
                                               [{1, 2, 5}, {3, 4}]]
        cls.possible_balanced_partitions[2] = [[{1}, {2, 4}, {3, 5}],
                                               [{1, 4}, {2, 5}, {3}],
                                               [{1, 2}, {3, 4}, {5}],
                                               [{1, 3}, {2}, {4, 5}],
                                               [{1, 4}, {2}, {3, 5}],
                                               [{1, 2}, {3, 5}, {4}],
                                               [{1, 5}, {2, 4}, {3}],
                                               [{1}, {2, 5}, {3, 4}],
                                               [{1, 4}, {2, 3}, {5}],
                                               [{1, 3}, {2, 4}, {5}],
                                               [{1, 5}, {2, 3}, {4}],
                                               [{1}, {2, 3}, {4, 5}],
                                               [{1, 3}, {2, 5}, {4}],
                                               [{1, 2}, {3}, {4, 5}],
                                               [{1, 5}, {2}, {3, 4}]]
        cls.possible_balanced_partitions[3] = [[{1, 2}, {3}, {4}, {5}],
                                               [{1}, {2}, {3, 5}, {4}],
                                               [{1, 5}, {2}, {3}, {4}],
                                               [{1}, {2, 4}, {3}, {5}],
                                               [{1}, {2}, {3}, {4, 5}],
                                               [{1}, {2}, {3, 4}, {5}],
                                               [{1}, {2, 3}, {4}, {5}],
                                               [{1, 4}, {2}, {3}, {5}],
                                               [{1, 3}, {2}, {4}, {5}],
                                               [{1}, {2, 5}, {3}, {4}]]
        cls.possible_balanced_partitions[4] = [[{1}, {2}, {3}, {4}, {5}]]

    def test_stirling_number_args(self):
        """Argument verification of StirlingRatio."""
        s = spn.utils.StirlingNumber()
        with self.assertRaises(ValueError):
            s[0, 0]
        with self.assertRaises(ValueError):
            s[2, 0]
        with self.assertRaises(ValueError):
            s[1, 2]
        with self.assertRaises(IndexError):
            s[1]
        with self.assertRaises(IndexError):
            s[1, 1, 1]

    def test_stirling_number_allocation(self):
        """Test memory allocation of StirlingNumber cache."""
        s = spn.utils.StirlingNumber()
        self.assertListEqual(list(s._StirlingNumber__numbers.shape), [100, 100])
        s[100, 100]
        self.assertListEqual(list(s._StirlingNumber__numbers.shape), [100, 100])
        s[101, 100]
        self.assertListEqual(list(s._StirlingNumber__numbers.shape), [200, 100])
        s[101, 101]
        self.assertListEqual(list(s._StirlingNumber__numbers.shape), [200, 200])
        s[601, 51]
        self.assertListEqual(list(s._StirlingNumber__numbers.shape), [601, 200])
        s[901, 401]
        self.assertListEqual(list(s._StirlingNumber__numbers.shape), [1202, 401])

    def test_stirling_number(self):
        """Calculation of Stirling number."""
        def test(s, n, k, true_array):
            s[n, k]
            s_array = np.tril(s._StirlingNumber__numbers[:n, :k])
            np.testing.assert_array_equal(s_array, np.array(true_array))
        # Initialize
        s = spn.utils.StirlingNumber()
        # Run tests
        test(s, 1, 1, [[1]])
        test(s, 5, 3, [[1, 0, 0],
                       [1, 1, 0],
                       [1, 3, 1],
                       [1, 7, 6],
                       [1, 15, 25]])
        test(s, 5, 5, [[1, 0, 0, 0, 0],
                       [1, 1, 0, 0, 0],
                       [1, 3, 1, 0, 0],
                       [1, 7, 6, 1, 0],
                       [1, 15, 25, 10, 1]])
        test(s, 10, 8, [[1, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 0, 0, 0, 0, 0, 0],
                        [1, 3, 1, 0, 0, 0, 0, 0],
                        [1, 7, 6, 1, 0, 0, 0, 0],
                        [1, 15, 25, 10, 1, 0, 0, 0],
                        [1, 31, 90, 65, 15, 1, 0, 0],
                        [1, 63, 301, 350, 140, 21, 1, 0],
                        [1, 127, 966, 1701, 1050, 266, 28, 1],
                        [1, 255, 3025, 7770, 6951, 2646, 462, 36],
                        [1, 511, 9330, 34105, 42525, 22827, 5880, 750]])
        test(s, 10, 10, [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                         [1, 3, 1, 0, 0, 0, 0, 0, 0, 0],
                         [1, 7, 6, 1, 0, 0, 0, 0, 0, 0],
                         [1, 15, 25, 10, 1, 0, 0, 0, 0, 0],
                         [1, 31, 90, 65, 15, 1, 0, 0, 0, 0],
                         [1, 63, 301, 350, 140, 21, 1, 0, 0, 0],
                         [1, 127, 966, 1701, 1050, 266, 28, 1, 0, 0],
                         [1, 255, 3025, 7770, 6951, 2646, 462, 36, 1, 0],
                         [1, 511, 9330, 34105, 42525, 22827, 5880, 750, 45, 1]])
        # Test if indexing works as it should
        self.assertEqual(s[9, 4], 7770)
        self.assertEqual(s[26, 8], 5749622251945664950)
        # Test overflow detection
        # Max 64 bit int is:       9223372036854775807
        # s[26, 9] is larger than that
        self.assertEqual(s[26, 9], -1)

    def test_stirling_ratio_args(self):
        """Argument verification of StirlingRatio."""
        r = spn.utils.StirlingRatio()
        with self.assertRaises(ValueError):
            r[0, 0]
        with self.assertRaises(ValueError):
            r[2, 0]
        with self.assertRaises(ValueError):
            r[1, 2]
        with self.assertRaises(IndexError):
            r[1]
        with self.assertRaises(IndexError):
            r[1, 1, 1]

    def test_stirling_ratio_allocation(self):
        """Test memory allocation of StirlingRatio cache."""
        r = spn.utils.StirlingRatio()
        self.assertListEqual(list(r._StirlingRatio__ratios.shape), [100, 100])
        r[100, 100]
        self.assertListEqual(list(r._StirlingRatio__ratios.shape), [100, 100])
        r[101, 100]
        self.assertListEqual(list(r._StirlingRatio__ratios.shape), [200, 100])
        r[101, 101]
        self.assertListEqual(list(r._StirlingRatio__ratios.shape), [200, 200])
        r[601, 51]
        self.assertListEqual(list(r._StirlingRatio__ratios.shape), [601, 200])
        r[901, 401]
        self.assertListEqual(list(r._StirlingRatio__ratios.shape), [1202, 401])

    def test_stirling_ratio(self):
        """Calculation of Stirling ratio."""
        def test(r, s, n, k, size_n, size_k):
            """Compare ratio calculated explicitly with StirlingNumber
               with the ratio from StirlingRatio."""
            self.assertAlmostEqual(s[n + 1, k] / s[n, k], r[n, k])
            r1 = (np.tril(s._StirlingNumber__numbers[1:size_n + 1, :size_k]) /
                  (np.tril(s._StirlingNumber__numbers[:size_n, :size_k]) +
                   (1 - np.tri(size_n, size_k))))
            r2 = np.tril(r._StirlingRatio__ratios[:size_n, :size_k])
            np.testing.assert_array_almost_equal(r1, r2)
        # Initialize
        r = spn.utils.StirlingRatio()
        s = spn.utils.StirlingNumber()
        # Test for growing values of n, k
        test(r, s, 1, 1, 1, 1)
        test(r, s, 4, 1, 4, 1)
        test(r, s, 4, 3, 4, 3)
        test(r, s, 4, 4, 4, 4)
        test(r, s, 10, 3, 10, 4)
        test(r, s, 10, 6, 10, 6)
        test(r, s, 10, 10, 10, 10)
        test(r, s, 14, 5, 14, 10)
        test(r, s, 12, 12, 14, 12)
        test(r, s, 14, 14, 14, 14)
        # Test if indexing works as it should
        self.assertAlmostEqual(r[9, 4], 4.38931788932)

    def test_random_partition_args(self):
        """Argument verification of random_partition."""
        # input_set
        with self.assertRaises(TypeError):
            spn.utils.random_partition(1, 1)
        with self.assertRaises(ValueError):
            spn.utils.random_partition([], 1)
        # num_subsets
        with self.assertRaises(ValueError):
            spn.utils.random_partition([1], 0)
        with self.assertRaises(ValueError):
            spn.utils.random_partition([1], 2)
        # stirling
        with self.assertRaises(TypeError):
            spn.utils.random_partition([1], 1, stirling=list())
        # rnd
        with self.assertRaises(TypeError):
            spn.utils.random_partition([1], 1, rnd=list())

    def test_random_partition(self):
        """Test sampling random partitions."""
        stirling = spn.utils.Stirling()
        for num_subsets in range(1, len(TestPartition.test_set) + 1):
            # Run test for various num_subsets
            with self.subTest(num_subsets=num_subsets):
                possible_partitions = TestPartition.possible_partitions[num_subsets - 1]
                counts = [0 for p in possible_partitions]
                # Sample many times
                num_tests = 10000
                for _ in range(num_tests):
                    out = spn.utils.random_partition(TestPartition.test_set,
                                                     num_subsets, stirling)
                    i = possible_partitions.index(out)
                    counts[i] += 1
                # Check if counts are uniform
                expected = num_tests / len(possible_partitions)
                for c in counts:
                    self.assertGreater(c, 0.8 * expected)
                    self.assertLess(c, 1.2 * expected)

    def test_random_partition_customrnd(self):
        """Test sampling random partitions."""
        stirling = spn.utils.Stirling()
        for num_subsets in range(1, len(TestPartition.test_set) + 1):
            # Run test for various num_subsets
            with self.subTest(num_subsets=num_subsets):
                possible_partitions = TestPartition.possible_partitions[num_subsets - 1]
                # TEST 1 - Initialize seed with 100
                rnd = random.Random(100)
                counts1 = [0 for p in possible_partitions]
                # Sample many times
                num_tests = 10000
                for _ in range(num_tests):
                    out = spn.utils.random_partition(TestPartition.test_set,
                                                     num_subsets, stirling,
                                                     rnd)
                    i = possible_partitions.index(out)
                    counts1[i] += 1
                # Check if counts are uniform
                expected = num_tests / len(possible_partitions)
                for c in counts1:
                    self.assertGreater(c, 0.8 * expected)
                    self.assertLess(c, 1.2 * expected)
                # TEST 2 - Initialize seed with 100
                rnd = random.Random(100)  # Use seed 100
                counts2 = [0 for p in possible_partitions]
                # Sample many times
                num_tests = 10000
                for _ in range(num_tests):
                    out = spn.utils.random_partition(TestPartition.test_set,
                                                     num_subsets, stirling,
                                                     rnd)
                    i = possible_partitions.index(out)
                    counts2[i] += 1
                # Check if counts are uniform
                expected = num_tests / len(possible_partitions)
                for c in counts2:
                    self.assertGreater(c, 0.8 * expected)
                    self.assertLess(c, 1.2 * expected)
                # COMPARE IF COUNTS ARE IDENTICAL
                self.assertListEqual(counts1, counts2)

    def test_all_partitions_args(self):
        """Argument verification of all_partitions."""
        # input_set
        with self.assertRaises(TypeError):
            spn.utils.all_partitions(1, 1)
        with self.assertRaises(ValueError):
            spn.utils.all_partitions([], 1)
        # num_subsets
        with self.assertRaises(ValueError):
            spn.utils.all_partitions([1], 0)
        with self.assertRaises(ValueError):
            spn.utils.all_partitions([1], 2)

    def test_all_partitions(self):
        """Test generation of all partitions of a set."""
        for num_subsets in range(1, len(TestPartition.test_set) + 1):
            # Run test for various num_subsets
            with self.subTest(num_subsets=num_subsets):
                possible_partitions = TestPartition.possible_partitions[num_subsets - 1]
                out = spn.utils.all_partitions(TestPartition.test_set,
                                               num_subsets)
                # Note, we cannot test the below by converting them to sets
                # since the elements are not hashable.
                assert_list_elements_equal(possible_partitions, out)

    def run_test_random_partitions(self, fun, balanced):
        """Generic test for sampling a subset of random partitions."""
        def sample_many(rnd=None):
            """Sample many times."""
            counts = [0 for p in possible_partitions]
            num_tests = 2000
            # Sample partitions many times
            for _ in range(num_tests):
                import inspect
                # Since we request `max_num_partitions` which is less than
                # all possible partitions in some cases, and more in others,
                # we test all possibilities
                if len(inspect.signature(fun).parameters) > 5:
                    out = fun(TestPartition.test_set, num_subsets,
                              max_num_partitions, balanced=balanced,
                              stirling=stirling, rnd=rnd)
                else:
                    out = fun(TestPartition.test_set, num_subsets,
                              max_num_partitions, balanced=balanced,
                              rnd=rnd)
                # Verify the sample
                self.assertEqual(len(out), num_partitions)
                # Count partitions
                for p in out:
                    i = possible_partitions.index(p)
                    counts[i] += 1
            # Check if counts are uniform
            expected = (num_tests * num_partitions) / len(possible_partitions)
            for c in counts:
                self.assertGreater(c, 0.8 * expected)
                self.assertLess(c, 1.2 * expected)
            return counts

        max_num_partitions = 3
        stirling = spn.utils.Stirling()
        for num_subsets in range(1, len(TestPartition.test_set) + 1):
            # Run test for various num_subsets
            with self.subTest(num_subsets=num_subsets):
                num_partitions = min(stirling.number[len(TestPartition.test_set),
                                                     num_subsets],
                                     max_num_partitions)
                if balanced:
                    possible_partitions = TestPartition.possible_balanced_partitions[
                        num_subsets - 1]
                else:
                    possible_partitions = TestPartition.possible_partitions[num_subsets - 1]
                # Test with rnd = None
                sample_many(None)
                # Test with custom rnd
                c1 = sample_many(random.Random(100))
                c2 = sample_many(random.Random(100))
                self.assertListEqual(c1, c2)

    def test_random_partitions_by_sampling_args(self):
        """Argument verification of random_partitions_by_sampling."""
        # input_set
        with self.assertRaises(TypeError):
            spn.utils.random_partitions_by_sampling(1, 1, 1)
        with self.assertRaises(ValueError):
            spn.utils.random_partitions_by_sampling([], 1, 1)
        # stirling
        with self.assertRaises(TypeError):
            spn.utils.random_partitions_by_sampling([1], 1, 1, True, stirling=list())
        # rnd
        with self.assertRaises(TypeError):
            spn.utils.random_partitions_by_sampling([1], 1, 1, True, rnd=list())
        # num_partitions
        with self.assertRaises(ValueError):
            spn.utils.random_partitions_by_sampling([1], 1, 0)
        with self.assertRaises(ValueError):
            spn.utils.random_partitions_by_sampling([1], 1, np.iinfo(int).max + 1)
        # num_subsets
        with self.assertRaises(ValueError):
            spn.utils.random_partitions_by_sampling([1], 0, 1)
        with self.assertRaises(ValueError):
            spn.utils.random_partitions_by_sampling([1], 2, 1)

    def test_random_partitions_by_sampling(self):
        """Test sampling a subset of random partitions by repeated sampling."""
        self.run_test_random_partitions(spn.utils.random_partitions_by_sampling,
                                        balanced=False)
        self.run_test_random_partitions(spn.utils.random_partitions_by_sampling,
                                        balanced=True)

    def test_random_partitions_by_enumeration_args(self):
        """Argument verification of random_partitions_by_enumeration."""
        # input_set
        with self.assertRaises(TypeError):
            spn.utils.random_partitions_by_enumeration(1, 1, 1)
        with self.assertRaises(ValueError):
            spn.utils.random_partitions_by_enumeration([], 1, 1)
        # num_partitions
        with self.assertRaises(ValueError):
            spn.utils.random_partitions_by_enumeration([1], 1, 0)
        with self.assertRaises(ValueError):
            spn.utils.random_partitions_by_enumeration([1], 1,
                                                       np.iinfo(int).max + 1)
        # num_subsets
        with self.assertRaises(ValueError):
            spn.utils.random_partitions_by_enumeration([1], 0, 1)
        with self.assertRaises(ValueError):
            spn.utils.random_partitions_by_enumeration([1], 2, 1)
        # rnd
        with self.assertRaises(TypeError):
            spn.utils.random_partitions([1], 1, 1, True, list())

    def test_random_partitions_by_enumeration(self):
        """Test sampling a subset of random partitions by enumeration."""
        self.run_test_random_partitions(spn.utils.random_partitions_by_enumeration,
                                        balanced=False)
        self.run_test_random_partitions(spn.utils.random_partitions_by_enumeration,
                                        balanced=True)

    def test_random_partitions_args(self):
        """Argument verification of random_partitions."""
        # input_set
        with self.assertRaises(TypeError):
            spn.utils.random_partitions(1, 1, 1)
        with self.assertRaises(ValueError):
            spn.utils.random_partitions([], 1, 1)
        # stirling
        with self.assertRaises(TypeError):
            spn.utils.random_partitions([1], 1, 1, True, stirling=list())
        # rnd
        with self.assertRaises(TypeError):
            spn.utils.random_partitions([1], 1, 1, True, rnd=list())
        # num_partitions
        with self.assertRaises(ValueError):
            spn.utils.random_partitions([1], 1, 0)
        with self.assertRaises(ValueError):
            spn.utils.random_partitions([1], 1, np.iinfo(int).max + 1)
        # num_subsets
        with self.assertRaises(ValueError):
            spn.utils.random_partitions([1], 0, 1)
        with self.assertRaises(ValueError):
            spn.utils.random_partitions([1], 2, 1)


if __name__ == '__main__':
    tf.test.main()
