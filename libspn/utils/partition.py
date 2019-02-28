"""Functions for generating a random set of `partitions of a set
<https://en.wikipedia.org/wiki/Partition_of_a_set>`_."""

import random
import numpy as np
from libspn.log import get_logger

logger = get_logger()


class StirlingNumber:
    """Class calculating the value ``S(n,k)`` of the
    `Stirling numbers of the second kind
    <http://en.wikipedia.org/wiki/Stirling_numbers_of_the_second_kind>`_,
    where ``n`` is the number of elements in the set, and ``k`` is the number
    of subsets.

    The calculations are performed in a lazy fashion and are cashed.
    Furthermore, all values of ``S(i,j)`` for ``0<i<=n``, ``0<j<=k`` are
    pre-calculated when ``S(n,j)`` is requested. This is partially a requirement
    of the recursive calculation procedure and partially a result of the fact
    that the numbers and ratios are potentially needed for all these values
    anyways when sampling random set partitions.

    To obtain the number ``S(n,k)``, use the indexing operator ``[n, k]`` on the
    object. The returned value can be negative (``-1``), if overflow occurred
    (the number is larger than fits ``long``).
    """

    def __init__(self):
        min_cache_size = 100  # Min size of the cache to avoid initial reallocations
        # Allocate
        self.__numbers = np.empty((min_cache_size, min_cache_size), dtype=int)
        self.__cur_n = 1
        self.__cur_k = 1
        # Initialize S(1,1) = 1
        self.__numbers[0, 0] = 1

    def __ensure_size(self, n, k):
        """Make sure that the matrix is at least of size ``(n, k)``, if not
        resize. Accommodate for future expansions by resizing at least 2x.
        """
        size_n = self.__numbers.shape[0]
        size_k = self.__numbers.shape[1]
        if size_n < n or size_k < k:
            # Expand at least 2x
            if size_n < n:
                n = max(n, 2 * size_n)
            else:
                n = size_n
            if size_k < k:
                k = max(k, 2 * size_k)
            else:
                k = size_k
            # Resize and copy data
            self.__numbers.resize(n, k)
            self.__numbers[:size_n, :size_k].flat = self.__numbers.flat[:size_n * size_k]

    def __getitem__(self, nk):
        """Returns the number ``S(n,k)`` for index ``[n, k]``. The returned
        value can be negative (``-1``), if overflow occurred (the number is
        larger than fits ``long``).
        """
        if type(nk) is not tuple or len(nk) != 2:
            raise IndexError("Invalid index, must be: [n, k]")
        n = nk[0]
        k = nk[1]
        # Verify arguments
        if n < 1:
            raise ValueError("Number of set elements <1")
        if k < 1:
            raise ValueError("Number of subsets <1")
        if k > n:
            raise ValueError("Number of subsets larger than number of set elements")
        # If already computed, just return
        if n <= self.__cur_n and k <= self.__cur_k:
            return self.__numbers[n - 1, k - 1]
        # Ensure that we never "forget" values
        new_n = max(self.__cur_n, n)
        new_k = max(self.__cur_k, k)
        # Check if we have to resize the matrix
        self.__ensure_size(new_n, new_k)
        # Compute the numbers
        # - Set S(n, 1)=1
        self.__numbers[self.__cur_n:new_n, 0] = 1
        # - Set S(n,n)=1
        for i in range(self.__cur_k, new_k):
            self.__numbers[i, i] = 1
        # - Calculate remaining values
        #   S(n, k) = S(n-1, k-1) + k * S(n-1, k) for k<n
        err = np.seterr(all='raise')  # Except on overflow
        # -- first for existing n and any new k
        for j in range(self.__cur_k, new_k):
            for i in range(j + 1, self.__cur_n):
                # Detect previous overflows
                if (self.__numbers[i - 1, j - 1] >= 0 and
                        self.__numbers[i - 1, j] >= 0):
                    try:
                        self.__numbers[i, j] = (self.__numbers[i - 1, j - 1] +
                                                (j + 1) * self.__numbers[i - 1, j])
                    except FloatingPointError:  # Overflow detected
                        self.__numbers[i, j] = -1
                else:
                    self.__numbers[i, j] = -1
        # -- then for any new n
        for i in range(self.__cur_n, n):
            for j in range(1, min(i, new_k)):
                # Detect previous overflows
                if (self.__numbers[i - 1, j - 1] >= 0 and
                        self.__numbers[i - 1, j] >= 0):
                    try:
                        self.__numbers[i, j] = (self.__numbers[i - 1, j - 1] +
                                                (j + 1) * self.__numbers[i - 1, j])
                    except FloatingPointError:  # Overflow detected
                        self.__numbers[i, j] = -1
                else:
                    self.__numbers[i, j] = -1
        np.seterr(**err)  # Restore original settings
        # Update info about computed values
        self.__cur_n = new_n
        self.__cur_k = new_k
        return self.__numbers[n - 1, k - 1]


class StirlingRatio:
    """Class calculating the ratio ``R(n,k) = S(n+1,k) / S(n,k)`` of the
    `Stirling numbers of the second kind
    <http://en.wikipedia.org/wiki/Stirling_numbers_of_the_second_kind>`_,
    where ``n`` is the number of elements in the set, and ``k`` is the number
    of subsets.

    It calculates ``R(n,k) = S(n+1,k) / S(n,k)`` according to the equation
    given in "Some Properties and Applications of a Ratio of Stirling Numbers of
    the Second Kind", Sven Berg, Scandinavian Journal of Statistics, Vol. 2, No.
    2 (1975), pp. 91-94.

    The calculations are performed in a lazy fashion and are cashed.
    Furthermore, all values of ``R(i,j)`` for ``0<i<=n``, ``0<j<=k`` are
    pre-calculated when ``R(n,k)`` is requested. This is partially a requirement
    of the recursive calculation procedure and partially a result of the fact
    that the numbers and ratios are potentially needed for all these values
    anyways when sampling random set partitions.

    To obtain the ratio ``R(n,k)``, use the indexing operator ``[n, k]`` on the
    object.
    """

    def __init__(self):
        min_cache_size = 100  # Min size of the cache to avoid initial reallocations
        # Allocate
        self.__ratios = np.empty((min_cache_size, min_cache_size), dtype=float)
        self.__cur_n = 1
        self.__cur_k = 1
        # Initialize R(1,1) = 1
        self.__ratios[0, 0] = 1

    def __ensure_size(self, n, k):
        """Make sure that the matrix is at least of size ``(n, k)``, if not
        resize. Accommodate for future expansions by resizing at least 2x.
        """
        size_n = self.__ratios.shape[0]
        size_k = self.__ratios.shape[1]
        if size_n < n or size_k < k:
            # Expand at least 2x
            if size_n < n:
                n = max(n, 2 * size_n)
            else:
                n = size_n
            if size_k < k:
                k = max(k, 2 * size_k)
            else:
                k = size_k
            # Resize and copy data
            self.__ratios.resize(n, k)
            self.__ratios[:size_n, :size_k].flat = self.__ratios.flat[:size_n * size_k]

    def __getitem__(self, nk):
        """Returns the ratio ``R(n,k) = S(n+1,k) / S(n,k)`` for index ``[n, k]``.
        """
        if type(nk) is not tuple or len(nk) != 2:
            raise IndexError("Invalid index, must be: [n, k]")
        n = nk[0]
        k = nk[1]
        # Verify arguments
        if n < 1:
            raise ValueError("Number of set elements <1")
        if k < 1:
            raise ValueError("Number of subsets <1")
        if k > n:
            raise ValueError("Number of subsets larger than number of set elements")
        # If already computed, just return
        if n <= self.__cur_n and k <= self.__cur_k:
            return self.__ratios[n - 1, k - 1]
        # Ensure that we never "forget" values
        new_n = max(self.__cur_n, n)
        new_k = max(self.__cur_k, k)
        # Check if we have to resize the matrix
        self.__ensure_size(new_n, new_k)
        # Compute the ratios
        # - Set R(n, 1)=1
        self.__ratios[self.__cur_n:new_n, 0] = 1
        # - Set R(n,n)=n(n+1)/2
        for i in range(self.__cur_k, new_k):
            self.__ratios[i, i] = (i + 1) * (i + 2) / 2.0
        # - Calculate remaining values
        #   R(n, k) = k + (R(n-1, k) - k) * R(n-1, k-1) / R(n-1, k)
        #   for k<n
        # -- first for existing n and any new k
        for j in range(self.__cur_k, new_k):
            for i in range(j + 1, self.__cur_n):
                self.__ratios[i, j] = ((j + 1) +
                                       (((self.__ratios[i - 1, j] - j - 1) *
                                         self.__ratios[i - 1, j - 1]) /
                                        self.__ratios[i - 1, j]))
        # -- then for any new n
        for i in range(self.__cur_n, n):
            for j in range(1, min(i, new_k)):
                self.__ratios[i, j] = ((j + 1) +
                                       (((self.__ratios[i - 1, j] - j - 1) *
                                         self.__ratios[i - 1, j - 1]) /
                                        self.__ratios[i - 1, j]))
        # Update info about computed values
        self.__cur_n = new_n
        self.__cur_k = new_k
        return self.__ratios[n - 1, k - 1]


class Stirling:

    """A struct holding a :class:`StirlingNumber` and a :class:`StirlingRatio`.

    Attributes:
        number (StirlingNumber): Instance of :class:`StirlingNumber`.
        ratio (StirlingRatio): Instance of :class:`StirlingRatio`.
    """

    def __init__(self):
        self.number = StirlingNumber()
        self.ratio = StirlingRatio()


def random_partition(input_set: list, num_subsets: int,
                     stirling: Stirling = None,
                     rnd: random.Random = None):
    """Sample uniformly a random `partition of a set
    <https://en.wikipedia.org/wiki/Partition_of_a_set>`_.

    The algorithm follows the principle of recursive calculation of the
    `Stirling numbers of the second kind
    <http://en.wikipedia.org/wiki/Stirling_numbers_of_the_second_kind>`_. It
    observes that the only two ways of constructing a partition of ``n``
    elements into ``k`` non-empty sets is to either partition ``n-1`` elements
    into ``k-1`` sets and consider the remaining element a singleton, or
    partition ``n-1`` elements into ``k`` sets and add the remaining element to
    one of these sets. According to the Stirling number equation, the first case
    should be selected with probability ``S(n-1,k-1) / S(n,k)``, or equivalently
    ``1 - k * S(n-1,k) / S(n,k)``, which can be expressed in terms of ratio
    ``R(n,k) = S(n+1,k) / S(n,k)`` calculated by ``StirlingRatio`` class as
    ``1 - k / R(n-1,k)``. When selecting the second case, the set to which we
    place the remaining element should be chosen according to uniform
    distribution.

    Args:
        input_set (collection): A collection of elements of the input set.
        num_subsets (int): Number of subsets in each partition of the input set.
        stirling (Stirling): An instance of the :class:`Stirling` class for
                             caching/re-using computed Stirling numbers and
                             ratios. If set to ``None``, one will be created
                             internally.
        rnd (Random): Optional. A custom instance of a random number generator
                      ``random.Random`` that will be used instead of the
                      default global instance. This permits using a generator
                      with a custom state independent of the global one.

    Returns:
        list of set: A list of ``num_subsets`` subsets, where each subset is a
        set of elements of ``input_set``.
    """
    # Test args
    try:
        if type(input_set) is not list and type(input_set) is not tuple:
            input_set = list(input_set)  # For easy indexing
    except TypeError:
        raise TypeError("Input set is not iterable")
    if not input_set:
        raise ValueError("Input set is empty")
    n = len(input_set)
    if num_subsets <= 0:
        raise ValueError("Number of subsets <=0")
    if num_subsets > n:
        raise ValueError("Number of subsets larger than set elements")
    if stirling is None:
        stirling = Stirling()
    if not isinstance(stirling, Stirling):
        raise TypeError("stirling must be of type Stirling")
    if rnd is None:
        rnd = random._inst
    else:
        if not isinstance(rnd, random.Random):
            raise TypeError("rnd must be of type Random")
    # Initialize
    subsets = [set() for i in range(num_subsets)]
    DECISION_SINGLETON = 1  # Make the element a singleton
    DECISION_EXTEND = 2  # Add the element to existing subset
    decisions = [None] * n
    # Decide for each element if it should be a singleton
    for i in range(n - 1, -1, -1):
        # Check for the special case of num_elems == num_subsets
        # since the ratio cannot be calculated for such case
        if i + 1 == num_subsets:
            p_singleton = 1.0
        else:
            # n = i+1, k = num_subsets
            p_singleton = 1 - num_subsets / stirling.ratio[i, num_subsets]
        if rnd.random() < p_singleton:
            # The last element will be a singleton
            decisions[i] = DECISION_SINGLETON
            num_subsets -= 1
        else:
            # The last element will be added to one of the subsets in the
            # partition of the remaining elements.
            decisions[i] = DECISION_EXTEND
    # Assign elements to subsets
    s = 0
    for i in range(n):
        if decisions[i] == DECISION_SINGLETON:
            # Create a new subset
            subsets[s].add(input_set[i])
            s = s + 1
        else:
            # Add to one of existing subsets
            sel_subset = rnd.randrange(s)
            subsets[sel_subset].add(input_set[i])
    return subsets


def random_partitions_by_sampling(input_set: list, num_subsets: int,
                                  num_partitions: int,
                                  balanced: bool=False,
                                  stirling: Stirling=None,
                                  rnd: random.Random = None):
    """Generate a random sub-set of all `partitions of a set
    <https://en.wikipedia.org/wiki/Partition_of_a_set>`_ using repeated
    sampling.

    If the requested number of partitions ``num_partitions`` is larger than the
    number of all possible partitions, all possible partitions are returned.
    Note that if ``balanced`` is ``True``, only a subset of possible partitions
    will be returned (balanced ones). Therefore, if the ``num_partition`` is
    larger than the number of possible balanced partitions, this function
    will currently enter infinite loop!

    Args:
        input_set (collection): A collection of elements of the input set.
        num_subsets (int): Number of subsets in each partition of the input set.
        num_partitions (int): Number of partitions in the generated random
                              subset of all partitions.
        balanced (bool): If true, return only partitions consisting of subsets
                         with similar cardinality (differing by max 1).
        stirling (Stirling): An instance of the :class:`Stirling` class for
                             caching/re-using computed Stirling numbers and
                             ratios. If set to ``None``, one will be created
                             internally.
        rnd (Random): Optional. A custom instance of a random number generator
                      ``random.Random`` that will be used instead of the
                      default global instance. This permits using a generator
                      with a custom state independent of the global one.

    Returns:
        list of list of set: A list of ``num_partitions`` partitions, where each
        partition is a list of ``num_subsets`` subsets, where each subset is a
        set of elements of ``input_set``.
    """
    # Test args
    try:
        if type(input_set) is not list and type(input_set) is not tuple:
            input_set = list(input_set)  # For easy indexing
    except TypeError:
        raise TypeError("Input set is not iterable")
    if not input_set:
        raise ValueError("Input set is empty")
    n = len(input_set)
    if stirling is None:
        stirling = Stirling()
    if not isinstance(stirling, Stirling):
        raise TypeError("stirling must be of type Stirling")
    if num_partitions <= 0:
        raise ValueError("Number of partitions <=0")
    if num_partitions > np.iinfo(int).max:
        raise ValueError("More partitions requested than max int")
    num_all_partitions = stirling.number[n, num_subsets]
    if num_all_partitions >= 0 and num_partitions > num_all_partitions:
        logger.debug2("Requested %s partitions, but only %s possible for %s "
                      "elements and %s subsets, returning all %s partitions",
                      num_partitions, num_all_partitions, n, num_subsets,
                      num_all_partitions)
        num_partitions = num_all_partitions
    # Sample
    partitions = [None] * num_partitions
    p = 0
    while p < num_partitions:
        part = random_partition(input_set=input_set, num_subsets=num_subsets,
                                stirling=stirling, rnd=rnd)
        if balanced:
            sizes = [len(s) for s in part]
            max_size = max(sizes)
            if any(max_size - s > 1 for s in sizes):
                continue
        if part not in partitions[:p]:
            partitions[p] = part
            p += 1
    return partitions


def all_partitions(input_set: list, num_subsets: int):
    """Enumerate all `partitions of a set
    <https://en.wikipedia.org/wiki/Partition_of_a_set>`_ in lexicographic order.

    Args:
        input_set (collection): A collection of elements of the input set.
        num_subsets (int): Number of subsets in each partition of the input set.

    Returns:
        list of list of set: A list of all possible partitions, where each
        partition is a list of ``num_subsets`` subsets, where each subset is a
        set of elements of ``input_set``.
    """
    # Test args
    try:
        if type(input_set) is not list and type(input_set) is not tuple:
            input_set = list(input_set)  # For easy indexing
    except TypeError:
        raise TypeError("Input set is not iterable")
    if not input_set:
        raise ValueError("Input set is empty")
    s = len(input_set)
    if num_subsets <= 0:
        raise ValueError("Number of subsets <=0")
    if num_subsets > s:
        raise ValueError("Number of subsets larger than set elements")
    # Initialize
    partitions = []
    num_parts = min(num_subsets, s)  # No more parts than elements
    part_indices = [0] * s  # Part assignments
    max_vals = [-1] + [0] * (s - 1)  # mv[i] = max(pi[i-1], mv[i-1]), mv[0] = -1

    def is_incrementable(idx):
        """Check if value at idx can be incremented."""
        p = part_indices[idx]
        return ((p <= max_vals[idx]) and (p < num_parts - 1))

    def fill_right(idx):
        """Fill right of idx after incrementing. Fill with such assignments
        that each subset is not empty."""
        p = part_indices[idx]
        m = max_vals[idx]
        mx = max(m, p)
        i = num_parts - 1
        j = s - 1
        # Add missing assignments
        while i > mx:
            part_indices[j] = i
            max_vals[j] = i - 1
            i -= 1
            j -= 1
        # Fill rest with set number 0
        while j > idx:
            part_indices[j] = 0
            max_vals[j] = mx
            j -= 1

    def add_partition():
        part = [set() for i in range(num_parts)]
        for i, j in enumerate(part_indices):
            part[j].add(input_set[i])
        partitions.append(part)

    # Initialize the assignments so that each set is not empty
    fill_right(0)
    found = True

    while found:
        # Save partition
        add_partition()
        # Find first incrementable index from the right
        found = False
        for i in reversed(range(s)):
            if is_incrementable(i):
                found = True
                break
        # Increment
        part_indices[i] += 1
        # Fill right of the incremented index
        fill_right(i)

    # Return
    return partitions


def random_partitions_by_enumeration(input_set: list, num_subsets: int,
                                     num_partitions: int,
                                     balanced: bool=False,
                                     rnd: random.Random = None):
    """Generate a random sub-set of all `partitions of a set
    <https://en.wikipedia.org/wiki/Partition_of_a_set>`_ by first enumerating
    all partitions in lexicographic order.

    If the requested number of partitions ``num_partitions`` is larger than the
    number of all possible partitions, all possible partitions are returned.
    If ``balanced`` is ``True``, and  ``num_partitions`` is larger than the
    number of all balanced partitions, all balanced partitions are returned.

    Args:
        input_set (collection): A collection of elements of the input set.
        num_subsets (int): Number of subsets in each partition of the input set.
        num_partitions (int): Number of partitions in the generated random
                              subset of all partitions.
        balanced (bool): If true, return only partitions consisting of subsets
                         with similar cardinality (differing by max 1).
        rnd (Random): Optional. A custom instance of a random number generator
                      ``random.Random`` that will be used instead of the
                      default global instance. This permits using a generator
                      with a custom state independent of the global one.

    Returns:
        list of list of set: A list of ``num_partitions`` partitions, where each
        partition is a list of ``num_subsets`` subsets, where each subset is a
        set of elements of ``input_set``.
    """
    # Test args
    try:
        if type(input_set) is not list and type(input_set) is not tuple:
            input_set = list(input_set)  # For easy indexing
    except TypeError:
        raise TypeError("Input set is not iterable")
    if not input_set:
        raise ValueError("Input set is empty")
    n = len(input_set)
    if num_partitions <= 0:
        raise ValueError("Number of partitions <=0")
    if num_partitions > np.iinfo(int).max:
        raise ValueError("More partitions requested than max int")
    if rnd is None:
        rnd = random._inst
    else:
        if not isinstance(rnd, random.Random):
            raise TypeError("rnd must be of type Random")
    # Get partitions
    partitions = all_partitions(input_set, num_subsets)
    # Check number of partitions requested
    if num_partitions > len(partitions):
        logger.debug2("Requested %s partitions, but only %s possible for %s "
                      "elements and %s subsets, returning all %s partitions",
                      num_partitions, len(partitions), n, num_subsets,
                      len(partitions))
        num_partitions = len(partitions)
    # Keep only balanced partitions
    if balanced:
        def is_balanced(p):
            sizes = [len(s) for s in p]
            max_size = max(sizes)
            return all(max_size - s <= 1 for s in sizes)
        partitions = [p for p in partitions if is_balanced(p)]
        if num_partitions > len(partitions):
            logger.debug2("Requested %s partitions, but only %s possible for %s "
                          "elements and %s subsets, returning all %s partitions",
                          num_partitions, len(partitions), n, num_subsets,
                          len(partitions))
            num_partitions = len(partitions)
    # Shuffle partitions
    # WARNING: shuffle won't generate all possible permutations!
    #          See shuffle docs!
    rnd.shuffle(partitions)
    return partitions[:num_partitions]


def random_partitions(input_set: list, num_subsets: int, num_partitions: int,
                      balanced: bool=False, stirling: Stirling = None,
                      rnd: random.Random = None):
    """Generate a random sub-set of all `partitions of a set
    <https://en.wikipedia.org/wiki/Partition_of_a_set>`_ using either repeated
    sampling or enumeration of all partitions, depending on the relation of
    `num_partitions` and the number of all possible partitions. Enumeration
    is used only if the number of elements of the input set is smaller than 13,
    in which case the number of all possible partitions can reach ``1 323 652``.

    If the requested number of partitions ``num_partitions`` is larger than the
    number of all possible partitions, all possible partitions are returned.
    Note that if ``balanced`` is ``True``, only a subset of possible partitions
    will be returned (balanced ones). Therefore, if the ``num_partition`` is
    larger than the number of possible balanced partitions, this function
    might currently enter infinite loop!

    Args:
        input_set (collection): A collection of elements of the input set.
        num_subsets (int): Number of subsets in each partition of the input set.
        num_partitions (int): Number of partitions in the generated random
                              subset of all partitions.
        balanced (bool): If true, return only partitions consisting of subsets
                         with similar cardinality (differing by max 1).
        stirling (Stirling): An instance of the :class:`Stirling` class for
                             caching/re-using computed Stirling numbers and
                             ratios. If set to ``None``, one will be created
                             internally.
        rnd (Random): Optional. A custom instance of a random number generator
                      ``random.Random`` that will be used instead of the
                      default global instance. This permits using a generator
                      with a custom state independent of the global one.

    Returns:
        list of list of set: A list of ``num_partitions`` partitions, where each
        partition is a list of ``num_subsets`` subsets, where each subset is a
        set of elements of ``input_set``.
    """
    # Test args
    try:
        if type(input_set) is not list and type(input_set) is not tuple:
            input_set = list(input_set)  # For easy indexing
    except TypeError:
        raise TypeError("Input set is not iterable")
    if not input_set:
        raise ValueError("Input set is empty")
    n = len(input_set)
    if stirling is None:
        stirling = Stirling()
    if not isinstance(stirling, Stirling):
        raise TypeError("stirling must be of type Stirling")
    if num_partitions <= 0:
        raise ValueError("Number of partitions <=0")
    if num_partitions > np.iinfo(int).max:
        raise ValueError("More partitions requested than max int")
    num_all_partitions = stirling.number[n, num_subsets]
    if num_all_partitions >= 0 and num_partitions > num_all_partitions:
        logger.debug2("Requested %s partitions, but only %s possible for %s "
                      "elements and %s subsets, returning all %s partitions",
                      num_partitions, num_all_partitions, n, num_subsets,
                      num_all_partitions)
        num_partitions = num_all_partitions
    # Choose method
    if n < 13:
        if num_partitions / num_all_partitions < 0.1:
            return random_partitions_by_sampling(input_set, num_subsets,
                                                 num_partitions,
                                                 stirling=stirling,
                                                 balanced=balanced,
                                                 rnd=rnd)
        else:
            return random_partitions_by_enumeration(input_set, num_subsets,
                                                    num_partitions,
                                                    balanced=balanced,
                                                    rnd=rnd)
    else:
        return random_partitions_by_sampling(input_set, num_subsets,
                                             num_partitions,
                                             stirling=stirling,
                                             balanced=balanced,
                                             rnd=rnd)
