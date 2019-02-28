import libspn as spn
from functools import wraps


class _HashedSeq(list):
    """ This class guarantees that hash() will be called no more than once
        per element.  This is important because the memoize() will hash
        the key multiple times on a cache miss.

        Implementation taken from functools package
    """

    __slots__ = 'hashvalue'

    def __init__(self, tup, hash=hash):
        self[:] = tup
        self.hashvalue = hash(tup)

    def __hash__(self):
        return self.hashvalue


def _make_key(args, kwds, typed, kwd_mark=(object(),),
              fasttypes={int, str, frozenset, type(None)},
              sorted=sorted, tuple=tuple, type=type, len=len):
    """Make a cache key from optionally typed positional and keyword arguments

    The key is constructed in a way that is flat as possible rather than
    as a nested structure that would take more memory.

    If there is only a single argument and its data type is known to cache
    its hash value, then that argument is returned without a wrapper.  This
    saves space and improves lookup speed.

    Implementation taken from functools package
    """
    key = args
    if kwds:
        sorted_items = sorted(kwds.items())
        key += kwd_mark
        for item in sorted_items:
            key += item
    if typed:
        key += tuple(type(v) for v in args)
        if kwds:
            key += tuple(type(v) for k, v in sorted_items)
    elif len(key) == 1 and type(key[0]) in fasttypes:
        return key[0]
    return _HashedSeq(key)


def lru_cache(f):
    """Can be used as a decorator for least-recently used caching. This is helpful when traversing
    some edges several times during construction of the SPN, but also for reusing Tensors defined
    in upward computation vs. downward computation.

    Args:
        f (function): A function f(*args, **kwargs) to memoize.

    Returns:
        A wrapped function with same behavior with f but added memoization.
    """
    memo = {}

    @wraps(f)
    def helper(*args, **kwargs):
        if not spn.conf.memoization:
            return f(*args, **kwargs)
        key = _make_key(args, kwargs, typed=True)
        if key not in memo:
            memo[key] = f(*args, **kwargs)
        return memo[key]

    return helper
