import collections.abc


class Scope(collections.abc.Set):
    """Class storing a scope of an output value in the SPN graph.

    Each random variable in the graph is uniquely identified by the instance of
    the variable node where it is first created as well as an integer ID
    identifying the random variable within that variable node.

    A scope should be seen as an immutable set. Scopes are first created in
    variable nodes and then merged in other nodes.

    Scopes are hashable and can be used as keys in dictionaries.

    The constructor creates a singleton scope containing one variable.

    Args:
        node (VarNode): Node identifying a single random variable used to
                        initialize the scope.
        var_id (int): ID identifying a single random variable used to
                      initialize the scope.
    """

    def __init__(self, node, var_id):
        self.__set = frozenset([(node, var_id)])

    @classmethod
    def __empty(cls):
        return cls.__new__(cls)

    def __contains__(self, item):
        return item in self.__set

    def __len__(self):
        return self.__set.__len__()

    def __iter__(self):
        return self.__set.__iter__()

    def __or__(self, other):
        if not isinstance(other, Scope):
            return NotImplemented
        scope = Scope.__empty()
        scope.__set = self.__set.__or__(other.__set)
        return scope

    def __and__(self, other):
        if not isinstance(other, Scope):
            return NotImplemented
        scope = Scope.__empty()
        scope.__set = self.__set.__and__(other.__set)
        return scope

    def __sub__(self, other):
        if not isinstance(other, Scope):
            return NotImplemented
        scope = Scope.__empty()
        scope.__set = self.__set.__sub__(other.__set)
        return scope

    def __xor__(self, other):
        if not isinstance(other, Scope):
            return NotImplemented
        scope = Scope.__empty()
        scope.__set = self.__set.__xor__(other.__set)
        return scope

    def __ror__(self, other):
        return NotImplemented

    def __rand__(self, other):
        return NotImplemented

    def __rsub__(self, other):
        return NotImplemented

    def __rxor__(self, other):
        return NotImplemented

    def difference(self, other):
        if not isinstance(other, Scope):
            return NotImplemented
        scope = Scope.__empty()
        scope.__set = self.__set.difference(other.__set)
        return scope

    def intersection(self, other):
        if not isinstance(other, Scope):
            return NotImplemented
        scope = Scope.__empty()
        scope.__set = self.__set.intersection(other.__set)
        return scope

    def symmetric_difference(self, other):
        if not isinstance(other, Scope):
            return NotImplemented
        scope = Scope.__empty()
        scope.__set = self.__set.symmetric_difference(other.__set)
        return scope

    def union(self, other):
        if not isinstance(other, Scope):
            return NotImplemented
        scope = Scope.__empty()
        scope.__set = self.__set.union(other.__set)
        return scope

    def isdisjoint(self, other):
        if not isinstance(other, Scope):
            return NotImplemented
        return self.__set.isdisjoint(other.__set)

    def issubset(self, other):
        if not isinstance(other, Scope):
            return NotImplemented
        return self.__set.issubset(other.__set)

    def issuperset(self, other):
        if not isinstance(other, Scope):
            return NotImplemented
        return self.__set.issuperset(other.__set)

    def __eq__(self, other):
        return self.__set.__eq__(other.__set)

    def __ne__(self, other):
        return self.__set.__ne__(other.__set)

    def __ge__(self, other):
        return self.__set.__ge__(other.__set)

    def __gt__(self, other):
        return self.__set.__gt__(other.__set)

    def __le__(self, other):
        return self.__set.__le__(other.__set)

    def __lt__(self, other):
        return self.__set.__lt__(other.__set)

    def __reduce__(self):
        raise NotImplementedError

    def __reduce_ex__(self, protocol):
        raise NotImplementedError

    def __hash__(self):
        return self.__set.__hash__()

    @staticmethod
    def merge_scopes(scopes):
        """Merge the scopes in ``scopes`` into a single scope.

        Args:
            scopes (iterable of Scope): List or iterator of scopes to merge.

        Returns:
            Scope: Merged scope.
        """
        scope = Scope.__empty()
        scope.__set = frozenset([v for s in scopes for v in s])
        return scope

    def __repr__(self):
        return ("Scope({%s})" % ', '.join(
            ("%s:%s" % (v[0], v[1])) for v in self.__set))
