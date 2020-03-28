class DimensionPermutation:
    AUTO = 'auto'
    SPATIAL = 'spatial'
    REGIONS = 'regions'


def infer_dimension_permutation(shape):
    num_none = sum(1 if s is None else 0 for s in shape)

    if num_none == 0:
        raise ValueError(
            "Cannot infer permutation as there are no dynamic dimension sizes, provide permutation "
            "explicitly at instantiating layers or use a dynamic batch size."
        )

    if num_none > 1:
        raise ValueError(
            "Cannot infer permutation as there are multiple dynamic dimension sizes, provide "
            "permutation explicitly when instantiating layers."
        )

    none_index = list(shape).index(None)

    if none_index == 0:
        return DimensionPermutation.SPATIAL
    elif none_index == 2:
        return DimensionPermutation.REGIONS
    else:
        raise ValueError("Index of dynamically sized dimension was neither 0 (SPATIAL) "
                         "nor 2 (REGIONS).")
