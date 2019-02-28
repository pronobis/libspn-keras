"""LibSPN tools and utilities."""


def decode_bytes_array(arr):
    """Convert an array of bytes objects to an array of Unicode strings."""
    if arr.dtype.hasobject and type(arr.item(0)) is bytes:
        return arr.astype(str)
    else:
        return arr


def maybe_first(a, b):
    """Returns first argument 'a' if it is not None else 'b' """
    return b if a is None else a
