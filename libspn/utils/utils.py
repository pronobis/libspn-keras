# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------
"""LibSPN tools and utilities."""


def decode_bytes_array(arr):
    """Convert an array of bytes objects to an array of Unicode strings."""
    if arr.dtype.hasobject and type(arr.item(0)) is bytes:
        return arr.astype(str)
    else:
        return arr


def maybe_first(a, b):
    if a is None:
        return b
    return a
