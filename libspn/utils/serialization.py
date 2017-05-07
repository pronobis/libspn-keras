# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

"""LibSPN serialization tools."""
from collections import OrderedDict
import json


__types = {}
"""Dict of types registered for serialization."""


def register_serializable(cls):
    """Register the class for serialization."""
    __types[type2str(cls)] = cls


def str2type(name):
    """Convert type name to type."""
    return __types.get(name)


def type2str(cls):
    """Convert type to type name."""
    return cls.__qualname__


def _is_serializable(cls):
    """Return ``True`` for registered serializable types."""
    return (callable(getattr(cls, "serialize", None)) and
            cls in __types.values())


def _encode_json(obj):
    """Convert all registered serializable types into a dictionary with
    the ``__type__`` entry."""
    obj_type = type(obj)
    if _is_serializable(obj_type):
        # Encode custom object
        data = obj.serialize()
        return OrderedDict([('__type__', type2str(obj_type))] +
                           sorted(data.items(), key=lambda t: t[0]))
    else:
        return obj


def _decode_json(obj):
    """Convert all dictionaries with the ``__type__`` entry into
    an instance of a registered serializable type."""
    if isinstance(obj, dict) and '__type__' in obj:
        # Decode custom object
        obj_type = str2type(obj['__type__'])
        obj_instance = obj_type.__new__(obj_type)
        obj_instance.deserialize(obj)
        return obj_instance
    else:
        return obj


def json_dump(path, data, pretty=False):
    """Dump data into a JSON file."""
    with open(path, 'w') as json_file:
        json.dump(data, json_file, default=_encode_json,
                  sort_keys=False,
                  indent=2 if pretty else None)


def json_load(path):
    """Load data from a JSON file."""
    with open(path, 'r') as json_file:
        return json.load(json_file, object_hook=_decode_json)
