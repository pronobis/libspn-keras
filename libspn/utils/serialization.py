"""LibSPN serialization tools."""
from collections import OrderedDict
import json
import enum


__types = {}
"""Dict of types registered for serialization."""


def register_serializable(cls):
    """A class decorator registering the class for serialization."""
    __types[type2str(cls)] = cls
    return cls


def str2type(name):
    """Convert type name to type."""
    t = __types.get(name)
    if t is None:
        raise TypeError("Unknown type '%s'" % name)
    return t


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
        raise TypeError(repr(obj) + " is not JSON serializable")


def _decode_json(obj):
    """Convert all dictionaries with the ``__type__`` entry into
    an instance of a registered serializable type."""
    if isinstance(obj, dict) and '__type__' in obj:
        # Decode custom object
        obj_type = str2type(obj['__type__'])
        # A workaround for enums, which need their value in __new__
        if issubclass(obj_type, enum.Enum):
            obj_instance = obj_type.__new__(obj_type,
                                            obj_type.deserialize(obj))
        else:
            obj_instance = obj_type.__new__(obj_type)
            obj_instance.deserialize(obj)
        return obj_instance
    else:
        return obj


def json_dumps(data, pretty=False):
    """Dump data into a JSON string."""
    return json.dumps(data, default=_encode_json,
                      sort_keys=False,
                      indent=2 if pretty else None)


def json_loads(s):
    """Load data from a JSON string."""
    return json.loads(s, object_hook=_decode_json)


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
