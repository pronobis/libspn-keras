from libspn.utils.serialization import register_serializable
import enum


class Enum(enum.Enum):
    """Serializable Enum.

    Any enum subclassing from this automatically becomes serializable.
    No further registration is necessary.
    """

    def __init__(self, *args):
        # The init function is called by metaclass at the time of the enum
        # definition. Therefore, we can register the enum for serialization here.
        # This way all enums subclassing from this are automatically serializable.
        register_serializable(type(self))

    def serialize(self):
        return {'value': self.name}

    @classmethod
    def deserialize(cls, data):
        """This function does not follow the standard deserialization pattern,
        but instade it just returns the value. The reason is that with enum, the
        value must be passed directly to __new__ and cannot be deserialized
        after __new__. Therefore, we use a workaround inside _decode_json for
        such case.
        """
        name = data['value']

        # Lookup the value of the enum based on name
        # enums are created using value not name
        return cls[name]
