#!/usr/bin/env python3

from context import libspn as spn
from test import TestCase
import tensorflow as tf


class TestPartition(TestCase):

    def test_serialize_enums(self):
        class TestEnum(spn.utils.Enum):
            FOO = 1
            BAR = 2
            BAZ = 1

        # Serialize and deserialize
        data1 = {'val1': TestEnum.FOO,
                 'val2': TestEnum.BAR,
                 'val3': TestEnum.BAZ}
        data_json = spn.utils.json_dumps(data1)
        data2 = spn.utils.json_loads(data_json)

        # Check
        self.assertIs(data1['val1'], data2['val1'])
        self.assertIs(data1['val2'], data2['val2'])
        self.assertIs(data1['val3'], data2['val3'])


if __name__ == '__main__':
    tf.test.main()
