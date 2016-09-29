#!/usr/bin/env python3

# ------------------------------------------------------------------------
# Copyright (C) 2016 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

import unittest
from context import libspn as spn
import tensorflow as tf


class TestUtils(unittest.TestCase):

    def tearDown(self):
        tf.reset_default_graph()


if __name__ == '__main__':
    unittest.main()
