#!/usr/bin/env python3

# ------------------------------------------------------------------------
# Copyright (C) 2016 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

import unittest
import tensorflow as tf
# from context import libspn as spn


class TestInference(unittest.TestCase):

    def tearDown(self):
        tf.reset_default_graph()

    @classmethod
    def setUpClass(cls):
        pass

    def test_hard_em(self):
        pass


if __name__ == '__main__':
    unittest.main()
