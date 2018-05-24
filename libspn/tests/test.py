# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

import tensorflow as tf
import os
import itertools


class TestCase(tf.test.TestCase):

    @classmethod
    def setUpClass(cls):
        super(TestCase, cls).setUpClass()
        cur_dir = os.path.realpath(os.path.join(os.getcwd(),
                                                os.path.dirname(__file__)))
        cls.out_dir = os.path.join(cur_dir, "out")
        cls.data_dir = os.path.join(cur_dir, "data")
        cls.logs_dir = os.path.join(cur_dir, "logs")

    @classmethod
    def path(cls, *p):
        dirs = p[:-1]
        filenames = p[-1]
        if isinstance(filenames, list):
            return [os.path.join(*dirs, i) for i in filenames]
        else:
            return os.path.join(*dirs, filenames)

    @classmethod
    def data_path(cls, *p):
        return cls.path(cls.data_dir, *p)

    @classmethod
    def logs_path(cls, *p):
        return cls.path(cls.logs_dir, *p)

    @classmethod
    def out_path(cls, *p):
        return cls.path(cls.out_dir, *p)

    @classmethod
    def write_tf_graph(cls, sess, *p):
        writer = tf.summary.FileWriter(cls.logs_path(*p), sess.graph)
        writer.add_graph(sess.graph)
        writer.close()

    def cid(self):
        return self.id().split('.')[-1]

    def sid(self):
        return self.id().split('.')[-2]


def arg_product(*args):
    return [tuple(elem) for elem in itertools.product(*args)]