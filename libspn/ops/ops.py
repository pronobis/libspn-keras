# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

"""Custom TensorFlow operations implemented in C++."""

import os
import tensorflow as tf

# Load module with custom ops
lib_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        'libspn_ops.so')
libspn_ops_module = tf.load_op_library(lib_path)

# Load operations into the namespace
gather_cols = libspn_ops_module.gather_columns
scatter_cols = libspn_ops_module.scatter_columns
scatter_values = libspn_ops_module.scatter_values
