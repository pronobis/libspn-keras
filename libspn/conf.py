# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

"""Global configuration options of LibSPN."""

import tensorflow as tf

dtype = tf.float32
"""Default dtype used by LibSPN."""

custom_gather_cols = True
"""Whether to use custom op for implementing
:meth:`~libspn.utils.gather_cols`."""

custom_gather_cols_3d = True
"""Whether to use custom op for implementing
:meth:`~libspn.utils.gather_cols_3d`."""

custom_scatter_cols = True
"""Whether to use custom op for implementing
:meth:`~libspn.utils.scatter_cols`."""

custom_scatter_values = True
"""Whether to use custom op for implementing
:meth:`~libspn.utils.scatter_values`."""

sumslayer_count_sum_strategy = "gather"
"""Strategy to apply when summing counts
within a SumsLayer. Can be 'segmented',
'gather' or 'None' """
