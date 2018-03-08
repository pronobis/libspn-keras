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

custom_scatter_cols = True
"""Whether to use custom op for implementing
:meth:`~libspn.utils.scatter_cols`."""

custom_scatter_values = True
"""Whether to use custom op for implementing
:meth:`~libspn.utils.scatter_values`."""

sumslayer_count_with_matmul = True
"""Whether to add the counts inside a 
SumsLayer while computing MPE path.
"""