"""Global configuration options of LibSPN."""

import tensorflow as tf

dtype = tf.float32
"""Default dtype used by LibSPN."""

sumslayer_count_sum_strategy = "gather"
"""Strategy to apply when summing counts
within a SumsLayer. Can be 'segmented',
'gather' or 'None' """

memoization = True
"""Whether to use LRU caches to function
return values in successive calls for reduced
graph size."""

custom_gradient = True
"""Whether or not to use custom gradient implementations,
implemented within the respective Op nodes."""

argmax_zero = False
"""Whether to always return zero when 
argmax in BaseSum is faced with multiple maxes. 
If False, selects random a 'winner' among 
the maxes.
"""
