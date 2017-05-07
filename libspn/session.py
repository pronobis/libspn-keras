# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

"""LibSVM session and running helpers."""

from contextlib import contextmanager
import tensorflow as tf
from libspn.log import get_logger

logger = get_logger()


@contextmanager
def session():
    """Context manager initializing and deinitializing the standard TensorFlow
    session infrastructure.

    Use like this::

        with spn.session() as (sess, run):
            while run():
                sess.run(something)
    """
    # Op for initializing variables
    # As stated in https://github.com/tensorflow/tensorflow/issues/3819
    # all does not include local (e.g. epoch counter)
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    # Create a session
    with tf.Session() as sess:
        # Initialize epoch counter and other variables
        sess.run(init_op)

        # Start input enqueue threads
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            yield sess, (lambda: not coord.should_stop())
        except tf.errors.OutOfRangeError:
            logger.info("Epoch limit reached")
        finally:
            # Ask threads to stop
            coord.request_stop()

        # Wait for thresds to finish
        coord.join(threads)
