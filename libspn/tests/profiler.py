import os

import tensorflow as tf
from google.protobuf import text_format
from tensorflow.python.client import timeline
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder


def profile_report(sess, ops, feed_dict, out_dir, filename_prefix, filename_suffix):
    """
    Creates a profile report. The filename prefix should report information about which performance
    test was used and the filename suffix should contain information about the parameters for a
    specific test. E.g. filename_prefix == 'sum_value_varying_sizes'
    """
    # Build a profiler
    profiler = tf.profiler.Profiler(sess.graph)

    # Run the graph while fetching metadata
    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    sess.run(ops, feed_dict=feed_dict, options=options, run_metadata=run_metadata)

    # Create the Timeline object, and write it to a json file
    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
    chrome_trace = fetched_timeline.generate_chrome_trace_format()
    os.makedirs(out_dir, exist_ok=True)

    # Use the TF profiler and create a report on Name Scope, Python functions and overall Graph
    profiler.add_step(0, run_metadata)
    for infix, fn in zip(
            ["NAME_SCOPE", "GRAPH", "PYTHON"],
            [profiler.profile_name_scope, profiler.profile_graph, profiler.profile_python]):
        # Build options and run the profiling function
        opts = ProfileOptionBuilder(ProfileOptionBuilder.time_and_memory()) \
            .with_step(0) \
            .with_timeline_output(
            '%s/timeline_%s_%s_%s.json' % (out_dir, filename_prefix, infix, filename_suffix)) \
            .build()
        fn(options=opts)

    # Export default report on Operations
    with open('%s/timeline_%s_OPERATIONS_%s.json' %
              (out_dir, filename_prefix, filename_suffix), 'w') as f:
        f.write(chrome_trace)

    ret = tf.profiler.advise(sess.graph, run_metadata)
    with open("%s/advise_%s_%s.txt" % (out_dir, filename_prefix,
                                       filename_suffix), 'w') as f:
        f.write(text_format.MessageToString(ret))
