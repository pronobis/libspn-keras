"""Visualizing TensorFlow graph."""
import numpy as np
import tensorflow as tf


def display_tf_graph(graph=None, max_const_size=32):
    """Visualize a TensorFlow graph in IPython/Jupyter.

    Args:
        graph: Graph or GraphDef to visualize.
               If ``None``, default graph is used.
        max_const_size: Max const size that will not be stripped from graph.
    """
    if graph is None:
        graph = tf.get_default_graph()

    # This function is stolen from
    # http://nbviewer.jupyter.org/github/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/deepdream/deepdream.ipynb
    if hasattr(graph, 'as_graph_def'):
        graph_def = graph.as_graph_def()
    else:
        graph_def = graph
    strip_def = _strip_consts(graph_def, max_const_size=max_const_size)
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html"
              onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)), id='graph' + str(np.random.rand()))

    iframe = """
        <iframe seamless style="width:1300px;height:620px;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))
    import IPython.display  # Import only if needed
    IPython.display.display(IPython.display.HTML(iframe))


def _strip_consts(graph_def, max_const_size=32):
    """Strip large constant values from graph_def."""
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add()
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = b"<stripped %d bytes>" % size
    return strip_def
