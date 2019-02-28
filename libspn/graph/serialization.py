"""SPN graph serialization."""

from libspn.log import get_logger
from libspn import utils
from libspn.graph.algorithms import traverse_graph
import tensorflow as tf

logger = get_logger()


def serialize_graph(root, save_param_vals=True, sess=None):
    """Convert an SPN graph rooted in ``root`` into a dictionary for serialization.

    The graph is converted to a dict here rather than a collection of Node since
    additional processing is done (retrieval of variable values inside a session)
    which cannot easily be done from within JSON encoder.

    Args:
        root (Node): Root of the SPN to be serialized.
        save_param_vals (bool): If ``True``, values of parameters will be
            evaluated in a session and stored. The TF variables of parameter
            nodes must already be initialized. If a valid session cannot be
            found, the parameter values will not be retrieved.
        sess (Session): Optional. Session used to retrieve parameter values.
                        If ``None``, the default session is used.

    Returns:
        dict: Dictionary with all the data to be serialized.
    """
    node_datas = []
    param_vars = {}

    def fun(node):
        data = node.serialize()
        # The nodes will not be deserialized automatically during JSON
        # decoding since they do not use the __type__ data field.
        data['node_type'] = utils.type2str(type(node))
        data_index = len(node_datas)
        node_datas.append(data)
        # Handle param variables
        if node.is_param:
            if save_param_vals:
                # Get all variables
                for k, v in data.items():
                    if isinstance(v, tf.Variable):
                        param_vars[(data_index, k)] = v
            else:
                # Ignore all variables
                for k, v in data.items():
                    if isinstance(v, tf.Variable):
                        data[k] = None

    # Check session
    if sess is None:
        sess = tf.get_default_session()
    if save_param_vals and sess is None:
        logger.debug1("No valid session found, "
                      "parameter values will not be saved!")
        save_param_vals = False

    # Serialize all nodes
    traverse_graph(root, fun=fun, skip_params=False)

    # Get and fill values of all variables
    if save_param_vals:
        param_vals = sess.run(param_vars)
        for (i, k), v in param_vals.items():
            node_datas[i][k] = v.tolist()

    return {'root': root.name, 'nodes': node_datas}


def deserialize_graph(data, load_param_vals=True, sess=None,
                      nodes_by_name=None):
    """Create an SPN graph based on the ``data`` dict during deserialization.

    Args:
        data (dict): Dictionary with all the data to be deserialized.
        load_param_vals (bool): If ``True``, saved values of parameters will
                                be loaded and assigned in a session.
        sess (Session): Optional. Session used to assign parameter values.
                        If ``None``, the default session is used.
        nodes_by_name (dict): A dictionary that will be filled with the
                              generated nodes organized by their original name
                              (one they had when they were serialized). Note
                              that the current name of a node might be different
                              if another node of the same name existed when the
                              nodes were loaded.

    Returns:
        Node: The root of the SPN graph.
    """
    if nodes_by_name is None:
        nodes_by_name = {}

    # Check session
    if sess is None:
        sess = tf.get_default_session()
    if load_param_vals and sess is None:
        logger.debug1("No valid session found, "
                      "parameter values will not be loaded!")
        load_param_vals = False

    # Deserialize all nodes
    node_datas = data['nodes']
    nodes = [None] * len(node_datas)
    ops = []
    for ni, d in enumerate(node_datas):
        node_type = utils.str2type(d['node_type'])
        node_instance = node_type.__new__(node_type)
        op = node_instance.deserialize(d)
        if node_instance.is_param and op is not None:
            ops.append(op)
        nodes_by_name[d['name']] = node_instance
        nodes[ni] = node_instance

    # Run any deserialization ops for parameter values
    if load_param_vals and ops:
        sess.run(ops)

    # Link nodes
    for n, nd in zip(nodes, node_datas):
        if n.is_op:
            n.deserialize_inputs(nd, nodes_by_name)

    # Retrieve root
    root = nodes_by_name[data['root']]
    return root
