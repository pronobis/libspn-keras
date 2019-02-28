"""Visualizing SPN graph."""

import os
from string import Template
import uuid
import json
from libspn.graph.algorithms import compute_graph_up


def display_spn_graph(root, skip_params=False):
    """Visualize an SPN graph in IPython/Jupyter.

    Args:
        root (Node): Root of the SPN to visualize.
        skip_params (bool): If ``True``, parameter nodes will not be shown.
    """
    # Graph description for HTML generation
    links = []
    nodes = []
    node_types = []
    leaf_counter = [1]

    def add_node(node, *input_out_sizes):
        """Add a node to the graph description. This function also computes the
        size of the output while traversing the graph. """
        # Get a unique node type number, for any node including leafs
        try:
            node_type = node_types.index(node.__class__)
        except:
            node_type = len(node_types)
            node_types.append(node.__class__)
        # Param and var nodes are added when processing an op node
        if node.is_op:
            # Create this node
            nodes.append({"id": node.name, "name": node.name,
                          "type": node_type, "tooltip": str(node)})
            # Create links to inputs (and param/var nodes)
            for inpt, size in zip(node.inputs,
                                  node._gather_input_sizes(*input_out_sizes)):
                if inpt:
                    if inpt.is_op:
                        # WARNING: Currently if a node has two inputs from the
                        # same node, they will be added correctly, but displayed
                        # on top of each other
                        links.append({"source": inpt.node.name,
                                      "target": node.name, "value": size})
                    elif not skip_params or not inpt.is_param:
                        # Unique id for a leaf node
                        leaf_id = inpt.node.name + "_" + str(leaf_counter[0])
                        # Add indices in the name of the node
                        leaf_name = (inpt.node.name if inpt.indices is None
                                     else inpt.node._name + str(inpt.indices))
                        leaf_type = node_types.index(inpt.node.__class__)  # Must exist
                        leaf_counter[0] += 1
                        nodes.append({"id": leaf_id, "name": leaf_name,
                                      "type": leaf_type, "tooltip": str(inpt.node)})
                        links.append({"source": leaf_id, "target": node.name,
                                      "value": size})

        # Return computed outputs size
        return node._compute_out_size(*input_out_sizes)

    # Compute graph & build HTML
    compute_graph_up(root, val_fun=add_node)
    html = _html_graph(nodes, links)
    import IPython.display  # Import only if needed
    IPython.display.display(IPython.display.HTML(html))


def _html_graph(nodes, links):
    """Generate HTML5 code displaying a graph."""
    # Load a the HTML template
    html_file = os.path.realpath(
        os.path.join(os.getcwd(),
                     os.path.dirname(__file__),
                     "graph.html"))
    with open(html_file, 'r') as content_file:
        template = Template(content_file.read())

    graph = {"nodes": nodes, "links": links}

    return template.substitute(svgId="S" + str(uuid.uuid4()).replace('-', ''),
                               graph=json.dumps(graph))
