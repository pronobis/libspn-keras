# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

from abc import ABC, abstractmethod
from libspn import utils
from libspn.graph.node import OpNode


class Loader(ABC):
    """An abstract class defining the interface of a loader loading an SPN from
    a file.

    Args:
        path (str): Full path to the file.
    """

    def __init__(self, path):
        self._path = path
        self._nodes_by_name = {}  # Dict of nodes indexed by original name

    def find_node(self, node_name):
        """Find a node by the name it had when it was saved. The current name of
        a node might be different if another node of the same name existed when
        the nodes were loaded.

        Args:
           node_name (str): Original node name.

        Returns:
            Node: The found node or ``None`` if the node could not be found.
        """
        return self._nodes_by_name.get(node_name, None)

    @abstractmethod
    def load(self):
        """Loads the SPN.

        Returns:
            Node: The root of the loaded SPN.
        """


class JSONLoader(Loader):
    """Loads an SPN from a JSON file.

    Args:
        path (str): Full path to the file.
    """

    def __init__(self, path):
        super().__init__(path)

    @utils.docinherit(Loader)
    def load(self):
        data = utils.json_load(self._path)
        # Deserialize all nodes
        node_datas = data['nodes']
        nodes = [None] * len(node_datas)
        self._nodes_by_name = {}
        for ni, d in enumerate(node_datas):
            node_type = utils.str2type(d['node_type'])
            node_instance = node_type.__new__(node_type)
            node_instance.deserialize(d)
            self._nodes_by_name[d['name']] = node_instance
            nodes[ni] = node_instance
        # Link nodes
        for n, nd in zip(nodes, node_datas):
            if isinstance(n, OpNode):
                n.deserialize_inputs(nd, self._nodes_by_name)
        # Retrieve root
        root = self._nodes_by_name[data['root']]
        return root
