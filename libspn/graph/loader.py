# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

from abc import ABC, abstractmethod
from libspn import utils
from libspn.log import get_logger
import tensorflow as tf


class Loader(ABC):
    """An abstract class defining the interface of a loader loading an SPN from
    a file.

    Args:
        path (str): Full path to the file.
        sess (Session): Optional. Session used to assign parameter values.
                        If ``None``, the default session is used.
    """

    def __init__(self, path, sess=None):
        self._path = path
        self._sess = sess
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
    def load(self, param_vals=True):
        """Loads the SPN.

        Returns:
            Node: The root of the loaded SPN.
            load_param_vals (bool): If ``True``, saved values of parameters will
                                    be loaded and assigned in a session.
        """


class JSONLoader(Loader):
    """Loads an SPN from a JSON file.

    Args:
        path (str): Full path to the file.
        sess (Session): Optional. Session used to assign parameter values.
                        If ``None``, the default session is used.
    """

    __logger = get_logger()
    __info = __logger.info

    def __init__(self, path, sess=None):
        super().__init__(path, sess)

    @utils.docinherit(Loader)
    def load(self, load_param_vals=True):
        self.__info("Loading SPN graph from file '%s'" % self._path)

        # Check session
        sess = tf.get_default_session() if self._sess is None else self._sess
        if load_param_vals and sess is None:
            self.__debug1("No valid session found, "
                          "parameter values will not be loaded!")
            load_param_vals = False

        # Load file
        data = utils.json_load(self._path)

        # Deserialize all nodes
        node_datas = data['nodes']
        nodes = [None] * len(node_datas)
        ops = []
        self._nodes_by_name = {}
        for ni, d in enumerate(node_datas):
            node_type = utils.str2type(d['node_type'])
            node_instance = node_type.__new__(node_type)
            op = node_instance.deserialize(d)
            if node_instance.is_param and op is not None:
                ops.append(op)
            self._nodes_by_name[d['name']] = node_instance
            nodes[ni] = node_instance

        # Run any deserialization ops for parameter values
        if load_param_vals and ops:
            sess.run(ops)

        # Link nodes
        for n, nd in zip(nodes, node_datas):
            if n.is_op:
                n.deserialize_inputs(nd, self._nodes_by_name)

        # Retrieve root
        root = self._nodes_by_name[data['root']]
        return root
