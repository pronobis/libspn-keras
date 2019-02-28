from abc import ABC, abstractmethod
from libspn import utils
from libspn.log import get_logger
from libspn.graph.serialization import deserialize_graph
import os


class Loader(ABC):
    """An abstract class defining the interface of a loader loading an SPN from a file.

    Args:
        path (str): Full path to the file.
    """

    def __init__(self, path):
        self._path = os.path.expanduser(path)
        self._nodes_by_name = {}  # Dict of nodes indexed by original name

    def find_node(self, node_name):
        """Find a node by the name it had when it was saved.

        Note that the current name of a node might be different if another node
        of the same name existed when the nodes were loaded.

        Args:
           node_name (str): Original node name.

        Returns:
            Node: The found node or ``None`` if the node could not be found.
        """
        return self._nodes_by_name.get(node_name, None)

    @abstractmethod
    def load(self, load_param_vals=True, sess=None):
        """Loads the SPN.

        Args:
            load_param_vals (bool): If ``True``, saved values of parameters will
                                    be loaded and assigned in a session.
            sess (Session): Optional. Session used to assign parameter values.
                            If ``None``, the default session is used.

        Returns:
            Node: The root of the loaded SPN.
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
    __debug1 = __logger.debug1

    def __init__(self, path):
        super().__init__(path)

    @utils.docinherit(Loader)
    def load(self, load_param_vals=True, sess=None):
        self.__info("Loading SPN graph from file '%s'" % self._path)
        data = utils.json_load(self._path)
        root = deserialize_graph(data, load_param_vals=load_param_vals,
                                 sess=sess, nodes_by_name=self._nodes_by_name)
        return root
