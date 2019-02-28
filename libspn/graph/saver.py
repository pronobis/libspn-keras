from abc import ABC, abstractmethod
from libspn import utils
from libspn.graph.serialization import serialize_graph
from libspn.log import get_logger
import os


class Saver(ABC):
    """An abstract class defining the interface of a saver saving an SPN to a file.

    Args:
        path (str): Full path to the file.
    """

    def __init__(self, path):
        self._path = os.path.expanduser(path)

    @abstractmethod
    def save(self, root, save_param_vals=True, sess=None):
        """Saves the SPN rooted in ``root`` to a file.

        Args:
            root (Node): Root of the SPN to be saved.
            save_param_vals (bool): If ``True``, values of parameters will be
                evaluated in a session and saved. The TF variables of parameter
                nodes must already be initialized. If a valid session cannot be
                found, the parameter values will not be saved.
            sess (Session): Optional. Session used to retrieve parameter values.
                            If ``None``, the default session is used.
        """


class JSONSaver(Saver):
    """Saves an SPN to a JSON file.

    Args:
        path (str): Full path to the file.
        pretty (bool): Use pretty printing.
        sess (Session): Optional. Session used to retrieve parameter values.
                        If ``None``, the default session is used.
    """

    __logger = get_logger()
    __info = __logger.info
    __debug1 = __logger.debug1

    def __init__(self, path, pretty=False):
        super().__init__(path)
        self._pretty = pretty

    @utils.docinherit(Saver)
    def save(self, root, save_param_vals=True, sess=None):
        self.__info("Saving SPN graph rooted in '%s' to file '%s'"
                    % (root, self._path))
        data = serialize_graph(root, save_param_vals=save_param_vals, sess=sess)
        utils.json_dump(self._path, data, pretty=self._pretty)
