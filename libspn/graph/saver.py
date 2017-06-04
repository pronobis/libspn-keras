# ------------------------------------------------------------------------
# Copyright (C) 2016-2017 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

from abc import ABC, abstractmethod
from libspn import utils
from libspn.graph.algorithms import traverse_graph
from libspn.log import get_logger
import tensorflow as tf


class Saver(ABC):
    """An abstract class defining the interface of a saver saving an SPN to
    a file.

    Args:
        path (str): Full path to the file.
        sess (Session): Optional. Session used to retrieve parameter values.
                        If ``None``, the default session is used.
    """

    def __init__(self, path, sess=None):
        self._path = path
        self._sess = sess

    @abstractmethod
    def save(self, root, save_param_vals=True):
        """Saves the SPN specified by ``root`` to a file.

        Args:
            root (Node): Root of the SPN to be saved.
            save_param_vals (bool): If ``True``, values of parameters will be
                evaluated in a session and saved. The TF variables of parameter
                nodes must already be initialized. If a valid session cannot be
                found, the parameter values will not be saved.
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

    def __init__(self, path, pretty=False, sess=None):
        super().__init__(path, sess)
        self._pretty = pretty

    @utils.docinherit(Saver)
    def save(self, root, save_param_vals=True):
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

        self.__info("Saving SPN graph rooted in '%s' to file '%s'"
                    % (root, self._path))

        # Check session
        sess = tf.get_default_session() if self._sess is None else self._sess
        if save_param_vals and sess is None:
            self.__debug1("No valid session found, "
                          "parameter values will not be saved!")
            save_param_vals = False

        # Serialize all nodes
        traverse_graph(root, fun=fun, skip_params=False)

        # Get and fill values of all variables
        if save_param_vals:
            param_vals = sess.run(param_vars)
            for (i, k), v in param_vals.items():
                node_datas[i][k] = v.tolist()

        # Write JSON
        data = {'root': root.name, 'nodes': node_datas}
        utils.json_dump(self._path, data, pretty=self._pretty)
