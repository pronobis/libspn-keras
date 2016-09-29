# ------------------------------------------------------------------------
# Copyright (C) 2016 Andrzej Pronobis - All Rights Reserved
#
# This file is part of LibSPN. Unauthorized use or copying of this file,
# via any medium is strictly prohibited. Proprietary and confidential.
# ------------------------------------------------------------------------

from abc import ABC, abstractmethod
from libspn import utils
from libspn.graph.algorithms import traverse_graph


class Saver(ABC):
    """An abstract class defining the interface of a saver saving an SPN to
    a file.

    Args:
        path (str): Full path to the file.
    """

    def __init__(self, path):
        self._path = path

    @abstractmethod
    def save(self, root):
        """Saves the SPN specified by ``root`` to the file.

        Args:
            root: Root of the SPN to be saved.
        """


class JSONSaver(Saver):
    """Saves an SPN to a JSON file.

    Args:
        path (str): Full path to the file.
    """

    def __init__(self, path, pretty=False):
        super().__init__(path)
        self._pretty = pretty

    @utils.docinherit(Saver)
    def save(self, root):
        node_datas = []

        def fun(node):
            data = node.serialize()
            # The nodes will not be deserialized automatically during JSON
            # decoding since they do not use the __type__ data field.
            data['node_type'] = utils.type2str(type(node))
            node_datas.append(data)

        # Serialize all nodes
        traverse_graph(root, fun=fun, skip_params=False)

        # Write JSON
        data = {'root': root.name, 'nodes': node_datas}
        utils.json_dump(self._path, data, pretty=self._pretty)
