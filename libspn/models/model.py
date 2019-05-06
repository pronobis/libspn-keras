from abc import ABC, abstractmethod
from libspn.log import get_logger
from libspn import utils


class Model(ABC):
    """An abstract class defining the interface of a model."""

    __logger = get_logger()
    __info = __logger.info

    def __init__(self):
        self._root = None

    def __repr__(self):
        return type(self).__qualname__

    @property
    def root(self):
        """OpNode: Root node of the model."""
        return self._root

    @abstractmethod
    def serialize(self, save_param_vals=True, sess=None):
        """Convert this model into a dictionary for serialization.

        Args:
            save_param_vals (bool): If ``True``, values of parameters will be
                evaluated in a session and stored. The TF variables of parameter
                nodes must already be initialized. If a valid session cannot be
                found, the parameter values will not be retrieved.
            sess (Session): Optional. Session used to retrieve parameter values.
                            If ``None``, the default session is used.

        Returns:
            dict: Dictionary with all the data to be serialized.
        """

    @abstractmethod
    def deserialize(self, data, load_param_vals=True, sess=None):
        """Initialize this model with the ``data`` dict during deserialization.

        Args:
            data (dict): Dictionary with all the data to be deserialized.
            load_param_vals (bool): If ``True``, saved values of parameters will
                                    be loaded and assigned in a session.
            sess (Session): Optional. Session used to assign parameter values.
                            If ``None``, the default session is used.
        """

    @abstractmethod
    def build():
        """Build the SPN graph of the model.

        Returns:
           Node: Root node of the generated model.
        """

    def save_to_json(self, path, pretty=False, save_param_vals=True, sess=None):
        """Saves the model to a JSON file.

        Args:
            path (str): Full path to the file.
            pretty (bool): Use pretty printing.
            save_param_vals (bool): If ``True``, values of parameters will be
                evaluated in a session and saved. The TF variables of parameter
                nodes must already be initialized. If a valid session cannot be
                found, the parameter values will not be saved.
            sess (Session): Optional. Session used to retrieve parameter values.
                            If ``None``, the default session is used.
        """
        self.__info("Saving %s to file '%s'" % (self, path))
        data = self.serialize(save_param_vals=save_param_vals, sess=sess)
        data['model_type'] = utils.type2str(type(self))
        utils.json_dump(path, data, pretty=pretty)

    @staticmethod
    def load_from_json(path, load_param_vals=True, sess=None):
        """Loads a model from a JSON file.

        Args:
            path (str): Full path to the file.
            load_param_vals (bool): If ``True``, saved values of parameters will
                                    be loaded and assigned in a session.
            sess (Session): Optional. Session used to assign parameter values.
                            If ``None``, the default session is used.

        Returns:
           Model: The model.
        """
        Model.__info("Loading model from file '%s'" % path)
        data = utils.json_load(path)
        model_type = utils.str2type(data['model_type'])
        model_instance = model_type.__new__(model_type)
        model_instance.deserialize(data, load_param_vals=load_param_vals,
                                   sess=sess)
        Model.__info("Loaded model %s" % model_instance)
        return model_instance
