from libspn.utils.enum import Enum


class SPNGraphKeys(Enum):
    """Enum of all GraphKeys that may be used to add specific variables and ops to predefined
    collections. """

    """Key for all SPN weights variables. """
    WEIGHTS = "weights"

    """Key for all SPN normal leaf locations. """
    NORMAL_LOC = "normal_loc"

    """Key for all SPN normal scales. """
    NORMAL_SCALE = "normal_scale"

    """Key for all normal leaf variables. """
    NORMAL_VARIABLES = "normal_variables"
    
    """Key for all parameters in an SPN """
    SPN_PARAMETERS = "spn_parameters"
