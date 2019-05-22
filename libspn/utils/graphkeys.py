from libspn.utils.enum import Enum


class SPNGraphKeys(Enum):
    """Enum of all GraphKeys that may be used to add specific variables and ops to predefined
    collections. """

    """Key for all SPN weights variables. """
    WEIGHTS = "weights"

    """Key for all SPN 'location-scale' distribution locs. """
    DIST_LOC = "dist_loc"

    """Key for all SPN 'location-scale' distribution scales. """
    DIST_SCALE = "dist_scale"

    """Key for all degrees of freedom variables. """
    DIST_DF = "dist_df"

    """Key for all normal leaf variables. """
    DIST_PARAMETERS = "normal_variables"

    """Key for all parameters in an SPN """
    SPN_PARAMETERS = "spn_parameters"