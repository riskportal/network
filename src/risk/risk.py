"""
risk/risk
~~~~~~~~~
"""

from ._annotation import AnnotationIO
from ._log import params, set_global_verbosity
from ._neighborhoods import NeighborhoodsAPI
from ._network import GraphAPI, NetworkIO, PlotterAPI


class RISK(NetworkIO, AnnotationIO, NeighborhoodsAPI, GraphAPI, PlotterAPI):
    """
    RISK: A class for network analysis and visualization.

    The RISK class integrates functionalities for loading networks, processing annotations,
    performing network-based statistical analysis to quantify neighborhood relationships,
    and visualizing networks and their properties.
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize the RISK class with configuration settings.

        Args:
            verbose (bool): If False, suppresses all log messages to the console. Defaults to True.
        """
        # Set global verbosity for logging
        set_global_verbosity(verbose)
        # Provide public access to network parameters
        self.params = params
        super().__init__()
