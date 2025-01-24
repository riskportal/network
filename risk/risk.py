"""
risk/risk
~~~~~~~~~
"""

from risk.network import NetworkIO
from risk.annotations import AnnotationsIO
from risk.neighborhoods import NeighborhoodsAPI
from risk.network.graph import GraphAPI
from risk.network.plotter import PlotterAPI

from risk.log import params, set_global_verbosity


class RISK(NetworkIO, AnnotationsIO, NeighborhoodsAPI, GraphAPI, PlotterAPI):
    """RISK: A class for network analysis and visualization.

    The RISK class integrates functionalities for loading networks, processing annotations,
    performing network-based statistical analysis to quantify neighborhood relationships,
    and visualizing networks and their properties.
    """

    def __init__(self, verbose: bool = True):
        """Initialize the RISK class with configuration settings.

        Args:
            verbose (bool): If False, suppresses all log messages to the console. Defaults to True.
        """
        # Set global verbosity for logging
        set_global_verbosity(verbose)
        # Provide public access to network parameters
        self.params = params
        super().__init__()
