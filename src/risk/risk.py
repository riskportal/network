"""
risk/risk
~~~~~~~~~
"""

from risk.annotation.io import AnnotationIO
from risk.log import params, set_global_verbosity
from risk.neighborhoods.api import NeighborhoodsAPI
from risk.network.graph.api import GraphAPI
from risk.network.io import NetworkIO
from risk.network.plotter.api import PlotterAPI


class RISK(NetworkIO, AnnotationIO, NeighborhoodsAPI, GraphAPI, PlotterAPI):
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
