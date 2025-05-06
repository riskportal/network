"""
risk/_network/_plotter/_api
~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from typing import List, Tuple, Union

import numpy as np

from ..._log import log_header
from .._graph import Graph
from ._plotter import Plotter


class PlotterAPI:
    """
    Handles the loading of network plotter objects.

    The PlotterAPI class provides methods to load and configure Plotter objects for plotting network graphs.
    """

    def __init__(self) -> None:
        pass

    def load_plotter(
        self,
        graph: Graph,
        figsize: Union[List, Tuple, np.ndarray] = (10, 10),
        background_color: str = "white",
        background_alpha: Union[float, None] = 1.0,
        pad: float = 0.3,
    ) -> Plotter:
        """
        Get a Plotter object for plotting.

        Args:
            graph (Graph): The graph to plot.
            figsize (List, Tuple, or np.ndarray, optional): Size of the plot. Defaults to (10, 10)., optional): Size of the figure. Defaults to (10, 10).
            background_color (str, optional): Background color of the plot. Defaults to "white".
            background_alpha (float, None, optional): Transparency level of the background color. If provided, it overrides
                any existing alpha values found in background_color. Defaults to 1.0.
            pad (float, optional): Padding value to adjust the axis limits. Defaults to 0.3.

        Returns:
            Plotter: A Plotter object configured with the given parameters.
        """
        log_header("Loading plotter")

        # Initialize and return a Plotter object
        return Plotter(
            graph,
            figsize=figsize,
            background_color=background_color,
            background_alpha=background_alpha,
            pad=pad,
        )
