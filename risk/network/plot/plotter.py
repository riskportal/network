"""
risk/network/plot/plotter
~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from risk.log import params
from risk.network.graph import NetworkGraph
from risk.network.plot.canvas import Canvas
from risk.network.plot.contour import Contour
from risk.network.plot.labels import Labels
from risk.network.plot.network import Network
from risk.network.plot.utils.colors import to_rgba
from risk.network.plot.utils.layout import calculate_bounding_box


class NetworkPlotter(Canvas, Network, Contour, Labels):
    """A class for visualizing network graphs with customizable options.

    The NetworkPlotter class uses a NetworkGraph object and provides methods to plot the network with
    flexible node and edge properties. It also supports plotting labels, contours, drawing the network's
    perimeter, and adjusting background colors.
    """

    def __init__(
        self,
        graph: NetworkGraph,
        figsize: Tuple = (10, 10),
        background_color: Union[str, List, Tuple, np.ndarray] = "white",
        background_alpha: Union[float, None] = 1.0,
        pad: float = 0.3,
    ) -> None:
        """Initialize the NetworkPlotter with a NetworkGraph object and plotting parameters.

        Args:
            graph (NetworkGraph): The network data and attributes to be visualized.
            figsize (Tuple, optional): Size of the figure in inches (width, height). Defaults to (10, 10).
            background_color (str, List, Tuple, np.ndarray, optional): Background color of the plot. Defaults to "white".
            background_alpha (float, None, optional): Transparency level of the background color. If provided, it overrides
                any existing alpha values found in background_color. Defaults to 1.0.
            pad (float, optional): Padding value to adjust the axis limits. Defaults to 0.3.
        """
        self.graph = graph
        # Initialize the plot with the specified parameters
        self.ax = self._initialize_plot(
            graph=graph,
            figsize=figsize,
            background_color=background_color,
            background_alpha=background_alpha,
            pad=pad,
        )
        super().__init__(graph=graph, ax=self.ax)

    def _initialize_plot(
        self,
        graph: NetworkGraph,
        figsize: Tuple,
        background_color: Union[str, List, Tuple, np.ndarray],
        background_alpha: Union[float, None],
        pad: float,
    ) -> plt.Axes:
        """Set up the plot with figure size and background color.

        Args:
            graph (NetworkGraph): The network data and attributes to be visualized.
            figsize (Tuple): Size of the figure in inches (width, height).
            background_color (str, List, Tuple, or np.ndarray): Background color of the plot. Can be a single color or an array of colors.
            background_alpha (float, None, optional): Transparency level of the background color. If provided, it overrides any existing
                alpha values found in `background_color`.
            pad (float, optional): Padding value to adjust the axis limits.

        Returns:
            plt.Axes: The axis object for the plot.
        """
        # Log the plotter settings
        params.log_plotter(
            figsize=figsize,
            background_color=background_color,
            background_alpha=background_alpha,
            pad=pad,
        )

        # Extract node coordinates from the network graph
        node_coordinates = graph.node_coordinates
        # Calculate the center and radius of the bounding box around the network
        center, radius = calculate_bounding_box(node_coordinates)

        # Create a new figure and axis for plotting
        fig, ax = plt.subplots(figsize=figsize)
        fig.tight_layout()  # Adjust subplot parameters to give specified padding
        # Set axis limits based on the calculated bounding box and radius
        ax.set_xlim([center[0] - radius - pad, center[0] + radius + pad])
        ax.set_ylim([center[1] - radius - pad, center[1] + radius + pad])
        ax.set_aspect("equal")  # Ensure the aspect ratio is equal

        # Set the background color of the plot
        # Convert color to RGBA using the to_rgba helper function
        fig.patch.set_facecolor(
            to_rgba(color=background_color, alpha=background_alpha, num_repeats=1)
        )  # num_repeats=1 for single color
        ax.invert_yaxis()  # Invert the y-axis to match typical image coordinates
        # Remove axis spines for a cleaner look
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Hide axis ticks and labels
        ax.set_xticks([])
        ax.set_yticks([])
        ax.patch.set_visible(False)  # Hide the axis background

        return ax

    @staticmethod
    def savefig(*args, pad_inches: float = 0.5, dpi: int = 100, **kwargs) -> None:
        """Save the current plot to a file with additional export options.

        Args:
            *args: Positional arguments passed to `plt.savefig`.
            pad_inches (float, optional): Padding around the figure when saving. Defaults to 0.5.
            dpi (int, optional): Dots per inch (DPI) for the exported image. Defaults to 300.
            **kwargs: Keyword arguments passed to `plt.savefig`, such as filename and format.
        """
        plt.savefig(*args, bbox_inches="tight", pad_inches=pad_inches, dpi=dpi, **kwargs)

    @staticmethod
    def show(*args, **kwargs) -> None:
        """Display the current plot.

        Args:
            *args: Positional arguments passed to `plt.show`.
            **kwargs: Keyword arguments passed to `plt.show`.
        """
        plt.show(*args, **kwargs)
