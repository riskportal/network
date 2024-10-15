"""
risk/network/plot/plotter
~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from risk.log import params
from risk.network.graph import NetworkGraph
from risk.network.plot.contour import Contour
from risk.network.plot.labels import Labels
from risk.network.plot.network import Network
from risk.network.plot.utils import calculate_bounding_box, to_rgba


class NetworkPlotter(Network, Contour, Labels):
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
    ) -> None:
        """Initialize the NetworkPlotter with a NetworkGraph object and plotting parameters.

        Args:
            graph (NetworkGraph): The network data and attributes to be visualized.
            figsize (tuple, optional): Size of the figure in inches (width, height). Defaults to (10, 10).
            background_color (str, list, tuple, np.ndarray, optional): Background color of the plot. Defaults to "white".
            background_alpha (float, None, optional): Transparency level of the background color. If provided, it overrides
                any existing alpha values found in background_color. Defaults to 1.0.
        """
        self.graph = graph
        # Initialize the plot with the specified parameters
        self.ax = self._initialize_plot(
            graph=graph,
            figsize=figsize,
            background_color=background_color,
            background_alpha=background_alpha,
        )
        super().__init__(graph=graph, ax=self.ax)

    def _initialize_plot(
        self,
        graph: NetworkGraph,
        figsize: Tuple,
        background_color: Union[str, List, Tuple, np.ndarray],
        background_alpha: Union[float, None],
    ) -> plt.Axes:
        """Set up the plot with figure size and background color.

        Args:
            graph (NetworkGraph): The network data and attributes to be visualized.
            figsize (tuple): Size of the figure in inches (width, height).
            background_color (str): Background color of the plot.
            background_alpha (float, None, optional): Transparency level of the background color. If provided, it overrides any
            existing alpha values found in background_color.

        Returns:
            plt.Axes: The axis object for the plot.
        """
        # Extract node coordinates from the network graph
        node_coordinates = graph.node_coordinates
        # Calculate the center and radius of the bounding box around the network
        center, radius = calculate_bounding_box(node_coordinates)

        # Create a new figure and axis for plotting
        fig, ax = plt.subplots(figsize=figsize)
        fig.tight_layout()  # Adjust subplot parameters to give specified padding
        # Set axis limits based on the calculated bounding box and radius
        ax.set_xlim([center[0] - radius - 0.3, center[0] + radius + 0.3])
        ax.set_ylim([center[1] - radius - 0.3, center[1] + radius + 0.3])
        ax.set_aspect("equal")  # Ensure the aspect ratio is equal

        # Set the background color of the plot
        # Convert color to RGBA using the to_rgba helper function
        fig.patch.set_facecolor(to_rgba(color=background_color, alpha=background_alpha))
        ax.invert_yaxis()  # Invert the y-axis to match typical image coordinates
        # Remove axis spines for a cleaner look
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Hide axis ticks and labels
        ax.set_xticks([])
        ax.set_yticks([])
        ax.patch.set_visible(False)  # Hide the axis background

        return ax

    def plot_title(
        self,
        title: Union[str, None] = None,
        subtitle: Union[str, None] = None,
        title_fontsize: int = 20,
        subtitle_fontsize: int = 14,
        font: str = "Arial",
        title_color: str = "black",
        subtitle_color: str = "gray",
        title_y: float = 0.975,
        title_space_offset: float = 0.075,
        subtitle_offset: float = 0.025,
    ) -> None:
        """Plot title and subtitle on the network graph with customizable parameters.

        Args:
            title (str, optional): Title of the plot. Defaults to None.
            subtitle (str, optional): Subtitle of the plot. Defaults to None.
            title_fontsize (int, optional): Font size for the title. Defaults to 16.
            subtitle_fontsize (int, optional): Font size for the subtitle. Defaults to 12.
            font (str, optional): Font family used for both title and subtitle. Defaults to "Arial".
            title_color (str, optional): Color of the title text. Defaults to "black".
            subtitle_color (str, optional): Color of the subtitle text. Defaults to "gray".
            title_y (float, optional): Y-axis position of the title. Defaults to 0.975.
            title_space_offset (float, optional): Fraction of figure height to leave for the space above the plot. Defaults to 0.075.
            subtitle_offset (float, optional): Offset factor to position the subtitle below the title. Defaults to 0.025.
        """
        # Log the title and subtitle parameters
        params.log_plotter(
            title=title,
            subtitle=subtitle,
            title_fontsize=title_fontsize,
            subtitle_fontsize=subtitle_fontsize,
            title_subtitle_font=font,
            title_color=title_color,
            subtitle_color=subtitle_color,
            subtitle_offset=subtitle_offset,
            title_y=title_y,
            title_space_offset=title_space_offset,
        )

        # Get the current figure and axis dimensions
        fig = self.ax.figure
        # Use a tight layout to ensure that title and subtitle do not overlap with the original plot
        fig.tight_layout(
            rect=[0, 0, 1, 1 - title_space_offset]
        )  # Leave space above the plot for title

        # Plot title if provided
        if title:
            # Set the title using figure's suptitle to ensure centering
            self.ax.figure.suptitle(
                title,
                fontsize=title_fontsize,
                color=title_color,
                fontname=font,
                x=0.5,  # Center the title horizontally
                ha="center",
                va="top",
                y=title_y,
            )

        # Plot subtitle if provided
        if subtitle:
            # Calculate the subtitle's y position based on title's position and subtitle_offset
            subtitle_y_position = title_y - subtitle_offset
            self.ax.figure.text(
                0.5,  # Ensure horizontal centering for subtitle
                subtitle_y_position,
                subtitle,
                ha="center",
                va="top",
                fontname=font,
                fontsize=subtitle_fontsize,
                color=subtitle_color,
            )

    def plot_circle_perimeter(
        self,
        scale: float = 1.0,
        linestyle: str = "dashed",
        linewidth: float = 1.5,
        color: Union[str, List, Tuple, np.ndarray] = "black",
        outline_alpha: Union[float, None] = 1.0,
        fill_alpha: Union[float, None] = 0.0,
    ) -> None:
        """Plot a circle around the network graph to represent the network perimeter.

        Args:
            scale (float, optional): Scaling factor for the perimeter diameter. Defaults to 1.0.
            linestyle (str, optional): Line style for the network perimeter circle (e.g., dashed, solid). Defaults to "dashed".
            linewidth (float, optional): Width of the circle's outline. Defaults to 1.5.
            color (str, list, tuple, or np.ndarray, optional): Color of the network perimeter circle. Defaults to "black".
            outline_alpha (float, None, optional): Transparency level of the circle outline. If provided, it overrides any existing alpha
                values found in color. Defaults to 1.0.
            fill_alpha (float, None, optional): Transparency level of the circle fill. If provided, it overrides any existing alpha values
                found in color. Defaults to 0.0.
        """
        # Log the circle perimeter plotting parameters
        params.log_plotter(
            perimeter_type="circle",
            perimeter_scale=scale,
            perimeter_linestyle=linestyle,
            perimeter_linewidth=linewidth,
            perimeter_color=(
                "custom" if isinstance(color, (list, tuple, np.ndarray)) else color
            ),  # np.ndarray usually indicates custom colors
            perimeter_outline_alpha=outline_alpha,
            perimeter_fill_alpha=fill_alpha,
        )

        # Convert color to RGBA using the to_rgba helper function - use outline_alpha for the perimeter
        color = to_rgba(color=color, alpha=outline_alpha)
        # Set the fill_alpha to 0 if not provided
        fill_alpha = fill_alpha if fill_alpha is not None else 0.0
        # Extract node coordinates from the network graph
        node_coordinates = self.graph.node_coordinates
        # Calculate the center and radius of the bounding box around the network
        center, radius = calculate_bounding_box(node_coordinates)
        # Scale the radius by the scale factor
        scaled_radius = radius * scale

        # Draw a circle to represent the network perimeter
        circle = plt.Circle(
            center,
            scaled_radius,
            linestyle=linestyle,
            linewidth=linewidth,
            color=color,
            fill=fill_alpha > 0,  # Fill the circle if fill_alpha is greater than 0
        )
        # Set the transparency of the fill if applicable
        if fill_alpha > 0:
            circle.set_facecolor(to_rgba(color=color, alpha=fill_alpha))

        self.ax.add_artist(circle)

    def plot_contour_perimeter(
        self,
        scale: float = 1.0,
        levels: int = 3,
        bandwidth: float = 0.8,
        grid_size: int = 250,
        color: Union[str, List, Tuple, np.ndarray] = "black",
        linestyle: str = "solid",
        linewidth: float = 1.5,
        outline_alpha: Union[float, None] = 1.0,
        fill_alpha: Union[float, None] = 0.0,
    ) -> None:
        """
        Plot a KDE-based contour around the network graph to represent the network perimeter.

        Args:
            scale (float, optional): Scaling factor for the perimeter size. Defaults to 1.0.
            levels (int, optional): Number of contour levels. Defaults to 3.
            bandwidth (float, optional): Bandwidth for the KDE. Controls smoothness. Defaults to 0.8.
            grid_size (int, optional): Grid resolution for the KDE. Higher values yield finer contours. Defaults to 250.
            color (str, list, tuple, or np.ndarray, optional): Color of the network perimeter contour. Defaults to "black".
            linestyle (str, optional): Line style for the network perimeter contour (e.g., dashed, solid). Defaults to "solid".
            linewidth (float, optional): Width of the contour's outline. Defaults to 1.5.
            outline_alpha (float, None, optional): Transparency level of the contour outline. If provided, it overrides any existing
                alpha values found in color. Defaults to 1.0.
            fill_alpha (float, None, optional): Transparency level of the contour fill. If provided, it overrides any existing alpha
                values found in color. Defaults to 0.0.
        """
        # Log the contour perimeter plotting parameters
        params.log_plotter(
            perimeter_type="contour",
            perimeter_scale=scale,
            perimeter_levels=levels,
            perimeter_bandwidth=bandwidth,
            perimeter_grid_size=grid_size,
            perimeter_linestyle=linestyle,
            perimeter_linewidth=linewidth,
            perimeter_color=("custom" if isinstance(color, (list, tuple, np.ndarray)) else color),
            perimeter_outline_alpha=outline_alpha,
            perimeter_fill_alpha=fill_alpha,
        )

        # Convert color to RGBA using the to_rgba helper function - use outline_alpha for the perimeter
        color = to_rgba(color=color, alpha=outline_alpha)
        # Extract node coordinates from the network graph
        node_coordinates = self.graph.node_coordinates
        # Scale the node coordinates if needed
        scaled_coordinates = node_coordinates * scale
        # Use the existing _draw_kde_contour method
        self._draw_kde_contour(
            ax=self.ax,
            pos=scaled_coordinates,
            nodes=list(range(len(node_coordinates))),  # All nodes are included
            levels=levels,
            bandwidth=bandwidth,
            grid_size=grid_size,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=fill_alpha,
        )

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
