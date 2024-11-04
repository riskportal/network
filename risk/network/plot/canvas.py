"""
risk/network/plot/canvas
~~~~~~~~~~~~~~~~~~~~~~~~
"""

from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from risk.log import params
from risk.network.graph import NetworkGraph
from risk.network.plot.utils.colors import to_rgba
from risk.network.plot.utils.layout import calculate_bounding_box


class Canvas:
    """A class for laying out the canvas in a network graph."""

    def __init__(self, graph: NetworkGraph, ax: plt.Axes) -> None:
        """Initialize the Canvas with a NetworkGraph and axis for plotting.

        Args:
            graph (NetworkGraph): The NetworkGraph object containing the network data.
            ax (plt.Axes): The axis to plot the canvas on.
        """
        self.graph = graph
        self.ax = ax

    def plot_title(
        self,
        title: Union[str, None] = None,
        subtitle: Union[str, None] = None,
        title_fontsize: int = 20,
        subtitle_fontsize: int = 14,
        font: str = "Arial",
        title_color: Union[str, List, Tuple, np.ndarray] = "black",
        subtitle_color: Union[str, List, Tuple, np.ndarray] = "gray",
        title_x: float = 0.5,
        title_y: float = 0.975,
        title_space_offset: float = 0.075,
        subtitle_offset: float = 0.025,
    ) -> None:
        """Plot title and subtitle on the network graph with customizable parameters.

        Args:
            title (str, optional): Title of the plot. Defaults to None.
            subtitle (str, optional): Subtitle of the plot. Defaults to None.
            title_fontsize (int, optional): Font size for the title. Defaults to 20.
            subtitle_fontsize (int, optional): Font size for the subtitle. Defaults to 14.
            font (str, optional): Font family used for both title and subtitle. Defaults to "Arial".
            title_color (str, List, Tuple, or np.ndarray, optional): Color of the title text. Can be a string or an array of colors.
                Defaults to "black".
            subtitle_color (str, List, Tuple, or np.ndarray, optional): Color of the subtitle text. Can be a string or an array of colors.
                Defaults to "gray".
            title_x (float, optional): X-axis position of the title. Defaults to 0.5.
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
                x=title_x,
                ha="center",
                va="top",
                y=title_y,
            )

        # Plot subtitle if provided
        if subtitle:
            # Calculate the subtitle's y position based on the midpoint of the title and subtitle_offset
            # Calculate the approximate height of the title in relative axis units
            title_height = title_fontsize / fig.bbox.height
            # Position the subtitle relative to the title's center (title_y - half the title height)
            subtitle_y_position = title_y - (title_height / 2) - subtitle_offset
            self.ax.figure.text(
                0.5,  # Ensure horizontal centering for subtitle
                subtitle_y_position,  # Position subtitle based on the center of the title
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
        center_offset_x: float = 0.0,
        center_offset_y: float = 0.0,
        linestyle: str = "dashed",
        linewidth: float = 1.5,
        color: Union[str, List, Tuple, np.ndarray] = "black",
        outline_alpha: Union[float, None] = 1.0,
        fill_alpha: Union[float, None] = 0.0,
    ) -> None:
        """Plot a circle around the network graph to represent the network perimeter.

        Args:
            scale (float, optional): Scaling factor for the perimeter diameter. Defaults to 1.0.
            center_offset_x (float, optional): Horizontal offset as a fraction of the diameter.
                Negative values shift the center left, positive values shift it right. Defaults to 0.0.
            center_offset_y (float, optional): Vertical offset as a fraction of the diameter.
                Negative values shift the center down, positive values shift it up. Defaults to 0.0.
            linestyle (str, optional): Line style for the network perimeter circle (e.g., dashed, solid). Defaults to "dashed".
            linewidth (float, optional): Width of the circle's outline. Defaults to 1.5.
            color (str, List, Tuple, or np.ndarray, optional): Color of the network perimeter circle. Defaults to "black".
            outline_alpha (float, None, optional): Transparency level of the circle outline. If provided, it overrides any existing alpha
                values found in color. Defaults to 1.0.
            fill_alpha (float, None, optional): Transparency level of the circle fill. If provided, it overrides any existing alpha values
                found in color. Defaults to 0.0.
        """
        # Log the circle perimeter plotting parameters
        params.log_plotter(
            perimeter_type="circle",
            perimeter_scale=scale,
            perimeter_center_offset_x=center_offset_x,
            perimeter_center_offset_y=center_offset_y,
            perimeter_linestyle=linestyle,
            perimeter_linewidth=linewidth,
            perimeter_color=(
                "custom" if isinstance(color, (list, tuple, np.ndarray)) else color
            ),  # np.ndarray usually indicates custom colors
            perimeter_outline_alpha=outline_alpha,
            perimeter_fill_alpha=fill_alpha,
        )

        # Extract node coordinates from the network graph
        node_coordinates = self.graph.node_coordinates
        # Calculate the center and radius of the bounding box around the network
        center, radius = calculate_bounding_box(node_coordinates)
        # Adjust the center based on user-defined offsets
        adjusted_center = _calculate_adjusted_center(
            center, radius, center_offset_x, center_offset_y
        )
        # Scale the radius by the scale factor
        scaled_radius = radius * scale

        # Convert color to RGBA using the to_rgba helper function - use outline_alpha for the perimeter
        outline_color_rgba = to_rgba(
            color=color, alpha=outline_alpha, num_repeats=1
        )  # num_repeats=1 for a single color
        fill_color_rgba = to_rgba(
            color=color, alpha=fill_alpha, num_repeats=1
        )  # num_repeats=1 for a single color

        # Draw a circle to represent the network perimeter
        circle = plt.Circle(
            adjusted_center,
            scaled_radius,
            linestyle=linestyle,
            linewidth=linewidth,
            color=outline_color_rgba,
        )
        # Set the transparency of the fill if applicable
        circle.set_facecolor(
            to_rgba(color=fill_color_rgba, num_repeats=1)
        )  # num_repeats=1 for a single color

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
            color (str, List, Tuple, or np.ndarray, optional): Color of the network perimeter contour. Defaults to "black".
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

        # Convert color to RGBA using outline_alpha for the line (outline)
        outline_color_rgba = to_rgba(color=color, num_repeats=1)  # num_repeats=1 for a single color
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
            color=outline_color_rgba,
            linestyle=linestyle,
            linewidth=linewidth,
            fill_alpha=fill_alpha,
        )


def _calculate_adjusted_center(
    center: Tuple[float, float],
    radius: float,
    center_offset_x: float = 0.0,
    center_offset_y: float = 0.0,
) -> Tuple[float, float]:
    """Calculate the adjusted center for the network perimeter circle based on user-defined offsets.

    Args:
        center (Tuple[float, float]): Original center coordinates of the network graph.
        radius (float): Radius of the bounding box around the network.
        center_offset_x (float, optional): Horizontal offset as a fraction of the diameter.
            Negative values shift the center left, positive values shift it right. Allowed
            values are in the range [-1, 1]. Defaults to 0.0.
        center_offset_y (float, optional): Vertical offset as a fraction of the diameter.
            Negative values shift the center down, positive values shift it up. Allowed
            values are in the range [-1, 1]. Defaults to 0.0.

    Returns:
        Tuple[float, float]: Adjusted center coordinates after applying the offsets.

    Raises:
        ValueError: If the center offsets are outside the valid range [-1, 1].
    """
    # Flip the y-axis to match the plot orientation
    flipped_center_offset_y = -center_offset_y
    # Validate the center offsets
    if not -1 <= center_offset_x <= 1:
        raise ValueError("Horizontal center offset must be in the range [-1, 1].")
    if not -1 <= center_offset_y <= 1:
        raise ValueError("Vertical center offset must be in the range [-1, 1].")

    # Calculate adjusted center by applying offset fractions of the diameter
    adjusted_center_x = center[0] + (center_offset_x * radius * 2)
    adjusted_center_y = center[1] + (flipped_center_offset_y * radius * 2)

    # Return the adjusted center coordinates
    return adjusted_center_x, adjusted_center_y
