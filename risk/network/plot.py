"""
risk/network/plot
~~~~~~~~~~~~~~~~~
"""

from typing import Any, Dict, List, Tuple, Union

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy import linalg
from scipy.ndimage import label
from scipy.stats import gaussian_kde

from risk.log import params, logger
from risk.network.graph import NetworkGraph


class NetworkPlotter:
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
    ) -> None:
        """Initialize the NetworkPlotter with a NetworkGraph object and plotting parameters.

        Args:
            graph (NetworkGraph): The network data and attributes to be visualized.
            figsize (tuple, optional): Size of the figure in inches (width, height). Defaults to (10, 10).
            background_color (str, list, tuple, np.ndarray, optional): Background color of the plot. Defaults to "white".
        """
        self.graph = graph
        # Initialize the plot with the specified parameters
        self.ax = self._initialize_plot(graph, figsize, background_color)

    def _initialize_plot(
        self,
        graph: NetworkGraph,
        figsize: Tuple,
        background_color: Union[str, List, Tuple, np.ndarray],
    ) -> plt.Axes:
        """Set up the plot with figure size and background color.

        Args:
            graph (NetworkGraph): The network data and attributes to be visualized.
            figsize (tuple): Size of the figure in inches (width, height).
            background_color (str): Background color of the plot.

        Returns:
            plt.Axes: The axis object for the plot.
        """
        # Extract node coordinates from the network graph
        node_coordinates = graph.node_coordinates
        # Calculate the center and radius of the bounding box around the network
        center, radius = _calculate_bounding_box(node_coordinates)

        # Create a new figure and axis for plotting
        fig, ax = plt.subplots(figsize=figsize)
        fig.tight_layout()  # Adjust subplot parameters to give specified padding
        # Set axis limits based on the calculated bounding box and radius
        ax.set_xlim([center[0] - radius - 0.3, center[0] + radius + 0.3])
        ax.set_ylim([center[1] - radius - 0.3, center[1] + radius + 0.3])
        ax.set_aspect("equal")  # Ensure the aspect ratio is equal

        # Set the background color of the plot
        # Convert color to RGBA using the _to_rgba helper function
        fig.patch.set_facecolor(_to_rgba(background_color, 1.0))
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
        outline_alpha: float = 1.0,
        fill_alpha: float = 0.0,
    ) -> None:
        """Plot a circle around the network graph to represent the network perimeter.

        Args:
            scale (float, optional): Scaling factor for the perimeter diameter. Defaults to 1.0.
            linestyle (str, optional): Line style for the network perimeter circle (e.g., dashed, solid). Defaults to "dashed".
            linewidth (float, optional): Width of the circle's outline. Defaults to 1.5.
            color (str, list, tuple, or np.ndarray, optional): Color of the network perimeter circle. Defaults to "black".
            outline_alpha (float, optional): Transparency level of the circle outline. Defaults to 1.0.
            fill_alpha (float, optional): Transparency level of the circle fill. Defaults to 0.0.
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

        # Convert color to RGBA using the _to_rgba helper function - use outline_alpha for the perimeter
        color = _to_rgba(color, outline_alpha)
        # Extract node coordinates from the network graph
        node_coordinates = self.graph.node_coordinates
        # Calculate the center and radius of the bounding box around the network
        center, radius = _calculate_bounding_box(node_coordinates)
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
            circle.set_facecolor(_to_rgba(color, fill_alpha))

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
        outline_alpha: float = 1.0,
        fill_alpha: float = 0.0,
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
            outline_alpha (float, optional): Transparency level of the contour outline. Defaults to 1.0.
            fill_alpha (float, optional): Transparency level of the contour fill. Defaults to 0.0.
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

        # Convert color to RGBA using the _to_rgba helper function - use outline_alpha for the perimeter
        color = _to_rgba(color, outline_alpha)
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

    def plot_network(
        self,
        node_size: Union[int, np.ndarray] = 50,
        node_shape: str = "o",
        node_edgewidth: float = 1.0,
        edge_width: float = 1.0,
        node_color: Union[str, List, Tuple, np.ndarray] = "white",
        node_edgecolor: Union[str, List, Tuple, np.ndarray] = "black",
        edge_color: Union[str, List, Tuple, np.ndarray] = "black",
        node_alpha: float = 1.0,
        edge_alpha: float = 1.0,
    ) -> None:
        """Plot the network graph with customizable node colors, sizes, edge widths, and node edge widths.

        Args:
            node_size (int or np.ndarray, optional): Size of the nodes. Can be a single integer or an array of sizes. Defaults to 50.
            node_shape (str, optional): Shape of the nodes. Defaults to "o".
            node_edgewidth (float, optional): Width of the node edges. Defaults to 1.0.
            edge_width (float, optional): Width of the edges. Defaults to 1.0.
            node_color (str, list, tuple, or np.ndarray, optional): Color of the nodes. Can be a single color or an array of colors. Defaults to "white".
            node_edgecolor (str, list, tuple, or np.ndarray, optional): Color of the node edges. Defaults to "black".
            edge_color (str, list, tuple, or np.ndarray, optional): Color of the edges. Defaults to "black".
            node_alpha (float, optional): Alpha value (transparency) for the nodes. Defaults to 1.0. Annotated node_color alphas will override this value.
            edge_alpha (float, optional): Alpha value (transparency) for the edges. Defaults to 1.0.
        """
        # Log the plotting parameters
        params.log_plotter(
            network_node_size=(
                "custom" if isinstance(node_size, np.ndarray) else node_size
            ),  # np.ndarray usually indicates custom sizes
            network_node_shape=node_shape,
            network_node_edgewidth=node_edgewidth,
            network_edge_width=edge_width,
            network_node_color=(
                "custom" if isinstance(node_color, np.ndarray) else node_color
            ),  # np.ndarray usually indicates custom colors
            network_node_edgecolor=node_edgecolor,
            network_edge_color=edge_color,
            network_node_alpha=node_alpha,
            network_edge_alpha=edge_alpha,
        )

        # Convert colors to RGBA using the _to_rgba helper function
        # If node_colors was generated using get_annotated_node_colors, its alpha values will override node_alpha
        node_color = _to_rgba(node_color, node_alpha, num_repeats=len(self.graph.network.nodes))
        node_edgecolor = _to_rgba(node_edgecolor, 1.0, num_repeats=len(self.graph.network.nodes))
        edge_color = _to_rgba(edge_color, edge_alpha, num_repeats=len(self.graph.network.edges))

        # Extract node coordinates from the network graph
        node_coordinates = self.graph.node_coordinates

        # Draw the nodes of the graph
        nx.draw_networkx_nodes(
            self.graph.network,
            pos=node_coordinates,
            node_size=node_size,
            node_shape=node_shape,
            node_color=node_color,
            edgecolors=node_edgecolor,
            linewidths=node_edgewidth,
            ax=self.ax,
        )
        # Draw the edges of the graph
        nx.draw_networkx_edges(
            self.graph.network,
            pos=node_coordinates,
            width=edge_width,
            edge_color=edge_color,
            ax=self.ax,
        )

    def plot_subnetwork(
        self,
        nodes: Union[List, Tuple, np.ndarray],
        node_size: Union[int, np.ndarray] = 50,
        node_shape: str = "o",
        node_edgewidth: float = 1.0,
        edge_width: float = 1.0,
        node_color: Union[str, List, Tuple, np.ndarray] = "white",
        node_edgecolor: Union[str, List, Tuple, np.ndarray] = "black",
        edge_color: Union[str, List, Tuple, np.ndarray] = "black",
        node_alpha: float = 1.0,
        edge_alpha: float = 1.0,
    ) -> None:
        """Plot a subnetwork of selected nodes with customizable node and edge attributes.

        Args:
            nodes (list, tuple, or np.ndarray): List of node labels to include in the subnetwork. Accepts nested lists.
            node_size (int or np.ndarray, optional): Size of the nodes. Can be a single integer or an array of sizes. Defaults to 50.
            node_shape (str, optional): Shape of the nodes. Defaults to "o".
            node_edgewidth (float, optional): Width of the node edges. Defaults to 1.0.
            edge_width (float, optional): Width of the edges. Defaults to 1.0.
            node_color (str, list, tuple, or np.ndarray, optional): Color of the nodes. Defaults to "white".
            node_edgecolor (str, list, tuple, or np.ndarray, optional): Color of the node edges. Defaults to "black".
            edge_color (str, list, tuple, or np.ndarray, optional): Color of the edges. Defaults to "black".
            node_alpha (float, optional): Transparency for the nodes. Defaults to 1.0.
            edge_alpha (float, optional): Transparency for the edges. Defaults to 1.0.

        Raises:
            ValueError: If no valid nodes are found in the network graph.
        """
        # Flatten nested lists of nodes, if necessary
        if any(isinstance(item, (list, tuple, np.ndarray)) for item in nodes):
            nodes = [node for sublist in nodes for node in sublist]

        # Filter to get node IDs and their coordinates
        node_ids = [
            self.graph.node_label_to_node_id_map.get(node)
            for node in nodes
            if node in self.graph.node_label_to_node_id_map
        ]
        if not node_ids:
            raise ValueError("No nodes found in the network graph.")

        # Check if node_color is a single color or a list of colors
        if not isinstance(node_color, (str, tuple, np.ndarray)):
            node_color = [
                node_color[nodes.index(node)]
                for node in nodes
                if node in self.graph.node_label_to_node_id_map
            ]

        # Convert colors to RGBA using the _to_rgba helper function
        node_color = _to_rgba(node_color, node_alpha, num_repeats=len(node_ids))
        node_edgecolor = _to_rgba(node_edgecolor, 1.0, num_repeats=len(node_ids))
        edge_color = _to_rgba(edge_color, edge_alpha, num_repeats=len(self.graph.network.edges))

        # Get the coordinates of the filtered nodes
        node_coordinates = {node_id: self.graph.node_coordinates[node_id] for node_id in node_ids}

        # Draw the nodes in the subnetwork
        nx.draw_networkx_nodes(
            self.graph.network,
            pos=node_coordinates,
            nodelist=node_ids,
            node_size=node_size,
            node_shape=node_shape,
            node_color=node_color,
            edgecolors=node_edgecolor,
            linewidths=node_edgewidth,
            ax=self.ax,
        )
        # Draw the edges between the specified nodes in the subnetwork
        subgraph = self.graph.network.subgraph(node_ids)
        nx.draw_networkx_edges(
            subgraph,
            pos=node_coordinates,
            width=edge_width,
            edge_color=edge_color,
            ax=self.ax,
        )

    def plot_contours(
        self,
        levels: int = 5,
        bandwidth: float = 0.8,
        grid_size: int = 250,
        color: Union[str, List, Tuple, np.ndarray] = "white",
        linestyle: str = "solid",
        linewidth: float = 1.5,
        alpha: float = 1.0,
        fill_alpha: float = 0.2,
    ) -> None:
        """Draw KDE contours for nodes in various domains of a network graph, highlighting areas of high density.

        Args:
            levels (int, optional): Number of contour levels to plot. Defaults to 5.
            bandwidth (float, optional): Bandwidth for KDE. Controls the smoothness of the contour. Defaults to 0.8.
            grid_size (int, optional): Resolution of the grid for KDE. Higher values create finer contours. Defaults to 250.
            color (str, list, tuple, or np.ndarray, optional): Color of the contours. Can be a single color or an array of colors. Defaults to "white".
            linestyle (str, optional): Line style for the contours. Defaults to "solid".
            linewidth (float, optional): Line width for the contours. Defaults to 1.5.
            alpha (float, optional): Transparency level of the contour lines. Defaults to 1.0.
            fill_alpha (float, optional): Transparency level of the contour fill. Defaults to 0.2.
        """
        # Log the contour plotting parameters
        params.log_plotter(
            contour_levels=levels,
            contour_bandwidth=bandwidth,
            contour_grid_size=grid_size,
            contour_color=(
                "custom" if isinstance(color, np.ndarray) else color
            ),  # np.ndarray usually indicates custom colors
            contour_alpha=alpha,
            contour_fill_alpha=fill_alpha,
        )

        # Ensure color is converted to RGBA with repetition matching the number of domains
        color = _to_rgba(color, alpha, num_repeats=len(self.graph.domain_id_to_node_ids_map))
        # Extract node coordinates from the network graph
        node_coordinates = self.graph.node_coordinates
        # Draw contours for each domain in the network
        for idx, (_, node_ids) in enumerate(self.graph.domain_id_to_node_ids_map.items()):
            if len(node_ids) > 1:
                self._draw_kde_contour(
                    self.ax,
                    node_coordinates,
                    node_ids,
                    color=color[idx],
                    levels=levels,
                    bandwidth=bandwidth,
                    grid_size=grid_size,
                    linestyle=linestyle,
                    linewidth=linewidth,
                    alpha=alpha,
                    fill_alpha=fill_alpha,
                )

    def plot_subcontour(
        self,
        nodes: Union[List, Tuple, np.ndarray],
        levels: int = 5,
        bandwidth: float = 0.8,
        grid_size: int = 250,
        color: Union[str, List, Tuple, np.ndarray] = "white",
        linestyle: str = "solid",
        linewidth: float = 1.5,
        alpha: float = 1.0,
        fill_alpha: float = 0.2,
    ) -> None:
        """Plot a subcontour for a given set of nodes or a list of node sets using Kernel Density Estimation (KDE).

        Args:
            nodes (list, tuple, or np.ndarray): List of node labels or list of lists of node labels to plot the contour for.
            levels (int, optional): Number of contour levels to plot. Defaults to 5.
            bandwidth (float, optional): Bandwidth for KDE. Controls the smoothness of the contour. Defaults to 0.8.
            grid_size (int, optional): Resolution of the grid for KDE. Higher values create finer contours. Defaults to 250.
            color (str, list, tuple, or np.ndarray, optional): Color of the contour. Can be a string (e.g., 'white') or RGBA array. Defaults to "white".
            linestyle (str, optional): Line style for the contour. Defaults to "solid".
            linewidth (float, optional): Line width for the contour. Defaults to 1.5.
            alpha (float, optional): Transparency level of the contour lines. Defaults to 1.0.
            fill_alpha (float, optional): Transparency level of the contour fill. Defaults to 0.2.

        Raises:
            ValueError: If no valid nodes are found in the network graph.
        """
        # Check if nodes is a list of lists or a flat list
        if any(isinstance(item, (list, tuple, np.ndarray)) for item in nodes):
            # If it's a list of lists, iterate over sublists
            node_groups = nodes
        else:
            # If it's a flat list of nodes, treat it as a single group
            node_groups = [nodes]

        # Convert color to RGBA using the _to_rgba helper function
        color_rgba = _to_rgba(color, alpha)

        # Iterate over each group of nodes (either sublists or flat list)
        for sublist in node_groups:
            # Filter to get node IDs and their coordinates for each sublist
            node_ids = [
                self.graph.node_label_to_node_id_map.get(node)
                for node in sublist
                if node in self.graph.node_label_to_node_id_map
            ]
            if not node_ids or len(node_ids) == 1:
                raise ValueError(
                    "No nodes found in the network graph or insufficient nodes to plot."
                )

            # Draw the KDE contour for the specified nodes
            node_coordinates = self.graph.node_coordinates
            self._draw_kde_contour(
                self.ax,
                node_coordinates,
                node_ids,
                color=color_rgba,
                levels=levels,
                bandwidth=bandwidth,
                grid_size=grid_size,
                linestyle=linestyle,
                linewidth=linewidth,
                alpha=alpha,
                fill_alpha=fill_alpha,
            )

    def _draw_kde_contour(
        self,
        ax: plt.Axes,
        pos: np.ndarray,
        nodes: List,
        levels: int = 5,
        bandwidth: float = 0.8,
        grid_size: int = 250,
        color: Union[str, np.ndarray] = "white",
        linestyle: str = "solid",
        linewidth: float = 1.5,
        alpha: float = 1.0,
        fill_alpha: float = 0.2,
    ) -> None:
        """Draw a Kernel Density Estimate (KDE) contour plot for a set of nodes on a given axis.

        Args:
            ax (plt.Axes): The axis to draw the contour on.
            pos (np.ndarray): Array of node positions (x, y).
            nodes (list): List of node indices to include in the contour.
            levels (int, optional): Number of contour levels. Defaults to 5.
            bandwidth (float, optional): Bandwidth for the KDE. Controls smoothness. Defaults to 0.8.
            grid_size (int, optional): Grid resolution for the KDE. Higher values yield finer contours. Defaults to 250.
            color (str or np.ndarray): Color for the contour. Can be a string or RGBA array. Defaults to "white".
            linestyle (str, optional): Line style for the contour. Defaults to "solid".
            linewidth (float, optional): Line width for the contour. Defaults to 1.5.
            alpha (float, optional): Transparency level for the contour lines. Defaults to 1.0.
            fill_alpha (float, optional): Transparency level for the contour fill. Defaults to 0.2.
        """
        # Extract the positions of the specified nodes
        points = np.array([pos[n] for n in nodes])
        if len(points) <= 1:
            return None  # Not enough points to form a contour

        # Check if the KDE forms a single connected component
        connected = False
        z = None  # Initialize z to None to avoid UnboundLocalError
        while not connected and bandwidth <= 100.0:
            try:
                # Perform KDE on the points with the given bandwidth
                kde = gaussian_kde(points.T, bw_method=bandwidth)
                xmin, ymin = points.min(axis=0) - bandwidth
                xmax, ymax = points.max(axis=0) + bandwidth
                x, y = np.mgrid[
                    xmin : xmax : complex(0, grid_size), ymin : ymax : complex(0, grid_size)
                ]
                z = kde(np.vstack([x.ravel(), y.ravel()])).reshape(x.shape)
                # Check if the KDE forms a single connected component
                connected = _is_connected(z)
                if not connected:
                    bandwidth += 0.05  # Increase bandwidth slightly and retry
            except linalg.LinAlgError:
                bandwidth += 0.05  # Increase bandwidth and retry
            except Exception as e:
                # Catch any other exceptions and log them
                logger.error(f"Unexpected error when drawing KDE contour: {e}")
                return None

        # If z is still None, the KDE computation failed
        if z is None:
            logger.error("Failed to compute KDE. Skipping contour plot for these nodes.")
            return None

        # Define contour levels based on the density
        min_density, max_density = z.min(), z.max()
        if min_density == max_density:
            logger.warning(
                "Contour levels could not be created due to lack of variation in density."
            )
            return None

        # Create contour levels based on the density values
        contour_levels = np.linspace(min_density, max_density, levels)[1:]
        if len(contour_levels) < 2 or not np.all(np.diff(contour_levels) > 0):
            logger.error("Contour levels must be strictly increasing. Skipping contour plot.")
            return None

        # Set the contour color and linestyle
        contour_colors = [color for _ in range(levels - 1)]
        # Plot the filled contours using fill_alpha for transparency
        if fill_alpha > 0:
            ax.contourf(
                x,
                y,
                z,
                levels=contour_levels,
                colors=contour_colors,
                antialiased=True,
                alpha=fill_alpha,
            )

        # Plot the contour lines with the specified alpha for transparency
        c = ax.contour(
            x,
            y,
            z,
            levels=contour_levels,
            colors=contour_colors,
            linestyles=linestyle,
            linewidths=linewidth,
            alpha=alpha,
        )

        # Set linewidth for the contour lines to 0 for levels other than the base level
        for i in range(1, len(contour_levels)):
            c.collections[i].set_linewidth(0)

    def plot_labels(
        self,
        scale: float = 1.05,
        offset: float = 0.10,
        font: str = "Arial",
        fontsize: int = 10,
        fontcolor: Union[str, List, Tuple, np.ndarray] = "black",
        fontalpha: float = 1.0,
        arrow_linewidth: float = 1,
        arrow_style: str = "->",
        arrow_color: Union[str, List, Tuple, np.ndarray] = "black",
        arrow_alpha: float = 1.0,
        arrow_base_shrink: float = 0.0,
        arrow_tip_shrink: float = 0.0,
        max_labels: Union[int, None] = None,
        max_words: int = 10,
        min_words: int = 1,
        max_word_length: int = 20,
        min_word_length: int = 1,
        words_to_omit: Union[List, None] = None,
        overlay_ids: bool = False,
        ids_to_keep: Union[List, Tuple, np.ndarray, None] = None,
        ids_to_replace: Union[Dict, None] = None,
    ) -> None:
        """Annotate the network graph with labels for different domains, positioned around the network for clarity.

        Args:
            scale (float, optional): Scale factor for positioning labels around the perimeter. Defaults to 1.05.
            offset (float, optional): Offset distance for labels from the perimeter. Defaults to 0.10.
            font (str, optional): Font name for the labels. Defaults to "Arial".
            fontsize (int, optional): Font size for the labels. Defaults to 10.
            fontcolor (str, list, tuple, or np.ndarray, optional): Color of the label text. Can be a string or RGBA array. Defaults to "black".
            fontalpha (float, optional): Transparency level for the font color. Defaults to 1.0.
            arrow_linewidth (float, optional): Line width of the arrows pointing to centroids. Defaults to 1.
            arrow_style (str, optional): Style of the arrows pointing to centroids. Defaults to "->".
            arrow_color (str, list, tuple, or np.ndarray, optional): Color of the arrows. Defaults to "black".
            arrow_alpha (float, optional): Transparency level for the arrow color. Defaults to 1.0.
            arrow_base_shrink (float, optional): Distance between the text and the base of the arrow. Defaults to 0.0.
            arrow_tip_shrink (float, optional): Distance between the arrow tip and the centroid. Defaults to 0.0.
            max_labels (int, optional): Maximum number of labels to plot. Defaults to None (no limit).
            max_words (int, optional): Maximum number of words in a label. Defaults to 10.
            min_words (int, optional): Minimum number of words required to display a label. Defaults to 1.
            max_word_length (int, optional): Maximum number of characters in a word to display. Defaults to 20.
            min_word_length (int, optional): Minimum number of characters in a word to display. Defaults to 1.
            words_to_omit (list, optional): List of words to omit from the labels. Defaults to None.
            overlay_ids (bool, optional): Whether to overlay domain IDs in the center of the centroids. Defaults to False.
            ids_to_keep (list, tuple, np.ndarray, or None, optional): IDs of domains that must be labeled. To discover domain IDs,
                you can set `overlay_ids=True`. Defaults to None.
            ids_to_replace (dict, optional): A dictionary mapping domain IDs to custom labels (strings). The labels should be space-separated words.
                If provided, the custom labels will replace the default domain terms. To discover domain IDs, you can set `overlay_ids=True`.
                Defaults to None.

        Raises:
            ValueError: If the number of provided `ids_to_keep` exceeds `max_labels`.
        """
        # Log the plotting parameters
        params.log_plotter(
            label_perimeter_scale=scale,
            label_offset=offset,
            label_font=font,
            label_fontsize=fontsize,
            label_fontcolor=(
                "custom" if isinstance(fontcolor, np.ndarray) else fontcolor
            ),  # np.ndarray usually indicates custom colors
            label_fontalpha=fontalpha,
            label_arrow_linewidth=arrow_linewidth,
            label_arrow_style=arrow_style,
            label_arrow_color="custom" if isinstance(arrow_color, np.ndarray) else arrow_color,
            label_arrow_alpha=arrow_alpha,
            label_arrow_base_shrink=arrow_base_shrink,
            label_arrow_tip_shrink=arrow_tip_shrink,
            label_max_labels=max_labels,
            label_max_words=max_words,
            label_min_words=min_words,
            label_max_word_length=max_word_length,
            label_min_word_length=min_word_length,
            label_words_to_omit=words_to_omit,
            label_overlay_ids=overlay_ids,
            label_ids_to_keep=ids_to_keep,
            label_ids_to_replace=ids_to_replace,
        )

        # Set max_labels to the total number of domains if not provided (None)
        if max_labels is None:
            max_labels = len(self.graph.domain_id_to_node_ids_map)

        # Convert colors to RGBA using the _to_rgba helper function
        fontcolor = _to_rgba(
            fontcolor, fontalpha, num_repeats=len(self.graph.domain_id_to_node_ids_map)
        )
        arrow_color = _to_rgba(
            arrow_color, arrow_alpha, num_repeats=len(self.graph.domain_id_to_node_ids_map)
        )

        # Normalize words_to_omit to lowercase
        if words_to_omit:
            words_to_omit = set(word.lower() for word in words_to_omit)

        # Calculate the center and radius of the network
        domain_centroids = {}
        for domain_id, node_ids in self.graph.domain_id_to_node_ids_map.items():
            if node_ids:  # Skip if the domain has no nodes
                domain_centroids[domain_id] = self._calculate_domain_centroid(node_ids)

        # Initialize dictionaries and lists for valid indices
        valid_indices = []
        filtered_domain_centroids = {}
        filtered_domain_terms = {}
        # Handle the ids_to_keep logic
        if ids_to_keep:
            # Convert ids_to_keep to remove accidental duplicates
            ids_to_keep = set(ids_to_keep)
            # Check if the number of provided ids_to_keep exceeds max_labels
            if max_labels is not None and len(ids_to_keep) > max_labels:
                raise ValueError(
                    f"Number of provided IDs ({len(ids_to_keep)}) exceeds max_labels ({max_labels})."
                )

            # Process the specified IDs first
            for domain in ids_to_keep:
                if (
                    domain in self.graph.domain_id_to_domain_terms_map
                    and domain in domain_centroids
                ):
                    # Handle ids_to_replace logic here for ids_to_keep
                    if ids_to_replace and domain in ids_to_replace:
                        terms = ids_to_replace[domain].split(" ")
                    else:
                        terms = self.graph.domain_id_to_domain_terms_map[domain].split(" ")

                    # Apply words_to_omit, word length constraints, and max_words
                    if words_to_omit:
                        terms = [term for term in terms if term.lower() not in words_to_omit]
                    terms = [
                        term for term in terms if min_word_length <= len(term) <= max_word_length
                    ]
                    terms = terms[:max_words]

                    # Check if the domain passes the word count condition
                    if len(terms) >= min_words:
                        filtered_domain_centroids[domain] = domain_centroids[domain]
                        filtered_domain_terms[domain] = " ".join(terms)
                        valid_indices.append(
                            list(domain_centroids.keys()).index(domain)
                        )  # Track the valid index

        # Calculate remaining labels to plot after processing ids_to_keep
        remaining_labels = (
            max_labels - len(ids_to_keep) if ids_to_keep and max_labels else max_labels
        )
        # Process remaining domains to fill in additional labels, if there are slots left
        if remaining_labels and remaining_labels > 0:
            for idx, (domain, centroid) in enumerate(domain_centroids.items()):
                # Check if the domain is NaN and continue if true
                if pd.isna(domain) or (isinstance(domain, float) and np.isnan(domain)):
                    continue  # Skip NaN domains
                if ids_to_keep and domain in ids_to_keep:
                    continue  # Skip domains already handled by ids_to_keep

                # Handle ids_to_replace logic first
                if ids_to_replace and domain in ids_to_replace:
                    terms = ids_to_replace[domain].split(" ")
                else:
                    terms = self.graph.domain_id_to_domain_terms_map[domain].split(" ")

                # Apply words_to_omit, word length constraints, and max_words
                if words_to_omit:
                    terms = [term for term in terms if term.lower() not in words_to_omit]

                terms = [term for term in terms if min_word_length <= len(term) <= max_word_length]
                terms = terms[:max_words]
                # Check if the domain passes the word count condition
                if len(terms) >= min_words:
                    filtered_domain_centroids[domain] = centroid
                    filtered_domain_terms[domain] = " ".join(terms)
                    valid_indices.append(idx)  # Track the valid index

                # Stop once we've reached the max_labels limit
                if len(filtered_domain_centroids) >= max_labels:
                    break

        # Calculate the bounding box around the network
        center, radius = _calculate_bounding_box(self.graph.node_coordinates, radius_margin=scale)
        # Calculate the best positions for labels
        best_label_positions = _calculate_best_label_positions(
            filtered_domain_centroids, center, radius, offset
        )

        # Annotate the network with labels
        for idx, (domain, pos) in zip(valid_indices, best_label_positions.items()):
            centroid = filtered_domain_centroids[domain]
            annotations = filtered_domain_terms[domain].split(" ")[:max_words]
            self.ax.annotate(
                "\n".join(annotations),
                xy=centroid,
                xytext=pos,
                textcoords="data",
                ha="center",
                va="center",
                fontsize=fontsize,
                fontname=font,
                color=fontcolor[idx],
                arrowprops=dict(
                    arrowstyle=arrow_style,
                    color=arrow_color[idx],
                    linewidth=arrow_linewidth,
                    shrinkA=arrow_base_shrink,
                    shrinkB=arrow_tip_shrink,
                ),
            )
            # Overlay domain ID at the centroid if requested
            if overlay_ids:
                self.ax.text(
                    centroid[0],
                    centroid[1],
                    domain,
                    ha="center",
                    va="center",
                    fontsize=fontsize,
                    fontname=font,
                    color=fontcolor[idx],
                    alpha=fontalpha,
                )

    def plot_sublabel(
        self,
        nodes: Union[List, Tuple, np.ndarray],
        label: str,
        radial_position: float = 0.0,
        scale: float = 1.05,
        offset: float = 0.10,
        font: str = "Arial",
        fontsize: int = 10,
        fontcolor: Union[str, List, Tuple, np.ndarray] = "black",
        fontalpha: float = 1.0,
        arrow_linewidth: float = 1,
        arrow_style: str = "->",
        arrow_color: Union[str, List, Tuple, np.ndarray] = "black",
        arrow_alpha: float = 1.0,
        arrow_base_shrink: float = 0.0,
        arrow_tip_shrink: float = 0.0,
    ) -> None:
        """Annotate the network graph with a label for the given nodes, with one arrow pointing to each centroid of sublists of nodes.

        Args:
            nodes (list, tuple, or np.ndarray): List of node labels or list of lists of node labels.
            label (str): The label to be annotated on the network.
            radial_position (float, optional): Radial angle for positioning the label, in degrees (0-360). Defaults to 0.0.
            scale (float, optional): Scale factor for positioning the label around the perimeter. Defaults to 1.05.
            offset (float, optional): Offset distance for the label from the perimeter. Defaults to 0.10.
            font (str, optional): Font name for the label. Defaults to "Arial".
            fontsize (int, optional): Font size for the label. Defaults to 10.
            fontcolor (str, list, tuple, or np.ndarray, optional): Color of the label text. Defaults to "black".
            fontalpha (float, optional): Transparency level for the font color. Defaults to 1.0.
            arrow_linewidth (float, optional): Line width of the arrow pointing to the centroid. Defaults to 1.
            arrow_style (str, optional): Style of the arrows pointing to the centroid. Defaults to "->".
            arrow_color (str, list, tuple, or np.ndarray, optional): Color of the arrow. Defaults to "black".
            arrow_alpha (float, optional): Transparency level for the arrow color. Defaults to 1.0.
            arrow_base_shrink (float, optional): Distance between the text and the base of the arrow. Defaults to 0.0.
            arrow_tip_shrink (float, optional): Distance between the arrow tip and the centroid. Defaults to 0.0.
        """
        # Check if nodes is a list of lists or a flat list
        if any(isinstance(item, (list, tuple, np.ndarray)) for item in nodes):
            # If it's a list of lists, iterate over sublists
            node_groups = nodes
        else:
            # If it's a flat list of nodes, treat it as a single group
            node_groups = [nodes]

        # Convert fontcolor and arrow_color to RGBA
        fontcolor_rgba = _to_rgba(fontcolor, fontalpha)
        arrow_color_rgba = _to_rgba(arrow_color, arrow_alpha)

        # Calculate the bounding box around the network
        center, radius = _calculate_bounding_box(self.graph.node_coordinates, radius_margin=scale)
        # Convert radial position to radians, adjusting for a 90-degree rotation
        radial_radians = np.deg2rad(radial_position - 90)
        label_position = (
            center[0] + (radius + offset) * np.cos(radial_radians),
            center[1] + (radius + offset) * np.sin(radial_radians),
        )

        # Iterate over each group of nodes (either sublists or flat list)
        for sublist in node_groups:
            # Map node labels to IDs
            node_ids = [
                self.graph.node_label_to_node_id_map.get(node)
                for node in sublist
                if node in self.graph.node_label_to_node_id_map
            ]
            if not node_ids or len(node_ids) == 1:
                raise ValueError(
                    "No nodes found in the network graph or insufficient nodes to plot."
                )

            # Calculate the centroid of the provided nodes in this sublist
            centroid = self._calculate_domain_centroid(node_ids)
            # Annotate the network with the label and an arrow pointing to each centroid
            self.ax.annotate(
                label,
                xy=centroid,
                xytext=label_position,
                textcoords="data",
                ha="center",
                va="center",
                fontsize=fontsize,
                fontname=font,
                color=fontcolor_rgba,
                arrowprops=dict(
                    arrowstyle=arrow_style,
                    color=arrow_color_rgba,
                    linewidth=arrow_linewidth,
                    shrinkA=arrow_base_shrink,
                    shrinkB=arrow_tip_shrink,
                ),
            )

    def _calculate_domain_centroid(self, nodes: List) -> tuple:
        """Calculate the most centrally located node in .

        Args:
            nodes (list): List of node labels to include in the subnetwork.

        Returns:
            tuple: A tuple containing the domain's central node coordinates.
        """
        # Extract positions of all nodes in the domain
        node_positions = self.graph.node_coordinates[nodes, :]
        # Calculate the pairwise distance matrix between all nodes in the domain
        distances_matrix = np.linalg.norm(node_positions[:, np.newaxis] - node_positions, axis=2)
        # Sum the distances for each node to all other nodes in the domain
        sum_distances = np.sum(distances_matrix, axis=1)
        # Identify the node with the smallest total distance to others (the centroid)
        central_node_idx = np.argmin(sum_distances)
        # Map the domain to the coordinates of its central node
        domain_central_node = node_positions[central_node_idx]
        return domain_central_node

    def get_annotated_node_colors(
        self,
        cmap: str = "gist_rainbow",
        color: Union[str, None] = None,
        min_scale: float = 0.8,
        max_scale: float = 1.0,
        scale_factor: float = 1.0,
        alpha: float = 1.0,
        nonenriched_color: Union[str, List, Tuple, np.ndarray] = "white",
        nonenriched_alpha: float = 1.0,
        random_seed: int = 888,
    ) -> np.ndarray:
        """Adjust the colors of nodes in the network graph based on enrichment.

        Args:
            cmap (str, optional): Colormap to use for coloring the nodes. Defaults to "gist_rainbow".
            color (str or None, optional): Color to use for the nodes. If None, the colormap will be used. Defaults to None.
            min_scale (float, optional): Minimum scale for color intensity. Defaults to 0.8.
            max_scale (float, optional): Maximum scale for color intensity. Defaults to 1.0.
            scale_factor (float, optional): Factor for adjusting the color scaling intensity. Defaults to 1.0.
            alpha (float, optional): Alpha value for enriched nodes. Defaults to 1.0.
            nonenriched_color (str, list, tuple, or np.ndarray, optional): Color for non-enriched nodes. Defaults to "white".
            nonenriched_alpha (float, optional): Alpha value for non-enriched nodes. Defaults to 1.0.
            random_seed (int, optional): Seed for random number generation. Defaults to 888.

        Returns:
            np.ndarray: Array of RGBA colors adjusted for enrichment status.
        """
        # Get the initial domain colors for each node, which are returned as RGBA
        network_colors = self.graph.get_domain_colors(
            cmap=cmap,
            color=color,
            min_scale=min_scale,
            max_scale=max_scale,
            scale_factor=scale_factor,
            random_seed=random_seed,
        )
        # Apply the alpha value for enriched nodes
        network_colors[:, 3] = alpha  # Apply the alpha value to the enriched nodes' A channel
        # Convert the non-enriched color to RGBA using the _to_rgba helper function
        nonenriched_color = _to_rgba(nonenriched_color, nonenriched_alpha)
        # Adjust node colors: replace any fully black nodes (RGB == 0) with the non-enriched color and its alpha
        adjusted_network_colors = np.where(
            np.all(network_colors[:, :3] == 0, axis=1, keepdims=True),  # Check RGB values only
            np.array([nonenriched_color]),  # Apply the non-enriched color with alpha
            network_colors,  # Keep the original colors for enriched nodes
        )
        return adjusted_network_colors

    def get_annotated_node_sizes(
        self, enriched_size: int = 50, nonenriched_size: int = 25
    ) -> np.ndarray:
        """Adjust the sizes of nodes in the network graph based on whether they are enriched or not.

        Args:
            enriched_size (int): Size for enriched nodes. Defaults to 50.
            nonenriched_size (int): Size for non-enriched nodes. Defaults to 25.

        Returns:
            np.ndarray: Array of node sizes, with enriched nodes larger than non-enriched ones.
        """
        # Merge all enriched nodes from the domain_id_to_node_ids_map dictionary
        enriched_nodes = set()
        for _, node_ids in self.graph.domain_id_to_node_ids_map.items():
            enriched_nodes.update(node_ids)

        # Initialize all node sizes to the non-enriched size
        node_sizes = np.full(len(self.graph.network.nodes), nonenriched_size)
        # Set the size for enriched nodes
        for node in enriched_nodes:
            if node in self.graph.network.nodes:
                node_sizes[node] = enriched_size

        return node_sizes

    def get_annotated_contour_colors(
        self,
        cmap: str = "gist_rainbow",
        color: Union[str, None] = None,
        min_scale: float = 0.8,
        max_scale: float = 1.0,
        scale_factor: float = 1.0,
        random_seed: int = 888,
    ) -> np.ndarray:
        """Get colors for the contours based on node annotations or a specified colormap.

        Args:
            cmap (str, optional): Name of the colormap to use for generating contour colors. Defaults to "gist_rainbow".
            color (str or None, optional): Color to use for the contours. If None, the colormap will be used. Defaults to None.
            min_scale (float, optional): Minimum intensity scale for the colors generated by the colormap.
                Controls the dimmest colors. Defaults to 0.8.
            max_scale (float, optional): Maximum intensity scale for the colors generated by the colormap.
                Controls the brightest colors. Defaults to 1.0.
            scale_factor (float, optional): Exponent for adjusting color scaling based on enrichment scores.
                A higher value increases contrast by dimming lower scores more. Defaults to 1.0.
            random_seed (int, optional): Seed for random number generation to ensure reproducibility. Defaults to 888.

        Returns:
            np.ndarray: Array of RGBA colors for contour annotations.
        """
        return self._get_annotated_domain_colors(
            cmap=cmap,
            color=color,
            min_scale=min_scale,
            max_scale=max_scale,
            scale_factor=scale_factor,
            random_seed=random_seed,
        )

    def get_annotated_label_colors(
        self,
        cmap: str = "gist_rainbow",
        color: Union[str, None] = None,
        min_scale: float = 0.8,
        max_scale: float = 1.0,
        scale_factor: float = 1.0,
        random_seed: int = 888,
    ) -> np.ndarray:
        """Get colors for the labels based on node annotations or a specified colormap.

        Args:
            cmap (str, optional): Name of the colormap to use for generating label colors. Defaults to "gist_rainbow".
            color (str or None, optional): Color to use for the labels. If None, the colormap will be used. Defaults to None.
            min_scale (float, optional): Minimum intensity scale for the colors generated by the colormap.
                Controls the dimmest colors. Defaults to 0.8.
            max_scale (float, optional): Maximum intensity scale for the colors generated by the colormap.
                Controls the brightest colors. Defaults to 1.0.
            scale_factor (float, optional): Exponent for adjusting color scaling based on enrichment scores.
                A higher value increases contrast by dimming lower scores more. Defaults to 1.0.
            random_seed (int, optional): Seed for random number generation to ensure reproducibility. Defaults to 888.

        Returns:
            np.ndarray: Array of RGBA colors for label annotations.
        """
        return self._get_annotated_domain_colors(
            cmap=cmap,
            color=color,
            min_scale=min_scale,
            max_scale=max_scale,
            scale_factor=scale_factor,
            random_seed=random_seed,
        )

    def _get_annotated_domain_colors(
        self,
        cmap: str = "gist_rainbow",
        color: Union[str, None] = None,
        min_scale: float = 0.8,
        max_scale: float = 1.0,
        scale_factor: float = 1.0,
        random_seed: int = 888,
    ) -> np.ndarray:
        """Get colors for the domains based on node annotations, or use a specified color.

        Args:
            cmap (str, optional): Colormap to use for generating domain colors. Defaults to "gist_rainbow".
            color (str or None, optional): Color to use for the domains. If None, the colormap will be used. Defaults to None.
            min_scale (float, optional): Minimum scale for color intensity when generating domain colors.
                Defaults to 0.8.
            max_scale (float, optional): Maximum scale for color intensity when generating domain colors.
                Defaults to 1.0.
            scale_factor (float, optional): Factor for adjusting the contrast in the colors generated based on
                enrichment. Higher values increase the contrast. Defaults to 1.0.
            random_seed (int, optional): Seed for random number generation to ensure reproducibility. Defaults to 888.

        Returns:
            np.ndarray: Array of RGBA colors for each domain.
        """
        # Generate domain colors based on the enrichment data
        node_colors = self.graph.get_domain_colors(
            cmap=cmap,
            color=color,
            min_scale=min_scale,
            max_scale=max_scale,
            scale_factor=scale_factor,
            random_seed=random_seed,
        )
        annotated_colors = []
        for _, node_ids in self.graph.domain_id_to_node_ids_map.items():
            if len(node_ids) > 1:
                # For multi-node domains, choose the brightest color based on RGB sum
                domain_colors = np.array([node_colors[node] for node in node_ids])
                brightest_color = domain_colors[
                    np.argmax(domain_colors[:, :3].sum(axis=1))  # Sum the RGB values
                ]
                annotated_colors.append(brightest_color)
            else:
                # Single-node domains default to white (RGBA)
                default_color = np.array([1.0, 1.0, 1.0, 1.0])
                annotated_colors.append(default_color)

        return np.array(annotated_colors)

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


def _to_rgba(
    color: Union[str, List, Tuple, np.ndarray],
    alpha: float = 1.0,
    num_repeats: Union[int, None] = None,
) -> np.ndarray:
    """Convert a color or array of colors to RGBA format, applying alpha only if the color is RGB.

    Args:
        color (Union[str, list, tuple, np.ndarray]): The color(s) to convert. Can be a string, list, tuple, or np.ndarray.
        alpha (float, optional): Alpha value (transparency) to apply if the color is in RGB format. Defaults to 1.0.
        num_repeats (int or None, optional): If provided, the color will be repeated this many times. Defaults to None.

    Returns:
        np.ndarray: The RGBA color or array of RGBA colors.
    """
    # Handle single color case (string, RGB, or RGBA)
    if isinstance(color, str) or (
        isinstance(color, (list, tuple, np.ndarray))
        and len(color) in [3, 4]
        and not any(isinstance(c, (list, tuple, np.ndarray)) for c in color)
    ):
        rgba_color = np.array(mcolors.to_rgba(color))
        # Only set alpha if the input is an RGB color or a string (not RGBA)
        if len(rgba_color) == 4 and (
            len(color) == 3 or isinstance(color, str)
        ):  # If it's RGB or a string, set the alpha
            rgba_color[3] = alpha

        # Repeat the color if num_repeats argument is provided
        if num_repeats is not None:
            return np.array([rgba_color] * num_repeats)

        return rgba_color

    # Handle array of colors case (including strings, RGB, and RGBA)
    elif isinstance(color, (list, tuple, np.ndarray)):
        rgba_colors = []
        for c in color:
            # Ensure each element is either a valid string or a list/tuple of length 3 (RGB) or 4 (RGBA)
            if isinstance(c, str) or (
                isinstance(c, (list, tuple, np.ndarray)) and len(c) in [3, 4]
            ):
                rgba_c = np.array(mcolors.to_rgba(c))
                # Apply alpha only to RGB colors (not RGBA) and strings
                if len(rgba_c) == 4 and (len(c) == 3 or isinstance(c, str)):
                    rgba_c[3] = alpha

                rgba_colors.append(rgba_c)
            else:
                raise ValueError(f"Invalid color: {c}. Must be a valid RGB/RGBA or string color.")

        # Repeat the colors if num_repeats argument is provided
        if num_repeats is not None and len(rgba_colors) == 1:
            return np.array([rgba_colors[0]] * num_repeats)

        return np.array(rgba_colors)

    else:
        raise ValueError("Color must be a valid RGB/RGBA or array of RGB/RGBA colors.")


def _is_connected(z: np.ndarray) -> bool:
    """Determine if a thresholded grid represents a single, connected component.

    Args:
        z (np.ndarray): A binary grid where the component connectivity is evaluated.

    Returns:
        bool: True if the grid represents a single connected component, False otherwise.
    """
    _, num_features = label(z)
    return num_features == 1  # Return True if only one connected component is found


def _calculate_bounding_box(
    node_coordinates: np.ndarray, radius_margin: float = 1.05
) -> Tuple[np.ndarray, float]:
    """Calculate the bounding box of the network based on node coordinates.

    Args:
        node_coordinates (np.ndarray): Array of node coordinates (x, y).
        radius_margin (float, optional): Margin factor to apply to the bounding box radius. Defaults to 1.05.

    Returns:
        tuple: Center of the bounding box and the radius (adjusted by the radius margin).
    """
    # Find minimum and maximum x, y coordinates
    x_min, y_min = np.min(node_coordinates, axis=0)
    x_max, y_max = np.max(node_coordinates, axis=0)
    # Calculate the center of the bounding box
    center = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2])
    # Calculate the radius of the bounding box, adjusted by the margin
    radius = max(x_max - x_min, y_max - y_min) / 2 * radius_margin
    return center, radius


def _calculate_best_label_positions(
    filtered_domain_centroids: Dict[str, Any], center: np.ndarray, radius: float, offset: float
) -> Dict[str, Any]:
    """Calculate and optimize label positions for clarity.

    Args:
        filtered_domain_centroids (dict): Centroids of the filtered domains.
        center (np.ndarray): The center coordinates for label positioning.
        radius (float): The radius for positioning labels around the center.
        offset (float): The offset distance from the radius for positioning labels.

    Returns:
        dict: Optimized positions for labels.
    """
    num_domains = len(filtered_domain_centroids)
    # Calculate equidistant positions around the center for initial label placement
    equidistant_positions = _calculate_equidistant_positions_around_center(
        center, radius, offset, num_domains
    )
    # Create a mapping of domains to their initial label positions
    label_positions = {
        domain: position
        for domain, position in zip(filtered_domain_centroids.keys(), equidistant_positions)
    }
    # Optimize the label positions to minimize distance to domain centroids
    return _optimize_label_positions(label_positions, filtered_domain_centroids)


def _calculate_equidistant_positions_around_center(
    center: np.ndarray, radius: float, label_offset: float, num_domains: int
) -> List[np.ndarray]:
    """Calculate positions around a center at equidistant angles.

    Args:
        center (np.ndarray): The central point around which positions are calculated.
        radius (float): The radius at which positions are calculated.
        label_offset (float): The offset added to the radius for label positioning.
        num_domains (int): The number of positions (or domains) to calculate.

    Returns:
        list[np.ndarray]: List of positions (as 2D numpy arrays) around the center.
    """
    # Calculate equidistant angles in radians around the center
    angles = np.linspace(0, 2 * np.pi, num_domains, endpoint=False)
    # Compute the positions around the center using the angles
    return [
        center + (radius + label_offset) * np.array([np.cos(angle), np.sin(angle)])
        for angle in angles
    ]


def _optimize_label_positions(
    best_label_positions: Dict[str, Any], domain_centroids: Dict[str, Any]
) -> Dict[str, Any]:
    """Optimize label positions around the perimeter to minimize total distance to centroids.

    Args:
        best_label_positions (dict): Initial positions of labels around the perimeter.
        domain_centroids (dict): Centroid positions of the domains.

    Returns:
        dict: Optimized label positions.
    """
    while True:
        improvement = False  # Start each iteration assuming no improvement
        # Iterate through each pair of labels to check for potential improvements
        for i in range(len(domain_centroids)):
            for j in range(i + 1, len(domain_centroids)):
                # Calculate the current total distance
                current_distance = _calculate_total_distance(best_label_positions, domain_centroids)
                # Evaluate the total distance after swapping two labels
                swapped_distance = _swap_and_evaluate(best_label_positions, i, j, domain_centroids)
                # If the swap improves the total distance, perform the swap
                if swapped_distance < current_distance:
                    labels = list(best_label_positions.keys())
                    best_label_positions[labels[i]], best_label_positions[labels[j]] = (
                        best_label_positions[labels[j]],
                        best_label_positions[labels[i]],
                    )
                    improvement = True  # Found an improvement, so continue optimizing

        if not improvement:
            break  # Exit the loop if no improvement was found in this iteration

    return best_label_positions


def _calculate_total_distance(
    label_positions: Dict[str, Any], domain_centroids: Dict[str, Any]
) -> float:
    """Calculate the total distance from label positions to their domain centroids.

    Args:
        label_positions (dict): Positions of labels around the perimeter.
        domain_centroids (dict): Centroid positions of the domains.

    Returns:
        float: The total distance from labels to centroids.
    """
    total_distance = 0
    # Iterate through each domain and calculate the distance to its centroid
    for domain, pos in label_positions.items():
        centroid = domain_centroids[domain]
        total_distance += np.linalg.norm(centroid - pos)

    return total_distance


def _swap_and_evaluate(
    label_positions: Dict[str, Any],
    i: int,
    j: int,
    domain_centroids: Dict[str, Any],
) -> float:
    """Swap two labels and evaluate the total distance after the swap.

    Args:
        label_positions (dict): Positions of labels around the perimeter.
        i (int): Index of the first label to swap.
        j (int): Index of the second label to swap.
        domain_centroids (dict): Centroid positions of the domains.

    Returns:
        float: The total distance after swapping the two labels.
    """
    # Get the list of labels from the dictionary keys
    labels = list(label_positions.keys())
    swapped_positions = label_positions.copy()
    # Swap the positions of the two specified labels
    swapped_positions[labels[i]], swapped_positions[labels[j]] = (
        swapped_positions[labels[j]],
        swapped_positions[labels[i]],
    )
    # Calculate and return the total distance after the swap
    return _calculate_total_distance(swapped_positions, domain_centroids)
