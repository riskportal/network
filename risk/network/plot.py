"""
risk/network/plot
~~~~~~~~~~~~~~~~~
"""

from typing import Any, Dict, List, Tuple, Union

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.ndimage import label
from scipy.stats import gaussian_kde

from risk.log import params
from risk.network.graph import NetworkGraph


class NetworkPlotter:
    """A class responsible for visualizing network graphs with various customization options.

    The NetworkPlotter class takes in a NetworkGraph object, which contains the network's data and attributes,
    and provides methods for plotting the network with customizable node and edge properties,
    as well as optional features like drawing the network's perimeter and setting background colors.
    """

    def __init__(
        self,
        network_graph: NetworkGraph,
        figsize: tuple = (10, 10),
        background_color: str = "white",
        plot_outline: bool = True,
        outline_color: str = "black",
        outline_scale: float = 1.0,
    ) -> None:
        """Initialize the NetworkPlotter with a NetworkGraph object and plotting parameters.

        Args:
            network_graph (NetworkGraph): The network data and attributes to be visualized.
            figsize (tuple, optional): Size of the figure in inches (width, height). Defaults to (10, 10).
            background_color (str, optional): Background color of the plot. Defaults to "white".
            plot_outline (bool, optional): Whether to plot the network perimeter circle. Defaults to True.
            outline_color (str, optional): Color of the network perimeter circle. Defaults to "black".
            outline_scale (float, optional): Outline scaling factor for the perimeter diameter. Defaults to 1.0.
        """
        self.network_graph = network_graph
        self.ax = None  # Initialize the axis attribute
        # Initialize the plot with the given parameters
        self._initialize_plot(figsize, background_color, plot_outline, outline_color, outline_scale)

    def _initialize_plot(
        self,
        figsize: tuple,
        background_color: str,
        plot_outline: bool,
        outline_color: str,
        outline_scale: float,
    ) -> tuple:
        """Set up the plot with figure size, optional circle perimeter, and background color.

        Args:
            figsize (tuple): Size of the figure in inches (width, height).
            background_color (str): Background color of the plot.
            plot_outline (bool): Whether to plot the network perimeter circle.
            outline_color (str): Color of the network perimeter circle.
            outline_scale (float): Outline scaling factor for the perimeter diameter.

        Returns:
            tuple: The created matplotlib figure and axis.
        """
        # Extract node coordinates from the network graph
        node_coordinates = self.network_graph.node_coordinates
        # Calculate the center and radius of the bounding box around the network
        center, radius = _calculate_bounding_box(node_coordinates)
        # Scale the radius by the outline_scale factor
        scaled_radius = radius * outline_scale

        # Create a new figure and axis for plotting
        fig, ax = plt.subplots(figsize=figsize)
        fig.tight_layout()  # Adjust subplot parameters to give specified padding
        if plot_outline:
            # Draw a circle to represent the network perimeter
            circle = plt.Circle(
                center,
                scaled_radius,
                linestyle="--",
                color=outline_color,
                fill=False,
                linewidth=1.5,
            )
            ax.add_artist(circle)  # Add the circle to the plot

        # Set axis limits based on the calculated bounding box and scaled radius
        ax.set_xlim([center[0] - scaled_radius - 0.3, center[0] + scaled_radius + 0.3])
        ax.set_ylim([center[1] - scaled_radius - 0.3, center[1] + scaled_radius + 0.3])
        ax.set_aspect("equal")  # Ensure the aspect ratio is equal
        fig.patch.set_facecolor(background_color)  # Set the figure background color
        ax.invert_yaxis()  # Invert the y-axis to match typical image coordinates

        # Remove axis spines for a cleaner look
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Hide axis ticks and labels
        ax.set_xticks([])
        ax.set_yticks([])
        ax.patch.set_visible(False)  # Hide the axis background

        # Store the axis for further use and return the figure and axis
        self.ax = ax
        return fig, ax

    def plot_network(
        self,
        node_size: Union[int, np.ndarray] = 50,
        edge_width: float = 1.0,
        node_color: Union[str, np.ndarray] = "white",
        node_edgecolor: str = "black",
        edge_color: str = "black",
        node_shape: str = "o",
    ) -> None:
        """Plot the network graph with customizable node colors, sizes, and edge widths.

        Args:
            node_size (int or np.ndarray, optional): Size of the nodes. Can be a single integer or an array of sizes. Defaults to 50.
            edge_width (float, optional): Width of the edges. Defaults to 1.0.
            node_color (str or np.ndarray, optional): Color of the nodes. Can be a single color or an array of colors. Defaults to "white".
            node_edgecolor (str, optional): Color of the node edges. Defaults to "black".
            edge_color (str, optional): Color of the edges. Defaults to "black".
            node_shape (str, optional): Shape of the nodes. Defaults to "o".
        """
        # Log the plotting parameters
        params.log_plotter(
            network_node_size="custom" if isinstance(node_size, np.ndarray) else node_size,
            network_edge_width=edge_width,
            network_node_color="custom" if isinstance(node_color, np.ndarray) else node_color,
            network_node_edgecolor=node_edgecolor,
            network_edge_color=edge_color,
            network_node_shape=node_shape,
        )
        # Extract node coordinates from the network graph
        node_coordinates = self.network_graph.node_coordinates
        # Draw the nodes of the graph
        nx.draw_networkx_nodes(
            self.network_graph.G,
            pos=node_coordinates,
            node_size=node_size,
            node_color=node_color,
            node_shape=node_shape,
            alpha=1.00,
            edgecolors=node_edgecolor,
            ax=self.ax,
        )
        # Draw the edges of the graph
        nx.draw_networkx_edges(
            self.network_graph.G,
            pos=node_coordinates,
            width=edge_width,
            edge_color=edge_color,
            ax=self.ax,
        )

    def plot_subnetwork(
        self,
        nodes: list,
        node_size: Union[int, np.ndarray] = 50,
        edge_width: float = 1.0,
        node_color: Union[str, np.ndarray] = "white",
        node_edgecolor: str = "black",
        edge_color: str = "black",
        node_shape: str = "o",
    ) -> None:
        """Plot a subnetwork of selected nodes with customizable node and edge attributes.

        Args:
            nodes (list): List of node labels to include in the subnetwork.
            node_size (int or np.ndarray, optional): Size of the nodes. Can be a single integer or an array of sizes. Defaults to 50.
            edge_width (float, optional): Width of the edges. Defaults to 1.0.
            node_color (str or np.ndarray, optional): Color of the nodes. Can be a single color or an array of colors. Defaults to "white".
            node_edgecolor (str, optional): Color of the node edges. Defaults to "black".
            edge_color (str, optional): Color of the edges. Defaults to "black".
            node_shape (str, optional): Shape of the nodes. Defaults to "o".

        Raises:
            ValueError: If no valid nodes are found in the network graph.
        """
        # Log the plotting parameters for the subnetwork
        params.log_plotter(
            subnetwork_node_size="custom" if isinstance(node_size, np.ndarray) else node_size,
            subnetwork_edge_width=edge_width,
            subnetwork_node_color="custom" if isinstance(node_color, np.ndarray) else node_color,
            subnetwork_node_edgecolor=node_edgecolor,
            subnetwork_edge_color=edge_color,
            subnet_node_shape=node_shape,
        )
        # Filter to get node IDs and their coordinates
        node_ids = [
            self.network_graph.node_label_to_id_map.get(node)
            for node in nodes
            if node in self.network_graph.node_label_to_id_map
        ]
        if not node_ids:
            raise ValueError("No nodes found in the network graph.")

        # Get the coordinates of the filtered nodes
        node_coordinates = {
            node_id: self.network_graph.node_coordinates[node_id] for node_id in node_ids
        }
        # Draw the nodes in the subnetwork
        nx.draw_networkx_nodes(
            self.network_graph.G,
            pos=node_coordinates,
            nodelist=node_ids,
            node_size=node_size,
            node_color=node_color,
            node_shape=node_shape,
            alpha=1.00,
            edgecolors=node_edgecolor,
            ax=self.ax,
        )
        # Draw the edges between the specified nodes in the subnetwork
        subgraph = self.network_graph.G.subgraph(node_ids)
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
        alpha: float = 0.2,
        color: Union[str, np.ndarray] = "white",
    ) -> None:
        """Draw KDE contours for nodes in various domains of a network graph, highlighting areas of high density.

        Args:
            levels (int, optional): Number of contour levels to plot. Defaults to 5.
            bandwidth (float, optional): Bandwidth for KDE. Controls the smoothness of the contour. Defaults to 0.8.
            grid_size (int, optional): Resolution of the grid for KDE. Higher values create finer contours. Defaults to 250.
            alpha (float, optional): Transparency level of the contour fill. Defaults to 0.2.
            color (str or np.ndarray, optional): Color of the contours. Can be a string (e.g., 'white') or an array of colors. Defaults to "white".
        """
        # Log the contour plotting parameters
        params.log_plotter(
            contour_levels=levels,
            contour_bandwidth=bandwidth,
            contour_grid_size=grid_size,
            contour_alpha=alpha,
            contour_color="custom" if isinstance(color, np.ndarray) else color,
        )
        # Convert color string to RGBA array if necessary
        if isinstance(color, str):
            color = self.get_annotated_contour_colors(color=color)

        # Extract node coordinates from the network graph
        node_coordinates = self.network_graph.node_coordinates
        # Draw contours for each domain in the network
        for idx, (_, nodes) in enumerate(self.network_graph.domain_to_nodes.items()):
            if len(nodes) > 1:
                self._draw_kde_contour(
                    self.ax,
                    node_coordinates,
                    nodes,
                    color=color[idx],
                    levels=levels,
                    bandwidth=bandwidth,
                    grid_size=grid_size,
                    alpha=alpha,
                )

    def plot_subcontour(
        self,
        nodes: list,
        levels: int = 5,
        bandwidth: float = 0.8,
        grid_size: int = 250,
        alpha: float = 0.2,
        color: Union[str, np.ndarray] = "white",
    ) -> None:
        """Plot a subcontour for a given set of nodes using Kernel Density Estimation (KDE).

        Args:
            nodes (list): List of node labels to plot the contour for.
            levels (int, optional): Number of contour levels to plot. Defaults to 5.
            bandwidth (float, optional): Bandwidth for KDE. Controls the smoothness of the contour. Defaults to 0.8.
            grid_size (int, optional): Resolution of the grid for KDE. Higher values create finer contours. Defaults to 250.
            alpha (float, optional): Transparency level of the contour fill. Defaults to 0.2.
            color (str or np.ndarray, optional): Color of the contour. Can be a string (e.g., 'white') or RGBA array. Defaults to "white".

        Raises:
            ValueError: If no valid nodes are found in the network graph.
        """
        # Log the plotting parameters
        params.log_plotter(
            contour_levels=levels,
            contour_bandwidth=bandwidth,
            contour_grid_size=grid_size,
            contour_alpha=alpha,
            contour_color="custom" if isinstance(color, np.ndarray) else color,
        )
        # Filter to get node IDs and their coordinates
        node_ids = [
            self.network_graph.node_label_to_id_map.get(node)
            for node in nodes
            if node in self.network_graph.node_label_to_id_map
        ]
        if not node_ids or len(node_ids) == 1:
            raise ValueError("No nodes found in the network graph or insufficient nodes to plot.")

        # Draw the KDE contour for the specified nodes
        node_coordinates = self.network_graph.node_coordinates
        self._draw_kde_contour(
            self.ax,
            node_coordinates,
            node_ids,
            color=color,
            levels=levels,
            bandwidth=bandwidth,
            grid_size=grid_size,
            alpha=alpha,
        )

    def _draw_kde_contour(
        self,
        ax: plt.Axes,
        pos: np.ndarray,
        nodes: list,
        color: Union[str, np.ndarray],
        levels: int = 5,
        bandwidth: float = 0.8,
        grid_size: int = 250,
        alpha: float = 0.5,
    ) -> None:
        """Draw a Kernel Density Estimate (KDE) contour plot for a set of nodes on a given axis.

        Args:
            ax (plt.Axes): The axis to draw the contour on.
            pos (np.ndarray): Array of node positions (x, y).
            nodes (list): List of node indices to include in the contour.
            color (str or np.ndarray): Color for the contour.
            levels (int, optional): Number of contour levels. Defaults to 5.
            bandwidth (float, optional): Bandwidth for the KDE. Controls smoothness. Defaults to 0.8.
            grid_size (int, optional): Grid resolution for the KDE. Higher values yield finer contours. Defaults to 250.
            alpha (float, optional): Transparency level for the contour fill. Defaults to 0.5.
        """
        # Extract the positions of the specified nodes
        points = np.array([pos[n] for n in nodes])
        if len(points) <= 1:
            return  # Not enough points to form a contour

        connected = False
        while not connected and bandwidth <= 100.0:
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

        # Define contour levels based on the density
        min_density, max_density = z.min(), z.max()
        contour_levels = np.linspace(min_density, max_density, levels)[1:]
        contour_colors = [color for _ in range(levels - 1)]

        # Plot the filled contours if alpha > 0
        if alpha > 0:
            ax.contourf(
                x,
                y,
                z,
                levels=contour_levels,
                colors=contour_colors,
                alpha=alpha,
                extend="neither",
                antialiased=True,
            )

        # Plot the contour lines without antialiasing for clarity
        c = ax.contour(x, y, z, levels=contour_levels, colors=contour_colors)
        for i in range(1, len(contour_levels)):
            c.collections[i].set_linewidth(0)

    def plot_labels(
        self,
        perimeter_scale: float = 1.05,
        offset: float = 0.10,
        font: str = "Arial",
        fontsize: int = 10,
        fontcolor: Union[str, np.ndarray] = "black",
        arrow_linewidth: float = 1,
        arrow_color: Union[str, np.ndarray] = "black",
        num_words: int = 10,
        min_words: int = 1,
    ) -> None:
        """Annotate the network graph with labels for different domains, positioned around the network for clarity.

        Args:
            perimeter_scale (float, optional): Scale factor for positioning labels around the perimeter. Defaults to 1.05.
            offset (float, optional): Offset distance for labels from the perimeter. Defaults to 0.10.
            font (str, optional): Font name for the labels. Defaults to "Arial".
            fontsize (int, optional): Font size for the labels. Defaults to 10.
            fontcolor (str or np.ndarray, optional): Color of the label text. Can be a string or RGBA array. Defaults to "black".
            arrow_linewidth (float, optional): Line width of the arrows pointing to centroids. Defaults to 1.
            arrow_color (str or np.ndarray, optional): Color of the arrows. Can be a string or RGBA array. Defaults to "black".
            num_words (int, optional): Maximum number of words in a label. Defaults to 10.
            min_words (int, optional): Minimum number of words required to display a label. Defaults to 1.
        """
        # Log the plotting parameters
        params.log_plotter(
            label_perimeter_scale=perimeter_scale,
            label_offset=offset,
            label_font=font,
            label_fontsize=fontsize,
            label_fontcolor="custom" if isinstance(fontcolor, np.ndarray) else fontcolor,
            label_arrow_linewidth=arrow_linewidth,
            label_arrow_color="custom" if isinstance(arrow_color, np.ndarray) else arrow_color,
            label_num_words=num_words,
            label_min_words=min_words,
        )
        # Convert color strings to RGBA arrays if necessary
        if isinstance(fontcolor, str):
            fontcolor = self.get_annotated_contour_colors(color=fontcolor)
        if isinstance(arrow_color, str):
            arrow_color = self.get_annotated_contour_colors(color=arrow_color)

        # Calculate the center and radius of the network
        domain_centroids = self._calculate_domain_centroids()
        center, radius = _calculate_bounding_box(
            self.network_graph.node_coordinates, radius_margin=perimeter_scale
        )

        # Filter out domains with insufficient words for labeling
        filtered_domains = {
            domain: centroid
            for domain, centroid in domain_centroids.items()
            if len(self.network_graph.trimmed_domain_to_term[domain].split(" ")[:num_words])
            >= min_words
        }
        # Calculate the best positions for labels around the perimeter
        best_label_positions = _best_label_positions(filtered_domains, center, radius, offset)
        # Annotate the network with labels
        for idx, (domain, pos) in enumerate(best_label_positions.items()):
            centroid = filtered_domains[domain]
            annotations = self.network_graph.trimmed_domain_to_term[domain].split(" ")[:num_words]
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
                arrowprops=dict(arrowstyle="->", color=arrow_color[idx], linewidth=arrow_linewidth),
            )

    def _calculate_domain_centroids(self) -> Dict[Any, np.ndarray]:
        """Calculate the most centrally located node within each domain based on the node positions.

        Returns:
            Dict[Any, np.ndarray]: A dictionary mapping each domain to its central node's coordinates.
        """
        domain_central_nodes = {}
        for domain, nodes in self.network_graph.domain_to_nodes.items():
            if not nodes:  # Skip if the domain has no nodes
                continue

            # Extract positions of all nodes in the domain
            node_positions = self.network_graph.node_coordinates[nodes, :]
            # Calculate the pairwise distance matrix between all nodes in the domain
            distances_matrix = np.linalg.norm(
                node_positions[:, np.newaxis] - node_positions, axis=2
            )
            # Sum the distances for each node to all other nodes in the domain
            sum_distances = np.sum(distances_matrix, axis=1)
            # Identify the node with the smallest total distance to others (the centroid)
            central_node_idx = np.argmin(sum_distances)
            # Map the domain to the coordinates of its central node
            domain_central_nodes[domain] = node_positions[central_node_idx]

        return domain_central_nodes

    def get_annotated_node_colors(
        self, nonenriched_color: str = "white", random_seed: int = 888, **kwargs
    ) -> np.ndarray:
        """Adjust the colors of nodes in the network graph based on enrichment.

        Args:
            nonenriched_color (str, optional): Color for non-enriched nodes. Defaults to "white".
            random_seed (int, optional): Seed for random number generation. Defaults to 888.
            **kwargs: Additional keyword arguments for `get_domain_colors`.

        Returns:
            np.ndarray: Array of RGBA colors adjusted for enrichment status.
        """
        # Get the initial domain colors for each node
        network_colors = self.network_graph.get_domain_colors(**kwargs, random_seed=random_seed)
        if isinstance(nonenriched_color, str):
            # Convert the non-enriched color from string to RGBA
            nonenriched_color = mcolors.to_rgba(nonenriched_color)

        # Adjust node colors: replace any fully transparent nodes (enriched) with the non-enriched color
        adjusted_network_colors = np.where(
            np.all(network_colors == 0, axis=1, keepdims=True),
            np.array([nonenriched_color]),
            network_colors,
        )
        return adjusted_network_colors

    def get_annotated_node_sizes(
        self, enriched_nodesize: int = 50, nonenriched_nodesize: int = 25
    ) -> np.ndarray:
        """Adjust the sizes of nodes in the network graph based on whether they are enriched or not.

        Args:
            enriched_nodesize (int): Size for enriched nodes. Defaults to 50.
            nonenriched_nodesize (int): Size for non-enriched nodes. Defaults to 25.

        Returns:
            np.ndarray: Array of node sizes, with enriched nodes larger than non-enriched ones.
        """
        # Merge all enriched nodes from the domain_to_nodes dictionary
        enriched_nodes = set()
        for _, nodes in self.network_graph.domain_to_nodes.items():
            enriched_nodes.update(nodes)

        # Initialize all node sizes to the non-enriched size
        node_sizes = np.full(len(self.network_graph.G.nodes), nonenriched_nodesize)
        # Set the size for enriched nodes
        for node in enriched_nodes:
            if node in self.network_graph.G.nodes:
                node_sizes[node] = enriched_nodesize

        return node_sizes

    def get_annotated_contour_colors(self, random_seed: int = 888, **kwargs) -> np.ndarray:
        """Get colors for the contours based on node annotations.

        Args:
            random_seed (int, optional): Seed for random number generation. Defaults to 888.
            **kwargs: Additional keyword arguments for `_get_annotated_domain_colors`.

        Returns:
            np.ndarray: Array of RGBA colors for contour annotations.
        """
        return self._get_annotated_domain_colors(**kwargs, random_seed=random_seed)

    def get_annotated_label_colors(self, random_seed: int = 888, **kwargs) -> np.ndarray:
        """Get colors for the labels based on node annotations.

        Args:
            random_seed (int, optional): Seed for random number generation. Defaults to 888.
            **kwargs: Additional keyword arguments for `_get_annotated_domain_colors`.

        Returns:
            np.ndarray: Array of RGBA colors for label annotations.
        """
        return self._get_annotated_domain_colors(**kwargs, random_seed=random_seed)

    def _get_annotated_domain_colors(
        self, color: Union[str, list, None] = None, random_seed: int = 888, **kwargs
    ) -> np.ndarray:
        """Get colors for the domains based on node annotations.

        Args:
            color (str, list, or None, optional): If provided, use this color or list of colors for domains. Defaults to None.
            random_seed (int, optional): Seed for random number generation. Defaults to 888.
            **kwargs: Additional keyword arguments for `get_domain_colors`.

        Returns:
            np.ndarray: Array of RGBA colors for each domain.
        """
        if isinstance(color, str):
            # If a single color string is provided, convert it to RGBA and apply to all domains
            rgba_color = np.array(matplotlib.colors.to_rgba(color))
            return np.array([rgba_color for _ in self.network_graph.domain_to_nodes])

        # Generate colors for each domain using the provided arguments and random seed
        node_colors = self.network_graph.get_domain_colors(**kwargs, random_seed=random_seed)
        annotated_colors = []
        for _, nodes in self.network_graph.domain_to_nodes.items():
            if len(nodes) > 1:
                # For domains with multiple nodes, choose the brightest color (sum of RGB values)
                domain_colors = np.array([node_colors[node] for node in nodes])
                brightest_color = domain_colors[np.argmax(domain_colors.sum(axis=1))]
                annotated_colors.append(brightest_color)
            else:
                # Assign a default color (white) for single-node domains
                default_color = np.array([1.0, 1.0, 1.0, 1.0])
                annotated_colors.append(default_color)

        return np.array(annotated_colors)

    @staticmethod
    def close(*args, **kwargs) -> None:
        """Close the current plot.

        Args:
            *args: Positional arguments passed to `plt.close`.
            **kwargs: Keyword arguments passed to `plt.close`.
        """
        plt.close(*args, **kwargs)

    @staticmethod
    def savefig(*args, **kwargs) -> None:
        """Save the current plot to a file.

        Args:
            *args: Positional arguments passed to `plt.savefig`.
            **kwargs: Keyword arguments passed to `plt.savefig`, such as filename and format.
        """
        plt.savefig(*args, bbox_inches="tight", **kwargs)

    @staticmethod
    def show(*args, **kwargs) -> None:
        """Display the current plot.

        Args:
            *args: Positional arguments passed to `plt.show`.
            **kwargs: Keyword arguments passed to `plt.show`.
        """
        plt.show(*args, **kwargs)


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


def _best_label_positions(
    filtered_domains: Dict[str, Any], center: np.ndarray, radius: float, offset: float
) -> Dict[str, Any]:
    """Calculate and optimize label positions for clarity.

    Args:
        filtered_domains (dict): Centroids of the filtered domains.
        center (np.ndarray): The center coordinates for label positioning.
        radius (float): The radius for positioning labels around the center.
        offset (float): The offset distance from the radius for positioning labels.

    Returns:
        dict: Optimized positions for labels.
    """
    num_domains = len(filtered_domains)
    # Calculate equidistant positions around the center for initial label placement
    equidistant_positions = _equidistant_angles_around_center(center, radius, offset, num_domains)
    # Create a mapping of domains to their initial label positions
    label_positions = {
        domain: position for domain, position in zip(filtered_domains.keys(), equidistant_positions)
    }
    # Optimize the label positions to minimize distance to domain centroids
    return _optimize_label_positions(label_positions, filtered_domains)


def _equidistant_angles_around_center(
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
