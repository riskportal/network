import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.ndimage import label
from scipy.stats import gaussian_kde


class NetworkPlotter:
    """A class to handle plotting of network graphs."""

    def __init__(
        self,
        network_graph,
        figsize=(10, 10),
        background_color="white",
        plot_outline=True,
        outline_color="black",
        outline_scale=1.0,
    ):
        """Initialize the NetworkPlotter with a NetworkGraph object.

        Args:
            network_graph: A NetworkGraph object containing the network's data and attributes.
            figsize (tuple, optional): Size of the figure. Defaults to (10, 10).
            background_color (str, optional): Background color of the plot. Defaults to "white".
            plot_outline (bool, optional): Whether to plot the network perimeter circle. Defaults to True.
            outline_color (str, optional): Color of the network perimeter circle. Defaults to "black".
            outline_scale (float, optional): Outline scaling factor for the perimeter diameter. Defaults to 1.0.
        """
        self.network_graph = network_graph
        self.ax = None
        self._initialize_plot(figsize, background_color, plot_outline, outline_color, outline_scale)

    def _initialize_plot(
        self, figsize, background_color, plot_outline, outline_color, outline_scale
    ):
        """Initialize the plot with figure size, optional circle perimeter, and background color."""
        node_coordinates = self.network_graph.node_coordinates
        center, radius = _calculate_bounding_box(node_coordinates)
        scaled_radius = radius * outline_scale

        fig, ax = plt.subplots(figsize=figsize)
        fig.tight_layout()

        if plot_outline:
            circle = plt.Circle(
                center,
                scaled_radius,
                linestyle="--",
                color=outline_color,
                fill=False,
                linewidth=1.5,
            )
            ax.add_artist(circle)

        ax.set_xlim([center[0] - scaled_radius - 0.3, center[0] + scaled_radius + 0.3])
        ax.set_ylim([center[1] - scaled_radius - 0.3, center[1] + scaled_radius + 0.3])
        ax.set_aspect("equal")
        fig.patch.set_facecolor(background_color)
        ax.invert_yaxis()

        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.patch.set_visible(False)

        self.ax = ax
        return fig, ax

    def add_domain(self, key, nodes, color="chartreuse"):
        ...

    def plot_network(
        self,
        node_size=50,
        edge_width=1.0,
        node_color="white",
        edge_color="black",
        node_edgecolor="black",
    ):
        """Plot the network graph with customizable node colors, sizes, and edge widths."""
        node_coordinates = self.network_graph.node_coordinates

        nx.draw_networkx_nodes(
            self.network_graph.network,
            pos=node_coordinates,
            node_size=node_size,
            node_color=node_color,
            alpha=1.00,
            edgecolors=node_edgecolor,
            ax=self.ax,
        )
        nx.draw_networkx_edges(
            self.network_graph.network,
            pos=node_coordinates,
            width=edge_width,
            edge_color=edge_color,
            ax=self.ax,
        )

    def plot_contours(
        self,
        levels=5,
        bandwidth=0.8,
        grid_size=200,
        alpha=0.2,
        color="white",
    ):
        """Draws KDE contours for nodes in various domains of a network graph, highlighting areas of high density."""
        # Check if color is a list of colors or a single color string
        if isinstance(color, str):
            color = self.get_annotated_contour_colors(color=color)

        node_coordinates = self.network_graph.node_coordinates

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

    def _draw_kde_contour(
        self, ax, pos, nodes, color, levels=5, bandwidth=0.8, grid_size=200, alpha=0.5
    ):
        """Draws a Kernel Density Estimate (KDE) contour plot for a set of nodes on a given axis."""
        points = np.array([pos[n] for n in nodes])
        if len(points) <= 1:
            return

        connected = False
        while not connected and bandwidth <= 100.0:
            kde = gaussian_kde(points.T, bw_method=bandwidth)
            xmin, ymin = points.min(axis=0) - bandwidth
            xmax, ymax = points.max(axis=0) + bandwidth
            x, y = np.mgrid[
                xmin : xmax : complex(0, grid_size), ymin : ymax : complex(0, grid_size)
            ]
            z = kde(np.vstack([x.ravel(), y.ravel()])).reshape(x.shape)
            thresholded_grid = _generate_thresholded_grid(z)
            connected = _is_connected(thresholded_grid)
            if not connected:
                bandwidth += 0.05

        min_density, max_density = z.min(), z.max()
        contour_levels = np.linspace(min_density, max_density, levels)[1:]
        contour_colors = [color for _ in range(levels - 1)]

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

        c = ax.contour(x, y, z, levels=contour_levels, colors=contour_colors)
        for i in range(1, len(contour_levels)):
            c.collections[i].set_linewidth(0)

    def plot_labels(
        self,
        perimeter_scale=1.05,
        offset=0.10,
        font="Arial",
        fontsize=10,
        fontcolor="black",
        arrow_linewidth=1,
        arrow_color="black",
        num_words=10,
        min_words=1,
    ):
        """Annotates a network graph with labels for different domains, positioned around the network for clarity."""
        # Check if color is a list of colors or a single color string
        if isinstance(fontcolor, str):
            fontcolor = self.get_annotated_contour_colors(color=fontcolor)
        if isinstance(arrow_color, str):
            arrow_color = self.get_annotated_contour_colors(color=arrow_color)

        domain_centroids = self._calculate_domain_centroids()
        center, radius = _calculate_bounding_box(
            self.network_graph.node_coordinates, radius_margin=perimeter_scale
        )

        # Filter out domains with insufficient words
        filtered_domains = {
            domain: centroid
            for domain, centroid in domain_centroids.items()
            if len(self.network_graph.trimmed_domain_to_term[domain].split(" ")[:num_words])
            >= min_words
        }

        num_domains = len(filtered_domains)
        equidistant_positions = _equidistant_angles_around_center(
            center, radius, offset, num_domains
        )
        label_positions = {
            domain: position
            for domain, position in zip(filtered_domains.keys(), equidistant_positions)
        }
        best_label_positions = self._optimize_label_positions(label_positions, filtered_domains)

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

    def _calculate_domain_centroids(self):
        """Calculates the most centrally located node within each domain based on the node positions."""
        domain_central_nodes = {}
        for domain, nodes in self.network_graph.domain_to_nodes.items():
            if not nodes:  # Skip if the domain has no nodes
                continue
            node_positions = self.network_graph.node_coordinates[nodes, :]
            distances_matrix = np.linalg.norm(
                node_positions[:, np.newaxis] - node_positions, axis=2
            )
            sum_distances = np.sum(distances_matrix, axis=1)
            central_node_idx = np.argmin(sum_distances)
            domain_central_nodes[domain] = node_positions[central_node_idx]
        return domain_central_nodes

    def _optimize_label_positions(self, best_label_positions, domain_centroids):
        """Optimizes label positions around the perimeter to minimize total distance to centroids."""
        improvement = True
        while improvement:
            improvement = False
            for i in range(len(domain_centroids)):
                for j in range(i + 1, len(domain_centroids)):
                    current_distance = _calculate_total_distance(
                        best_label_positions, domain_centroids
                    )
                    swapped_distance = _swap_and_evaluate(
                        best_label_positions, i, j, domain_centroids
                    )
                    if swapped_distance < current_distance:
                        labels = list(best_label_positions.keys())
                        best_label_positions[labels[i]], best_label_positions[labels[j]] = (
                            best_label_positions[labels[j]],
                            best_label_positions[labels[i]],
                        )
                        improvement = True
        return best_label_positions

    def get_annotated_node_colors(self, nonenriched_color="white", random_seed=888, **kwargs):
        """Adjusts the colors of nodes in the network graph.

        Returns:
            Tuple of two elements (adjusted colors array, node sizes array).
        """
        network_colors = self.network_graph.get_domain_colors(**kwargs, random_seed=random_seed)
        if isinstance(nonenriched_color, str):
            nonenriched_color = mcolors.to_rgba(nonenriched_color)

        adjusted_network_colors = np.where(
            np.all(network_colors == 0, axis=1, keepdims=True),
            np.array([nonenriched_color]),
            network_colors,
        )

        return adjusted_network_colors

    def get_annotated_node_sizes(self, enriched_nodesize=50, nonenriched_nodesize=25):
        """Adjusts the sizes of nodes in the network graph based on whether they are enriched or not.

        Args:
            enriched_nodesize (int): Size for enriched nodes. Default is 50.
            nonenriched_nodesize (int): Size for non-enriched nodes. Default is 25.

        Returns:
            np.ndarray: Array of node sizes.
        """
        # Merge all enriched nodes from the network_graph.domain_to_nodes dictionary
        enriched_nodes = set()
        for _, nodes in self.network_graph.domain_to_nodes.items():
            enriched_nodes.update(nodes)

        # Initialize node sizes array
        node_sizes = np.full(len(self.network_graph.network.nodes), nonenriched_nodesize)
        # Set sizes for enriched nodes
        for node in enriched_nodes:
            if node in self.network_graph.network.nodes:
                node_sizes[node] = enriched_nodesize

        return node_sizes

    def get_annotated_contour_colors(self, random_seed=888, **kwargs):
        """Get colors for the contours based on node annotations."""
        return self._get_annotated_domain_colors(**kwargs, random_seed=random_seed)

    def get_annotated_label_colors(self, random_seed=888, **kwargs):
        """Get colors for the contours based on node annotations."""
        return self._get_annotated_domain_colors(**kwargs, random_seed=random_seed)

    def _get_annotated_domain_colors(self, color=None, random_seed=888, **kwargs):
        """Get colors for the contours based on node annotations.

        Args:
            random_seed (int, optional): Seed for random number generation. Defaults to 888.
            color (str or list or None, optional): If provided, use this color or list of colors for domains. Defaults to None.
            **kwargs: Additional keyword arguments for `get_domain_colors`.

        Returns:
            np.ndarray: Array of RGBA colors for each domain.
        """
        if isinstance(color, str):
            rgba_color = np.array(matplotlib.colors.to_rgba(color))
            return np.array([rgba_color for _ in self.network_graph.domain_to_nodes])

        node_colors = self.network_graph.get_domain_colors(**kwargs, random_seed=random_seed)
        annotated_colors = []

        for _, nodes in self.network_graph.domain_to_nodes.items():
            if len(nodes) > 1:
                domain_colors = np.array([node_colors[node] for node in nodes])
                brightest_color = domain_colors[np.argmax(domain_colors.sum(axis=1))]
                annotated_colors.append(brightest_color)
            else:
                default_color = np.array([1.0, 1.0, 1.0, 1.0])
                annotated_colors.append(default_color)  # Default color for single node domains

        return np.array(annotated_colors)

    @staticmethod
    def close(*args, **kwargs):
        """Close current plot."""
        plt.close(*args, **kwargs)

    @staticmethod
    def savefig(*args, **kwargs):
        """Save the current plot to a file."""
        plt.savefig(*args, bbox_inches="tight", **kwargs)

    @staticmethod
    def show(*args, **kwargs):
        """Display the current plot."""
        plt.show(*args, **kwargs)


def _is_connected(thresholded_grid):
    """Determines if a thresholded grid represents a single, connected component."""
    _, num_features = label(thresholded_grid)
    return num_features == 1


def _generate_thresholded_grid(z, threshold=0.05):
    """Generates a thresholded grid from a KDE grid by applying a threshold value."""
    return z > threshold


def _calculate_bounding_box(node_coordinates, radius_margin=1.05):
    """Calculates the bounding box of the network based on node coordinates."""
    x_min, y_min = np.min(node_coordinates, axis=0)
    x_max, y_max = np.max(node_coordinates, axis=0)
    center = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2])
    radius = max(x_max - x_min, y_max - y_min) / 2 * radius_margin
    return center, radius


def _determine_label_positions(domain_centroids, center, radius, label_offset):
    """Determines the label positions around the circle perimeter based on domain centroids."""
    label_positions = {}
    for domain, centroid in domain_centroids.items():
        direction = centroid - center
        direction /= np.linalg.norm(direction)
        label_positions[domain] = center + direction * (radius + label_offset)
    return label_positions


def _equidistant_angles_around_center(center, radius, label_offset, num_domains):
    """Calculates positions around a center at equidistant angles."""
    angles = np.linspace(0, 2 * np.pi, num_domains, endpoint=False)
    return [
        center + (radius + label_offset) * np.array([np.cos(angle), np.sin(angle)])
        for angle in angles
    ]


def _calculate_total_distance(label_positions, domain_centroids):
    """Calculates the total distance from label positions to their domain centroids."""
    total_distance = 0
    for domain, pos in label_positions.items():
        centroid = domain_centroids[domain]
        total_distance += np.linalg.norm(centroid - pos)
    return total_distance


def _swap_and_evaluate(label_positions, i, j, domain_centroids):
    """Swaps two labels and evaluates the total distance after the swap."""
    labels = list(label_positions.keys())
    swapped_positions = label_positions.copy()
    swapped_positions[labels[i]], swapped_positions[labels[j]] = (
        swapped_positions[labels[j]],
        swapped_positions[labels[i]],
    )
    return _calculate_total_distance(swapped_positions, domain_centroids)
