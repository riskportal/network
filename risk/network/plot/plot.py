from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.ndimage import label
from scipy.stats import gaussian_kde


class NetworkPlotter:
    """A class to handle plotting of network graphs."""

    def __init__(self, network_graph):
        """Initialize the NetworkPlotter with a NetworkGraph object.

        Args:
            network_graph: A NetworkGraph object containing the network's data and attributes.
        """
        self.network_graph = network_graph
        self.domain_to_nodes = self._create_domain_to_nodes_map()
        self.domain_to_term = self._create_domain_to_term_map()
        self.ax = None

    def _create_domain_to_nodes_map(self):
        """Creates a mapping from domains to the list of nodes belonging to each domain."""
        cleaned_domains_matrix = self.network_graph.domains_matrix.reset_index()[
            ["index", "primary domain"]
        ]
        node_domain_map = cleaned_domains_matrix.set_index("index")["primary domain"].to_dict()
        domain_to_nodes = defaultdict(list)
        for k, v in node_domain_map.items():
            domain_to_nodes[v].append(k)
        return domain_to_nodes

    def _create_domain_to_term_map(self):
        """Creates a mapping from domain IDs to their corresponding terms."""
        return dict(
            zip(
                self.network_graph.trimmed_domains_matrix["id"],
                self.network_graph.trimmed_domains_matrix["label"],
            )
        )

    def adjust_colors_and_sizes(self, enriched_nodesize=50, nonenriched_nodesize=25):
        """Adjusts the colors and sizes of nodes in the network graph.

        Args:
            enriched_nodesize: Size for enriched nodes.
            nonenriched_nodesize: Size for non-enriched nodes.

        Returns:
            Tuple of two elements (adjusted colors array, node sizes array).
        """
        adjusted_network_colors = np.where(
            np.all(self.network_graph.colors == 0, axis=1, keepdims=True),
            np.array([[1.0, 1.0, 1.0, 1.0]]),
            self.network_graph.colors,
        )
        node_sizes = self._adjust_node_sizes(
            adjusted_network_colors, enriched_nodesize, nonenriched_nodesize
        )
        return adjusted_network_colors, node_sizes

    def _adjust_node_sizes(self, array, enriched_nodesize=30, nonenriched_nodesize=10):
        """Adjusts node sizes based on whether a row has been enriched (converted from all zeros)."""
        is_nonenriched = np.all(array[:, :-1] == 1, axis=1) & (array[:, -1] == 1)
        node_sizes = np.where(is_nonenriched, nonenriched_nodesize, enriched_nodesize)
        return node_sizes

    def calculate_domain_centroids(self):
        """Calculates the most centrally located node within each domain based on the node positions."""
        domain_central_nodes = {}
        for domain, nodes in self.domain_to_nodes.items():
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

    def get_annotated_node_colors_and_sizes(self, enriched_nodesize=50, nonenriched_nodesize=25):
        """Get annotated node colors and sizes based on the network graph."""
        node_colors, node_sizes = self.adjust_colors_and_sizes(
            enriched_nodesize=enriched_nodesize, nonenriched_nodesize=nonenriched_nodesize
        )
        return node_colors, node_sizes

    def initialize_plot(
        self,
        figsize=(10, 10),
        background_color="white",
        network_perimeter_color="black",
        plot_network_perimeter=True,
    ):
        """Initialize the plot with figure size, optional circle perimeter, and background color.

        Args:
            figsize (tuple, optional): Size of the figure. Defaults to (10, 10).
            background_color (str, optional): Background color of the plot. Defaults to "white".
            network_perimeter_color (str, optional): Color of the network perimeter circle. Defaults to "black".
            plot_network_perimeter (bool, optional): Whether to plot the network perimeter circle. Defaults to True.
        """
        node_coordinates = self.network_graph.node_coordinates
        center, radius = self._calculate_bounding_box(node_coordinates)

        fig, ax = plt.subplots(figsize=figsize)
        fig.tight_layout()

        if plot_network_perimeter:
            circle = plt.Circle(
                center,
                radius,
                linestyle="--",
                color=network_perimeter_color,
                fill=False,
                linewidth=1.5,
            )
            ax.add_artist(circle)

        ax.set_xlim([center[0] - radius - 0.3, center[0] + radius + 0.3])
        ax.set_ylim([center[1] - radius - 0.3, center[1] + radius + 0.3])
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

    def plot_network(
        self,
        node_sizes=50,
        edge_widths=1.0,
        node_colors="white",
        edge_colors="black",
        node_edge_colors="black",
    ):
        """Plot the network graph with customizable node colors, sizes, and edge widths."""
        node_coordinates = self.network_graph.node_coordinates

        nx.draw_networkx_nodes(
            self.network_graph.network,
            pos=node_coordinates,
            node_size=node_sizes,
            node_color=node_colors,
            alpha=1.00,
            edgecolors=node_edge_colors,
            ax=self.ax,
        )
        nx.draw_networkx_edges(
            self.network_graph.network,
            pos=node_coordinates,
            width=edge_widths,
            edge_color=edge_colors,
            ax=self.ax,
        )

    def plot_contours(self, levels=5, bandwidth=0.8, grid_size=200, alpha=0.5):
        """Draws KDE contours for nodes in various domains of a network graph, highlighting areas of high density."""
        if self.ax is None:
            raise RuntimeError("Network must be plotted before plotting contours.")

        node_coordinates = self.network_graph.node_coordinates
        node_colors = self.network_graph.colors

        for domain, nodes in self.domain_to_nodes.items():
            if len(nodes) > 1:
                domain_colors = np.array([node_colors[node] for node in nodes])
                brightest_color = domain_colors[np.argmax(domain_colors.sum(axis=1))]
                self._draw_kde_contour(
                    self.ax,
                    node_coordinates,
                    nodes,
                    color=brightest_color,
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
            thresholded_grid = self._generate_thresholded_grid(z)
            connected = self._is_connected(thresholded_grid)
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

    def _is_connected(self, thresholded_grid):
        """Determines if a thresholded grid represents a single, connected component."""
        labeled_array, num_features = label(thresholded_grid)
        return num_features == 1

    def _generate_thresholded_grid(self, z, threshold=0.05):
        """Generates a thresholded grid from a KDE grid by applying a threshold value."""
        return z > threshold

    def plot_labels(
        self,
        radius_margin=1.05,
        offset=0.10,
        font="Arial",
        fontsize=10,
        color="black",
        arrow_linewidth=1,
        max_words_per_label=10,
    ):
        """Annotates a network graph with labels for different domains, positioned around the network for clarity."""
        if self.ax is None:
            raise RuntimeError("Network must be plotted before plotting labels.")

        domain_centroids = self.calculate_domain_centroids()
        center, radius = self._calculate_bounding_box(
            self.network_graph.node_coordinates, radius_margin=radius_margin
        )
        label_positions = self._determine_label_positions(domain_centroids, center, radius, offset)
        num_domains = len(domain_centroids)
        equidistant_positions = self._equidistant_angles_around_center(
            center, radius, offset, num_domains
        )
        label_positions = {
            domain: position
            for domain, position in zip(domain_centroids.keys(), equidistant_positions)
        }
        best_label_positions = self._optimize_label_positions(label_positions, domain_centroids)

        for domain, pos in best_label_positions.items():
            centroid = domain_centroids[domain]
            annotation = "\n".join(self.domain_to_term[domain].split(",")[:max_words_per_label])
            self.ax.annotate(
                annotation,
                xy=centroid,
                xytext=pos,
                textcoords="data",
                ha="center",
                va="center",
                fontsize=fontsize,
                fontname=font,
                color=color,
                arrowprops=dict(arrowstyle="->", color=color, linewidth=arrow_linewidth),
            )

    def _calculate_bounding_box(self, node_coordinates, radius_margin=1.05):
        """Calculates the bounding box of the network based on node coordinates."""
        x_min, y_min = np.min(node_coordinates, axis=0)
        x_max, y_max = np.max(node_coordinates, axis=0)
        center = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2])
        radius = max(x_max - x_min, y_max - y_min) / 2 * radius_margin
        return center, radius

    def _determine_label_positions(self, domain_centroids, center, radius, label_offset):
        """Determines the label positions around the circle perimeter based on domain centroids."""
        label_positions = {}
        for domain, centroid in domain_centroids.items():
            direction = centroid - center
            direction /= np.linalg.norm(direction)
            label_positions[domain] = center + direction * (radius + label_offset)
        return label_positions

    def _equidistant_angles_around_center(self, center, radius, label_offset, num_domains):
        """Calculates positions around a center at equidistant angles."""
        angles = np.linspace(0, 2 * np.pi, num_domains, endpoint=False)
        return [
            center + (radius + label_offset) * np.array([np.cos(angle), np.sin(angle)])
            for angle in angles
        ]

    def _optimize_label_positions(self, best_label_positions, domain_centroids):
        """Optimizes label positions around the perimeter to minimize total distance to centroids."""
        improvement = True
        while improvement:
            improvement = False
            for i in range(len(domain_centroids)):
                for j in range(i + 1, len(domain_centroids)):
                    current_distance = self._calculate_total_distance(
                        best_label_positions, domain_centroids
                    )
                    swapped_distance = self._swap_and_evaluate(
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

    def _calculate_total_distance(self, label_positions, domain_centroids):
        """Calculates the total distance from label positions to their domain centroids."""
        total_distance = 0
        for domain, pos in label_positions.items():
            centroid = domain_centroids[domain]
            total_distance += np.linalg.norm(centroid - pos)
        return total_distance

    def _swap_and_evaluate(self, label_positions, i, j, domain_centroids):
        """Swaps two labels and evaluates the total distance after the swap."""
        labels = list(label_positions.keys())
        swapped_positions = label_positions.copy()
        swapped_positions[labels[i]], swapped_positions[labels[j]] = (
            swapped_positions[labels[j]],
            swapped_positions[labels[i]],
        )
        return self._calculate_total_distance(swapped_positions, domain_centroids)

    def savefig(self, *args, **kwargs):
        """Save the current plot to a file."""
        plt.savefig(*args, bbox_inches="tight", **kwargs)

    def show(self, *args, **kwargs):
        """Display the current plot."""
        plt.show(*args, **kwargs)
