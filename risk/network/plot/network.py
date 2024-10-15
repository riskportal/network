"""
risk/network/plot/network
~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from typing import Any, List, Tuple, Union

import networkx as nx
import numpy as np

from risk.log import params
from risk.network.graph import NetworkGraph
from risk.network.plot.utils import to_rgba


class Network:
    """Class for plotting nodes and edges in a network graph."""

    def __init__(self, graph: NetworkGraph, ax: Any = None) -> None:
        """Initialize the NetworkPlotter class.

        Args:
            graph (NetworkGraph): The network data and attributes to be visualized.
            ax (Any, optional): Axes object to plot the network graph. Defaults to None.
        """
        self.graph = graph
        self.ax = ax

    def plot_network(
        self,
        node_size: Union[int, np.ndarray] = 50,
        node_shape: str = "o",
        node_edgewidth: float = 1.0,
        edge_width: float = 1.0,
        node_color: Union[str, List, Tuple, np.ndarray] = "white",
        node_edgecolor: Union[str, List, Tuple, np.ndarray] = "black",
        edge_color: Union[str, List, Tuple, np.ndarray] = "black",
        node_alpha: Union[float, None] = 1.0,
        edge_alpha: Union[float, None] = 1.0,
    ) -> None:
        """Plot the network graph with customizable node colors, sizes, edge widths, and node edge widths.

        Args:
            node_size (int or np.ndarray, optional): Size of the nodes. Can be a single integer or an array of sizes. Defaults to 50.
            node_shape (str, optional): Shape of the nodes. Defaults to "o".
            node_edgewidth (float, optional): Width of the node edges. Defaults to 1.0.
            edge_width (float, optional): Width of the edges. Defaults to 1.0.
            node_color (str, list, tuple, or np.ndarray, optional): Color of the nodes. Can be a single color or an array of colors.
                Defaults to "white".
            node_edgecolor (str, list, tuple, or np.ndarray, optional): Color of the node edges. Defaults to "black".
            edge_color (str, list, tuple, or np.ndarray, optional): Color of the edges. Defaults to "black".
            node_alpha (float, None, optional): Alpha value (transparency) for the nodes. If provided, it overrides any existing alpha
                values found in node_color. Defaults to 1.0. Annotated node_color alphas will override this value.
            edge_alpha (float, None, optional): Alpha value (transparency) for the edges. If provided, it overrides any existing alpha
                values found in edge_color. Defaults to 1.0.
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

        # Convert colors to RGBA using the to_rgba helper function
        # If node_colors was generated using get_annotated_node_colors, its alpha values will override node_alpha
        node_color = to_rgba(
            color=node_color, alpha=node_alpha, num_repeats=len(self.graph.network.nodes)
        )
        node_edgecolor = to_rgba(
            color=node_edgecolor, alpha=1.0, num_repeats=len(self.graph.network.nodes)
        )
        edge_color = to_rgba(
            color=edge_color, alpha=edge_alpha, num_repeats=len(self.graph.network.edges)
        )

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
        node_alpha: Union[float, None] = None,
        edge_alpha: Union[float, None] = None,
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
            node_alpha (float, None, optional): Transparency for the nodes. If provided, it overrides any existing alpha values
                found in node_color. Defaults to 1.0.
            edge_alpha (float, None, optional): Transparency for the edges. If provided, it overrides any existing alpha values
                found in node_color. Defaults to 1.0.

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

        # Convert colors to RGBA using the to_rgba helper function
        node_color = to_rgba(color=node_color, alpha=node_alpha, num_repeats=len(node_ids))
        node_edgecolor = to_rgba(color=node_edgecolor, alpha=1.0, num_repeats=len(node_ids))
        edge_color = to_rgba(
            color=edge_color, alpha=edge_alpha, num_repeats=len(self.graph.network.edges)
        )

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

    def get_annotated_node_colors(
        self,
        cmap: str = "gist_rainbow",
        color: Union[str, None] = None,
        min_scale: float = 0.8,
        max_scale: float = 1.0,
        scale_factor: float = 1.0,
        alpha: Union[float, None] = 1.0,
        nonenriched_color: Union[str, List, Tuple, np.ndarray] = "white",
        nonenriched_alpha: Union[float, None] = 1.0,
        random_seed: int = 888,
    ) -> np.ndarray:
        """Adjust the colors of nodes in the network graph based on enrichment.

        Args:
            cmap (str, optional): Colormap to use for coloring the nodes. Defaults to "gist_rainbow".
            color (str or None, optional): Color to use for the nodes. If None, the colormap will be used. Defaults to None.
            min_scale (float, optional): Minimum scale for color intensity. Defaults to 0.8.
            max_scale (float, optional): Maximum scale for color intensity. Defaults to 1.0.
            scale_factor (float, optional): Factor for adjusting the color scaling intensity. Defaults to 1.0.
            alpha (float, None, optional): Alpha value for enriched nodes. If provided, it overrides any existing alpha values
                found in color. Defaults to 1.0.
            nonenriched_color (str, list, tuple, or np.ndarray, optional): Color for non-enriched nodes. Defaults to "white".
            nonenriched_alpha (float, None, optional): Alpha value for non-enriched nodes. If provided, it overrides any existing
                alpha values found in nonenriched_color. Defaults to 1.0.
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
        # Convert the non-enriched color to RGBA using the to_rgba helper function
        nonenriched_color = to_rgba(color=nonenriched_color, alpha=nonenriched_alpha)
        # Adjust node colors: replace any fully black nodes (RGB == 0) with the non-enriched color and its alpha
        adjusted_network_colors = np.where(
            np.all(network_colors[:, :3] == 0, axis=1, keepdims=True),  # Check RGB values only
            np.array(nonenriched_color),  # Apply the non-enriched color with alpha
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
