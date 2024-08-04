from collections import defaultdict

import numpy as np
import pandas as pd
import random
import matplotlib
import matplotlib.cm as cm
from scipy.stats import zscore


class NetworkGraph:
    """A class to represent a network graph and process its nodes and edges."""

    def __init__(
        self,
        network,
        top_annotations,
        domains,
        trimmed_domains,
        node_label_to_id_map,
        node_enrichment_sums,
    ):
        """Initialize the NetworkGraph object.

        Args:
            network: The network graph.
            top_annotations: DataFrame of annotations data for the network nodes.
            domains: DataFrame of domain data for the network nodes.
            trimmed_domains: DataFrame of trimmed domain data for the network nodes.
        """
        self.top_annotations = top_annotations
        self.domain_to_nodes = self._create_domain_to_nodes_map(domains)
        self.domains = domains
        self.trimmed_domain_to_term = self._create_domain_to_term_map(trimmed_domains)
        self.trimmed_domains = trimmed_domains
        self.node_label_to_id_map = node_label_to_id_map
        self.node_enrichment_sums = node_enrichment_sums
        # NOTE: self.network and self.node_coordinates declared in _initialize_network
        self.network = None
        self.node_coordinates = None
        self._initialize_network(network)

    def _create_domain_to_nodes_map(self, domains):
        """Creates a mapping from domains to the list of nodes belonging to each domain."""
        cleaned_domains_matrix = domains.reset_index()[["index", "primary domain"]]
        node_to_domains = cleaned_domains_matrix.set_index("index")["primary domain"].to_dict()
        domain_to_nodes = defaultdict(list)
        for k, v in node_to_domains.items():
            domain_to_nodes[v].append(k)

        return domain_to_nodes

    def _create_domain_to_term_map(self, trimmed_domains):
        """Creates a mapping from domain IDs to their corresponding terms."""
        return dict(
            zip(
                trimmed_domains.index,
                trimmed_domains["label"],
            )
        )

    def _initialize_network(self, network):
        """Initialize the network by unfolding it and extracting node coordinates."""
        network_2d = _unfold_sphere_to_plane(network)
        # Unfold the network's 3D coordinates to 2D
        self.network = network_2d
        # Extract 2D coordinates of nodes
        self.node_coordinates = _extract_node_coordinates(network_2d)

    def get_domain_colors(self, min_scale=0.8, max_scale=1.0, random_seed=888, **kwargs):
        """Generate composite colors for domains.

        This method generates composite colors for nodes based on their enrichment scores and transforms
        them to ensure proper alpha values and intensity. For nodes with alpha == 0, it assigns new colors
        based on the closest valid neighbors within a specified distance.

        Returns:
            np.ndarray: Array of transformed colors.
        """
        # Get colors for each domain
        domain_colors = self._get_domain_colors(**kwargs, random_seed=random_seed)
        # Generate composite colors for nodes
        node_colors = self._get_composite_node_colors(domain_colors)
        # Transform colors to ensure proper alpha values and intensity
        transformed_colors = _transform_colors(
            node_colors,
            self.node_enrichment_sums,
            min_scale=min_scale,
            max_scale=max_scale,
        )

        return transformed_colors

    def _get_composite_node_colors(self, domain_colors):
        """Generate composite colors for nodes based on domain colors and counts.

        Args:
            domain_colors: Array of colors corresponding to each domain.

        Returns:
            composite_colors: Array of composite colors for each node.
        """
        # Determine the number of nodes
        num_nodes = len(self.node_coordinates)
        # Initialize composite colors array with shape (number of nodes, 4) for RGBA
        composite_colors = np.zeros((num_nodes, 4))

        # Assign colors to nodes based on domain_colors
        for domain_idx, nodes in self.domain_to_nodes.items():
            color = domain_colors[domain_idx]
            for node in nodes:
                composite_colors[node] = color

        return composite_colors

    def _get_domain_colors(self, **kwargs):
        """Get colors for each domain."""
        # Exclude non-numeric domain columns
        numeric_domains = [
            col for col in self.domains.columns if isinstance(col, (int, np.integer))
        ]
        domains = np.sort(numeric_domains)
        domain_colors = _get_colors(**kwargs, num_colors_to_generate=len(domains))
        return dict(zip(self.domain_to_nodes.keys(), domain_colors))


def _transform_colors(colors, enrichment_sums, min_scale=0.8, max_scale=1.0):
    """Transform colors to ensure proper alpha values and intensity based on enrichment sums."""
    if min_scale == max_scale:
        min_scale = max_scale - 0.001  # Avoid division by zero

    log_enrichment_sums = np.log1p(enrichment_sums)  # Use log1p to avoid log(0)
    # Normalize the capped enrichment sums to the range [0, 1]
    normalized_sums = log_enrichment_sums / np.max(log_enrichment_sums)
    # Scale normalized sums to the specified color range [min_scale, max_scale]
    scaled_sums = min_scale + (max_scale - min_scale) * normalized_sums
    # Adjust RGB values based on scaled sums
    for i in range(3):  # Only adjust RGB values
        colors[:, i] = scaled_sums * colors[:, i]

    return colors


def _unfold_sphere_to_plane(network):
    """Convert 3D coordinates to 2D by unfolding a sphere to a plane.

    Args:
        network: A network graph with 3D coordinates. Each node should have 'x', 'y', and 'z' attributes.

    Returns:
        network: The network graph with updated 2D coordinates (only 'x' and 'y').
    """
    for node in network.nodes():
        if "z" in network.nodes[node]:
            # Extract 3D coordinates
            x, y, z = network.nodes[node]["x"], network.nodes[node]["y"], network.nodes[node]["z"]
            # Calculate spherical coordinates theta and phi from Cartesian coordinates
            r = np.sqrt(x**2 + y**2 + z**2)
            theta = np.arctan2(y, x)
            phi = np.arccos(z / r)
            # Convert spherical coordinates to 2D plane coordinates
            unfolded_x = (theta + np.pi) / (2 * np.pi)  # Shift and normalize theta to [0, 1]
            unfolded_y = (np.pi - phi) / np.pi  # Reflect phi and normalize to [0, 1]
            unfolded_x = unfolded_x + 0.5 if unfolded_x < 0.5 else unfolded_x - 0.5
            # Update network node attributes
            network.nodes[node]["x"] = unfolded_x
            network.nodes[node]["y"] = -unfolded_y
            # Remove the 'z' coordinate as it's no longer needed
            del network.nodes[node]["z"]

    return network


def _extract_node_coordinates(graph):
    """Extract 2D coordinates of nodes from the graph.

    Args: graph: The network graph with node coordinates.
    Returns: node_coordinates: Array of node coordinates.
    """
    # Extract x and y coordinates from graph nodes
    x_coords = dict(graph.nodes.data("x"))
    y_coords = dict(graph.nodes.data("y"))
    coordinates_dicts = [x_coords, y_coords]
    # Combine x and y coordinates into a single array
    node_positions = {
        node: np.array([coords[node] for coords in coordinates_dicts]) for node in x_coords
    }
    node_coordinates = np.vstack(list(node_positions.values()))
    return node_coordinates


def _get_colors(num_colors_to_generate=10, cmap="hsv", random_seed=888, **kwargs):
    """Generate a list of RGBA colors from a specified cmap or use a direct color string.

    Args:
        num_colors_to_generate (int): The number of colors to generate.
        random_seed (int): Seed for random number generation.
        **kwargs: Additional keyword arguments for color map specification.

    Returns:
        List: List of RGBA colors.
    """
    # Set random seed for reproducibility
    random.seed(random_seed)

    if kwargs.get("color"):
        # If a direct color string is provided, generate a list with that color
        rgba = matplotlib.colors.to_rgba(kwargs["color"])
        rgbas = [rgba] * num_colors_to_generate
    else:
        colormap = cm.get_cmap(cmap)
        # Generate evenly distributed color positions
        color_positions = np.linspace(0, 1, num_colors_to_generate)
        random.shuffle(color_positions)  # Shuffle the positions to randomize colors
        # Generate colors based on shuffled positions
        rgbas = [colormap(pos) for pos in color_positions]

    return rgbas
