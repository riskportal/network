from collections import defaultdict

import numpy as np
import pandas as pd
import random
import matplotlib
import matplotlib.cm as cm


class NetworkGraph:
    """A class to represent a network graph and process its nodes and edges."""

    def __init__(
        self,
        network,
        annotation_matrix,
        domains_matrix,
        trimmed_domains_matrix,
        neighborhoods_binary_enrichment_matrix,
    ):
        """Initialize the NetworkGraph object.

        Args:
            network: The network graph.
            annotation_matrix: DataFrame of annotations data for the network nodes.
            domains_matrix: DataFrame of domain data for the network nodes.
            trimmed_domains_matrix: DataFrame of trimmed domain data for the network nodes.
            neighborhoods_binary_enrichment_matrix: Matrix of neighborhood binary enrichment data.
        """
        self.annotation_matrix = annotation_matrix
        self.domain_to_nodes = self._create_domain_to_nodes_map(domains_matrix)
        self.domains_matrix = domains_matrix
        self.trimmed_domain_to_term = self._create_domain_to_term_map(trimmed_domains_matrix)
        self.trimmed_domains_matrix = trimmed_domains_matrix
        self.neighborhoods_binary_enrichment_matrix = neighborhoods_binary_enrichment_matrix
        # NOTE: self.network and self.node_coordinates declared in _initialize_network
        self.network = None
        self.node_coordinates = None
        self._initialize_network(network)

    def _create_domain_to_nodes_map(self, domains_matrix):
        """Creates a mapping from domains to the list of nodes belonging to each domain."""
        cleaned_domains_matrix = domains_matrix.reset_index()[["index", "primary domain"]]
        node_domain_map = cleaned_domains_matrix.set_index("index")["primary domain"].to_dict()
        domain_to_nodes = defaultdict(list)
        for k, v in node_domain_map.items():
            domain_to_nodes[v].append(k)
        return domain_to_nodes

    def _create_domain_to_term_map(self, trimmed_domains_matrix):
        """Creates a mapping from domain IDs to their corresponding terms."""
        return dict(
            zip(
                trimmed_domains_matrix.index,
                trimmed_domains_matrix["label"],
            )
        )

    def _initialize_network(self, network):
        """Initialize the network by unfolding it and extracting node coordinates."""
        # Unfold the network's 3D coordinates to 2D
        self.network = _unfold_sphere_to_plane(network)
        # Extract 2D coordinates of nodes
        self.node_coordinates = _extract_node_coordinates(self.network)

    def get_domain_colors(self, random_seed=888, **kwargs):
        """Generate composite colors for domains.

        This method generates composite colors for nodes based on their enrichment scores and transforms
        them to ensure proper alpha values and intensity. For nodes with alpha == 0, it assigns new colors
        based on the closest valid neighbors within a specified distance.

        Returns:
            np.ndarray: Array of transformed colors.
        """
        # Create a DataFrame of node to enrichment score binary values
        node_to_enrichment_score_binary = self._create_node_to_enrichment_score_binary()
        # Calculate the count of nodes per domain
        node_to_domain_count = self._calculate_node_to_domain_count(node_to_enrichment_score_binary)
        # Generate composite colors for nodes
        composite_colors = _get_composite_node_colors(
            self._get_domain_colors(**kwargs, random_seed=random_seed),
            node_to_domain_count,
        )
        # Transform colors to ensure proper alpha values and intensity
        transformed_colors = self._transform_colors(composite_colors)

        return transformed_colors

    def _create_node_to_enrichment_score_binary(self):
        """Create a DataFrame of node to enrichment score binary values."""
        return pd.DataFrame(
            data=self.neighborhoods_binary_enrichment_matrix[
                :, self.annotation_matrix.index.values
            ],
            columns=[self.annotation_matrix.index.values, self.annotation_matrix["domain"]],
        )

    def _calculate_node_to_domain_count(self, node_to_enrichment_score_binary):
        """Calculate the count of nodes per domain."""
        return node_to_enrichment_score_binary.groupby(level="domain", axis=1).sum()

    def _get_domain_colors(self, **kwargs):
        """Get colors for each domain."""
        # Exclude non-numeric domain columns
        numeric_domains = [
            col for col in self.domains_matrix.columns if isinstance(col, (int, np.integer))
        ]
        domains = np.sort(numeric_domains)
        domain_colors = _get_colors(**kwargs, num_colors_to_generate=len(domains))
        return domain_colors

    def _transform_colors(self, colors):
        """Transform colors to ensure proper alpha values and intensity."""
        # Identify rows where alpha is 1.0
        rows_with_alpha_one = colors[:, 3] == 1
        # Generate random weights for color transformation
        random_weights = np.random.uniform(0.80, 1.00, colors[rows_with_alpha_one].shape[0])
        transformed_weights = 1.0 - (1.0 - random_weights) ** 2
        # Apply transformations to colors
        colors[rows_with_alpha_one, :3] *= random_weights[:, np.newaxis]
        colors[rows_with_alpha_one, 3] *= transformed_weights
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


def _get_composite_node_colors(domain2rgb, node_to_domain_count):
    """Generate composite colors for nodes based on domain colors and counts.

    Args: domain2rgb: Array of colors corresponding to each domain. node_to_domain_count: DataFrame of domain counts for each node.
    Returns: composite_colors: Array of composite colors for each node.
    """
    # Ensure domain2rgb is a numpy array
    if not isinstance(domain2rgb, np.ndarray):
        domain2rgb = np.array(domain2rgb)

    # Initialize composite colors array
    composite_colors = np.zeros((node_to_domain_count.shape[0], 4))  # Assuming RGBA
    for node_idx in range(node_to_domain_count.shape[0]):
        # Get domain counts for the current node
        domain_counts = node_to_domain_count.values[node_idx, :]
        max_count = np.max(domain_counts)
        # Normalize domain counts to avoid division by zero
        normalized_counts = domain_counts / max_count if max_count > 0 else domain_counts
        weighted_color_sum = np.zeros(4)
        # Compute the weighted sum of colors
        for domain_idx, count in enumerate(normalized_counts):
            color = domain2rgb[domain_idx]
            weighted_color = color * count
            weighted_color_sum += weighted_color

        # Average the colors if there are non-zero domains
        non_zero_domains = np.count_nonzero(normalized_counts)
        if non_zero_domains > 0:
            composite_color = weighted_color_sum / non_zero_domains
            composite_colors[node_idx] = composite_color

    # Replace NaNs with zeros
    composite_colors = np.nan_to_num(composite_colors)
    return composite_colors


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
