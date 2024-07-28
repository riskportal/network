import networkx as nx
import numpy as np
import pandas as pd
import random
import matplotlib.cm as cm


class NetworkGraph:
    """A class to represent a network graph and process its nodes and edges."""

    def __init__(
        self,
        network,
        annotation_matrix,
        domains_matrix,
        trimmed_domains_matrix,
        neighborhood_binary_enrichment_matrix_below_alpha,
        random_seed=888,
    ):
        """Initialize the NetworkGraph object.

        Args:
            network: The network graph.
            annotation_matrix: DataFrame of annotations data for the network nodes.
            domains_matrix: DataFrame of domain data for the network nodes.
            trimmed_domains_matrix: DataFrame of trimmed domain data for the network nodes.
            neighborhood_binary_enrichment_matrix_below_alpha: Matrix of neighborhood binary enrichment data.
            random_seed: Seed for random number generation.
        """
        self.annotation_matrix = annotation_matrix
        self.domains_matrix = domains_matrix
        self.trimmed_domains_matrix = trimmed_domains_matrix
        self.neighborhood_binary_enrichment_matrix_below_alpha = (
            neighborhood_binary_enrichment_matrix_below_alpha
        )
        self.random_seed = random_seed
        # NOTE: self.network, self.node_coordinates, self.node_order declared in _initialize_network
        self._initialize_network(network)
        self._clean_matrices()

    def _initialize_network(self, network):
        """Initialize the network by unfolding it and extracting node coordinates."""
        # Unfold the network's 3D coordinates to 2D
        self.network = _unfold_sphere_to_plane(network)
        # Extract 2D coordinates of nodes
        self.node_coordinates = _extract_node_coordinates(self.network)

    def _clean_matrices(self):
        """Remove invalid domains from matrices."""
        invalid_domain_id = 888888
        # Remove invalid domains from annotations matrix
        self.annotation_matrix = _remove_invalid_domains_with_id(
            self.annotation_matrix, "domain", invalid_domain_id
        )
        # Remove invalid domains from domains matrix
        self.domains_matrix = _remove_invalid_domains_with_id(
            self.domains_matrix, "primary domain", invalid_domain_id
        )
        # Remove invalid domains from trimmed domains matrix
        self.trimmed_domains_matrix = _remove_invalid_domains_with_id(
            self.trimmed_domains_matrix, "id", invalid_domain_id
        )

    def get_domain_colors(self, colormap="hsv"):
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
            self._get_domain_colors(colormap=colormap),
            node_to_domain_count,
            random_seed=self.random_seed,
        )
        # Transform colors to ensure proper alpha values and intensity
        transformed_colors = self._transform_colors(composite_colors)

        return transformed_colors

    def _create_node_to_enrichment_score_binary(self):
        """Create a DataFrame of node to enrichment score binary values."""
        return pd.DataFrame(
            data=self.neighborhood_binary_enrichment_matrix_below_alpha[
                :, self.annotation_matrix.index.values
            ],
            columns=[self.annotation_matrix.index.values, self.annotation_matrix["domain"]],
        )

    def _calculate_node_to_domain_count(self, node_to_enrichment_score_binary):
        """Calculate the count of nodes per domain."""
        return node_to_enrichment_score_binary.groupby(level="domain", axis=1).sum()

    def _get_domain_colors(self, colormap="hsv"):
        """Get colors for each domain."""
        # Exclude non-numeric domain columns
        numeric_domains = [
            col for col in self.domains_matrix.columns if isinstance(col, (int, np.integer))
        ]
        domains = np.sort(numeric_domains)
        domain_colors = _get_colors(colormap=colormap, num_colors_to_generate=len(domains))
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


def _remove_invalid_domains_with_id(df, column, domain_id):
    """Remove rows from DataFrame where the specified column matches the domain_id.

    Args: df: The DataFrame to filter. column: The column to check for invalid domain IDs. domain_id: The domain ID to filter out.
    Returns: df: The filtered DataFrame with rows containing the invalid domain ID removed.
    """
    return df[df[column] != domain_id]


def _get_composite_node_colors(domain2rgb, node_to_domain_count, random_seed=888):
    """Generate composite colors for nodes based on domain colors and counts.

    Args: domain2rgb: Array of colors corresponding to each domain. node_to_domain_count: DataFrame of domain counts for each node. random_seed: Seed for random number generation.
    Returns: composite_colors: Array of composite colors for each node.
    """
    random.seed(random_seed)
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


def _get_colors(colormap="plasma", num_colors_to_generate=10, random_seed=888):
    """Generate a list of RGBA colors from a specified colormap.

    Args:
        colormap: The name of the colormap to use for generating colors.
        num_colors_to_generate: The number of colors to generate.
        random_seed: Seed for random number generation.

    Returns:
        rgbas: List of RGBA colors.
    """
    random.seed(random_seed)
    # Get the specified colormap
    cmap = cm.get_cmap(colormap)
    # Generate evenly distributed color positions
    color_positions = np.linspace(0, 1, num_colors_to_generate)
    random.shuffle(color_positions)  # Shuffle the positions to randomize colors
    # Generate colors based on shuffled positions
    rgbas = [cmap(pos) for pos in color_positions]
    return rgbas
