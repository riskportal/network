"""
risk/network/graph
~~~~~~~~~~~~~~~~~~
"""

from collections import defaultdict
from typing import Any, Dict, List, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib


class NetworkGraph:
    """A class to represent a network graph and process its nodes and edges.

    The NetworkGraph class provides functionality to handle and manipulate a network graph,
    including managing domains, annotations, and node enrichment data. It also includes methods
    for transforming and mapping graph coordinates, as well as generating colors based on node
    enrichment.
    """

    def __init__(
        self,
        network: nx.Graph,
        top_annotations: pd.DataFrame,
        domains: pd.DataFrame,
        trimmed_domains: pd.DataFrame,
        node_label_to_node_id_map: Dict[str, Any],
        node_enrichment_sums: np.ndarray,
    ):
        """Initialize the NetworkGraph object.

        Args:
            network (nx.Graph): The network graph.
            top_annotations (pd.DataFrame): DataFrame containing annotations data for the network nodes.
            domains (pd.DataFrame): DataFrame containing domain data for the network nodes.
            trimmed_domains (pd.DataFrame): DataFrame containing trimmed domain data for the network nodes.
            node_label_to_node_id_map (dict): A dictionary mapping node labels to their corresponding IDs.
            node_enrichment_sums (np.ndarray): Array containing the enrichment sums for the nodes.
        """
        self.top_annotations = top_annotations
        self.domain_id_to_node_ids_map = self._create_domain_id_to_node_ids_map(domains)
        self.domains = domains
        self.domain_id_to_domain_terms_map = self._create_domain_id_to_domain_terms_map(
            trimmed_domains
        )
        self.node_enrichment_sums = node_enrichment_sums
        self.node_id_to_node_label_map = {v: k for k, v in node_label_to_node_id_map.items()}
        self.node_label_to_enrichment_map = dict(
            zip(node_label_to_node_id_map.keys(), node_enrichment_sums)
        )
        self.node_label_to_node_id_map = node_label_to_node_id_map
        # NOTE: Below this point, instance attributes (i.e., self) will be used!
        self.domain_id_to_node_labels_map = self._create_domain_id_to_node_labels_map()
        # Unfold the network's 3D coordinates to 2D and extract node coordinates
        self.network = _unfold_sphere_to_plane(network)
        self.node_coordinates = _extract_node_coordinates(self.network)

    def _create_domain_id_to_node_ids_map(self, domains: pd.DataFrame) -> Dict[str, Any]:
        """Create a mapping from domains to the list of node IDs belonging to each domain.

        Args:
            domains (pd.DataFrame): DataFrame containing domain information, including the 'primary domain' for each node.

        Returns:
            dict: A dictionary where keys are domain IDs and values are lists of node IDs belonging to each domain.
        """
        cleaned_domains_matrix = domains.reset_index()[["index", "primary domain"]]
        node_to_domains_map = cleaned_domains_matrix.set_index("index")["primary domain"].to_dict()
        domain_id_to_node_ids_map = defaultdict(list)
        for k, v in node_to_domains_map.items():
            domain_id_to_node_ids_map[v].append(k)

        return domain_id_to_node_ids_map

    def _create_domain_id_to_domain_terms_map(
        self, trimmed_domains: pd.DataFrame
    ) -> Dict[str, Any]:
        """Create a mapping from domain IDs to their corresponding terms.

        Args:
            trimmed_domains (pd.DataFrame): DataFrame containing domain IDs and their corresponding labels.

        Returns:
            dict: A dictionary mapping domain IDs to their corresponding terms.
        """
        return dict(
            zip(
                trimmed_domains.index,
                trimmed_domains["label"],
            )
        )

    def _create_domain_id_to_node_labels_map(self) -> Dict[int, List[str]]:
        """Create a map from domain IDs to node labels.

        Returns:
            dict: A dictionary mapping domain IDs to the corresponding node labels.
        """
        domain_id_to_label_map = {}
        for domain_id, node_ids in self.domain_id_to_node_ids_map.items():
            domain_id_to_label_map[domain_id] = [
                self.node_id_to_node_label_map[node_id] for node_id in node_ids
            ]

        return domain_id_to_label_map

    def get_domain_colors(
        self,
        cmap: str = "gist_rainbow",
        color: Union[str, None] = None,
        min_scale: float = 0.8,
        max_scale: float = 1.0,
        scale_factor: float = 1.0,
        random_seed: int = 888,
    ) -> np.ndarray:
        """Generate composite colors for domains based on enrichment or specified colors.

        Args:
            cmap (str, optional): Name of the colormap to use for generating domain colors. Defaults to "gist_rainbow".
            color (str or None, optional): A specific color to use for all generated colors. Defaults to None.
            min_scale (float, optional): Minimum intensity scale for the colors generated by the colormap.
                Controls the dimmest colors. Defaults to 0.8.
            max_scale (float, optional): Maximum intensity scale for the colors generated by the colormap.
                Controls the brightest colors. Defaults to 1.0.
            scale_factor (float, optional): Exponent for adjusting the color scaling based on enrichment scores.
                A higher value increases contrast by dimming lower scores more. Defaults to 1.0.
            random_seed (int, optional): Seed for random number generation to ensure reproducibility of color assignments.
                Defaults to 888.

        Returns:
            np.ndarray: Array of RGBA colors generated for each domain, based on enrichment or the specified color.
        """
        # Get colors for each domain
        domain_colors = self._get_domain_colors(cmap=cmap, color=color, random_seed=random_seed)
        # Generate composite colors for nodes
        node_colors = self._get_composite_node_colors(domain_colors)
        # Transform colors to ensure proper alpha values and intensity
        transformed_colors = _transform_colors(
            node_colors,
            self.node_enrichment_sums,
            min_scale=min_scale,
            max_scale=max_scale,
            scale_factor=scale_factor,
        )

        return transformed_colors

    def _get_composite_node_colors(self, domain_colors: np.ndarray) -> np.ndarray:
        """Generate composite colors for nodes based on domain colors and counts.

        Args:
            domain_colors (np.ndarray): Array of colors corresponding to each domain.

        Returns:
            np.ndarray: Array of composite colors for each node.
        """
        # Determine the number of nodes
        num_nodes = len(self.node_coordinates)
        # Initialize composite colors array with shape (number of nodes, 4) for RGBA
        composite_colors = np.zeros((num_nodes, 4))
        # Assign colors to nodes based on domain_colors
        for domain_id, nodes in self.domain_id_to_node_ids_map.items():
            color = domain_colors[domain_id]
            for node in nodes:
                composite_colors[node] = color

        return composite_colors

    def _get_domain_colors(
        self,
        cmap: str = "gist_rainbow",
        color: Union[str, None] = None,
        random_seed: int = 888,
    ) -> Dict[str, Any]:
        """Get colors for each domain.

        Args:
            cmap (str, optional): The name of the colormap to use. Defaults to "gist_rainbow".
            color (str or None, optional): A specific color to use for all generated colors. Defaults to None.
            random_seed (int, optional): Seed for random number generation. Defaults to 888.

        Returns:
            dict: A dictionary mapping domain keys to their corresponding RGBA colors.
        """
        # Get colors for each domain based on node positions
        domain_colors = _get_colors(
            self.network,
            self.domain_id_to_node_ids_map,
            cmap=cmap,
            color=color,
            random_seed=random_seed,
        )
        self.network, self.domain_id_to_node_ids_map
        return dict(zip(self.domain_id_to_node_ids_map.keys(), domain_colors))


def _transform_colors(
    colors: np.ndarray,
    enrichment_sums: np.ndarray,
    min_scale: float = 0.8,
    max_scale: float = 1.0,
    scale_factor: float = 1.0,
) -> np.ndarray:
    """Transform colors using power scaling to emphasize high enrichment sums more.

    Args:
        colors (np.ndarray): An array of RGBA colors.
        enrichment_sums (np.ndarray): An array of enrichment sums corresponding to the colors.
        min_scale (float, optional): Minimum scale for color intensity. Defaults to 0.8.
        max_scale (float, optional): Maximum scale for color intensity. Defaults to 1.0.
        scale_factor (float, optional): Exponent for scaling, where values > 1 increase contrast by dimming small
            values more. Defaults to 1.0.

    Returns:
        np.ndarray: The transformed array of RGBA colors with adjusted intensities.
    """
    if min_scale == max_scale:
        min_scale = max_scale - 10e-6  # Avoid division by zero

    # Normalize the enrichment sums to the range [0, 1]
    normalized_sums = enrichment_sums / np.max(enrichment_sums)
    # Apply power scaling to dim lower values and emphasize higher values
    scaled_sums = normalized_sums**scale_factor
    # Linearly scale the normalized sums to the range [min_scale, max_scale]
    scaled_sums = min_scale + (max_scale - min_scale) * scaled_sums
    # Adjust RGB values based on scaled sums
    for i in range(3):  # Only adjust RGB values
        colors[:, i] = scaled_sums * colors[:, i]

    return colors


def _unfold_sphere_to_plane(G: nx.Graph) -> nx.Graph:
    """Convert 3D coordinates to 2D by unfolding a sphere to a plane.

    Args:
        G (nx.Graph): A network graph with 3D coordinates. Each node should have 'x', 'y', and 'z' attributes.

    Returns:
        nx.Graph: The network graph with updated 2D coordinates (only 'x' and 'y').
    """
    for node in G.nodes():
        if "z" in G.nodes[node]:
            # Extract 3D coordinates
            x, y, z = G.nodes[node]["x"], G.nodes[node]["y"], G.nodes[node]["z"]
            # Calculate spherical coordinates theta and phi from Cartesian coordinates
            r = np.sqrt(x**2 + y**2 + z**2)
            theta = np.arctan2(y, x)
            phi = np.arccos(z / r)

            # Convert spherical coordinates to 2D plane coordinates
            unfolded_x = (theta + np.pi) / (2 * np.pi)  # Shift and normalize theta to [0, 1]
            unfolded_x = unfolded_x + 0.5 if unfolded_x < 0.5 else unfolded_x - 0.5
            unfolded_y = (np.pi - phi) / np.pi  # Reflect phi and normalize to [0, 1]
            # Update network node attributes
            G.nodes[node]["x"] = unfolded_x
            G.nodes[node]["y"] = -unfolded_y
            # Remove the 'z' coordinate as it's no longer needed
            del G.nodes[node]["z"]

    return G


def _extract_node_coordinates(G: nx.Graph) -> np.ndarray:
    """Extract 2D coordinates of nodes from the graph.

    Args:
        G (nx.Graph): The network graph with node coordinates.

    Returns:
        np.ndarray: Array of node coordinates with shape (num_nodes, 2).
    """
    # Extract x and y coordinates from graph nodes
    x_coords = dict(G.nodes.data("x"))
    y_coords = dict(G.nodes.data("y"))
    coordinates_dicts = [x_coords, y_coords]
    # Combine x and y coordinates into a single array
    node_positions = {
        node: np.array([coords[node] for coords in coordinates_dicts]) for node in x_coords
    }
    node_coordinates = np.vstack(list(node_positions.values()))
    return node_coordinates


def _get_colors(
    network,
    domain_id_to_node_ids_map,
    cmap: str = "gist_rainbow",
    color: Union[str, None] = None,
    random_seed: int = 888,
) -> List[Tuple]:
    """Generate a list of RGBA colors based on domain centroids, ensuring that domains
    close in space get maximally separated colors, while keeping some randomness.

    Args:
        network (NetworkX graph): The graph representing the network.
        domain_id_to_node_ids_map (dict): Mapping from domain IDs to lists of node IDs.
        cmap (str, optional): The name of the colormap to use. Defaults to "gist_rainbow".
        color (str or None, optional): A specific color to use for all generated colors.
        random_seed (int): Seed for random number generation. Defaults to 888.

    Returns:
        List[Tuple]: List of RGBA colors.
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    # Determine the number of colors to generate based on the number of domains
    num_colors_to_generate = len(domain_id_to_node_ids_map)
    if color:
        # Generate all colors as the same specified color
        rgba = matplotlib.colors.to_rgba(color)
        return [rgba] * num_colors_to_generate

    # Load colormap
    colormap = matplotlib.colormaps.get_cmap(cmap)
    # Step 1: Calculate centroids for each domain
    centroids = _calculate_centroids(network, domain_id_to_node_ids_map)
    # Step 2: Calculate pairwise distances between centroids
    centroid_array = np.array(centroids)
    dist_matrix = np.linalg.norm(centroid_array[:, None] - centroid_array, axis=-1)
    # Step 3: Assign distant colors to close centroids
    color_positions = _assign_distant_colors(dist_matrix, num_colors_to_generate)
    # Step 4: Randomly shift the entire color palette while maintaining relative distances
    global_shift = np.random.uniform(-0.1, 0.1)  # Small global shift to change the overall palette
    color_positions = (color_positions + global_shift) % 1  # Wrap around to keep within [0, 1]
    # Step 5: Ensure that all positions remain between 0 and 1
    color_positions = np.clip(color_positions, 0, 1)

    # Step 6: Generate RGBA colors based on positions
    return [colormap(pos) for pos in color_positions]


def _calculate_centroids(network, domain_id_to_node_ids_map):
    """Calculate the centroid for each domain based on node x and y coordinates in the network.

    Args:
        network (NetworkX graph): The graph representing the network.
        domain_id_to_node_ids_map (dict): Mapping from domain IDs to lists of node IDs.

    Returns:
        List[Tuple[float, float]]: List of centroids (x, y) for each domain.
    """
    centroids = []
    for domain_id, node_ids in domain_id_to_node_ids_map.items():
        # Extract x and y coordinates from the network nodes
        node_positions = np.array(
            [[network.nodes[node_id]["x"], network.nodes[node_id]["y"]] for node_id in node_ids]
        )
        # Compute the centroid as the mean of the x and y coordinates
        centroid = np.mean(node_positions, axis=0)
        centroids.append(tuple(centroid))

    return centroids


def _assign_distant_colors(dist_matrix, num_colors_to_generate):
    """Assign colors to centroids that are close in space, ensuring stark color differences.

    Args:
        dist_matrix (ndarray): Matrix of pairwise centroid distances.
        num_colors_to_generate (int): Number of colors to generate.

    Returns:
        np.array: Array of color positions in the range [0, 1].
    """
    color_positions = np.zeros(num_colors_to_generate)
    # Step 1: Sort indices by centroid proximity (based on sum of distances to others)
    proximity_order = sorted(
        range(num_colors_to_generate), key=lambda idx: np.sum(dist_matrix[idx])
    )
    # Step 2: Assign colors starting with the most distant points in proximity order
    for i, idx in enumerate(proximity_order):
        color_positions[idx] = i / num_colors_to_generate

    # Step 3: Adjust colors so that centroids close to one another are maximally distant on the color spectrum
    half_spectrum = int(num_colors_to_generate / 2)
    for i in range(half_spectrum):
        # Split the spectrum so that close centroids are assigned distant colors
        color_positions[proximity_order[i]] = (i * 2) / num_colors_to_generate
        color_positions[proximity_order[-(i + 1)]] = ((i * 2) + 1) / num_colors_to_generate

    return color_positions
