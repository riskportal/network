"""
risk/network/plot/utils/layout
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from typing import Tuple

import numpy as np


def calculate_bounding_box(
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


def calculate_centroids(network, domain_id_to_node_ids_map):
    """Calculate the centroid for each domain based on node x and y coordinates in the network.

    Args:
        network (NetworkX graph): The graph representing the network.
        domain_id_to_node_ids_map (Dict[int, Any]): Mapping from domain IDs to lists of node IDs.

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
