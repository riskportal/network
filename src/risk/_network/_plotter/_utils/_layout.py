"""
risk/_network/_plotter/_utils/_layout
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np


def calculate_bounding_box(
    node_coordinates: np.ndarray, radius_margin: float = 1.05
) -> Tuple[Tuple, float]:
    """
    Calculate the bounding box of the network based on node coordinates.

    Args:
        node_coordinates (np.ndarray): Array of node coordinates (x, y).
        radius_margin (float, optional): Margin factor to apply to the bounding box radius. Defaults to 1.05.

    Returns:
        Tuple[Tuple, float]: Center (x, y) and radius of the bounding box.
    """
    # Find minimum and maximum x, y coordinates
    x_min, y_min = np.min(node_coordinates, axis=0)
    x_max, y_max = np.max(node_coordinates, axis=0)
    # Calculate the center of the bounding box
    center = ((x_min + x_max) / 2, (y_min + y_max) / 2)
    # Calculate the radius of the bounding box, adjusted by the margin
    radius = max(x_max - x_min, y_max - y_min) / 2 * radius_margin
    return center, radius


def refine_center_iteratively(
    node_coordinates: np.ndarray,
    radius_margin: float = 1.05,
    max_iterations: int = 10,
    tolerance: float = 1e-2,
) -> Tuple[np.ndarray, float]:
    """
    Refine the center of the graph iteratively to minimize skew in node distribution.

    Args:
        node_coordinates (np.ndarray): Array of node coordinates (x, y).
        radius_margin (float, optional): Margin factor to apply to the bounding box radius. Defaults to 1.05.
        max_iterations (int, optional): Maximum number of iterations for refining the center. Defaults to 10.
        tolerance (float, optional): Stopping tolerance for center adjustment. Defaults to 1e-2.

    Returns:
        tuple: Refined center and the final radius.
    """
    # Initial center and radius based on the bounding box
    center, _ = calculate_bounding_box(node_coordinates, radius_margin)
    for _ in range(max_iterations):
        # Shift the coordinates based on the current center
        shifted_coordinates = node_coordinates - center
        # Calculate skew (difference in distance from the center)
        skew = np.mean(shifted_coordinates, axis=0)
        # If skew is below tolerance, stop
        if np.linalg.norm(skew) < tolerance:
            break

        # Adjust the center by moving it in the direction opposite to the skew
        center += skew

    # After refinement, recalculate the bounding radius
    shifted_coordinates = node_coordinates - center
    new_radius = np.max(np.linalg.norm(shifted_coordinates, axis=1)) * radius_margin

    return center, new_radius


def calculate_centroids(
    network: nx.Graph, domain_id_to_node_ids_map: Dict[int, Any]
) -> List[Tuple[float, float]]:
    """
    Calculate the centroid for each domain based on node x and y coordinates in the network.

    Args:
        network (nx.Graph): The graph representing the network.
        domain_id_to_node_ids_map (Dict[int, Any]): Mapping from domain IDs to lists of node IDs.

    Returns:
        List[Tuple[float, float]]: List of centroids (x, y) for each domain.
    """
    centroids = []
    for _, node_ids in domain_id_to_node_ids_map.items():
        # Extract x and y coordinates from the network nodes
        node_positions = np.array(
            [[network.nodes[node_id]["x"], network.nodes[node_id]["y"]] for node_id in node_ids]
        )
        # Compute the centroid as the mean of the x and y coordinates
        centroid = np.mean(node_positions, axis=0)
        centroids.append(tuple(centroid))

    return centroids
