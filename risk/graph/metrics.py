"""
risk/network/graph
~~~~~~~~~~~~~~~~~~
"""

import os
import sys
from contextlib import contextmanager
from tqdm import tqdm

import networkx as nx
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist

from risk.neighborhoods import get_network_neighborhoods


def get_best_surface_depth(
    G,
    compute_sphere=True,
    include_edge_weight=True,
    distance_metric="euclidean",
    neighborhood_diameter=0.5,
    louvain_resolution=None,
    random_walk_length=3,
    random_walk_num=250,
    lower_bound=0,
    upper_bound=1.024,
):
    """Find the optimal surface depth for the network.

    Args:
        G (NetworkX graph): The network graph.
        lower_bound (int): Lower bound for surface depth.
        upper_bound (int): Upper bound for surface depth.

    Returns:
        int: The best surface depth.
    """
    print("Warning: Optimizing surface depth is an expensive process...")
    max_score = -np.inf
    best_surface_depth = lower_bound
    total_iterations = int(np.ceil(np.log2((upper_bound - lower_bound) * 1000)))

    with tqdm(
        total=total_iterations,
        desc="Optimizing surface depth",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    ) as pbar:
        current_iteration = 0

        while upper_bound - lower_bound > 0.001:  # Use 1 as the minimum difference for convergence
            if current_iteration == 0:
                G_test_zero = calculate_edge_lengths(
                    G.copy(),
                    compute_sphere=compute_sphere,
                    include_edge_weight=include_edge_weight,
                    surface_depth=upper_bound,
                )
                with _suppress_print():
                    neighborhoods_test_zero = get_network_neighborhoods(
                        network=G_test_zero,
                        distance_metric=distance_metric,
                        neighborhood_diameter=neighborhood_diameter,
                        compute_sphere=compute_sphere,
                        louvain_resolution=louvain_resolution,
                        random_walk_length=random_walk_length,
                        random_walk_num=random_walk_num,
                    )
                max_value = np.max(neighborhoods_test_zero)
                distance_matrix = max_value - neighborhoods_test_zero
                np.fill_diagonal(distance_matrix, 0)
                distance_matrix = np.maximum(distance_matrix, 0)
                score_test = _compute_silhouette_score(distance_matrix)

                if score_test > max_score:
                    max_score = score_test
                    best_surface_depth = 0

            else:
                mid_surface_depth = (lower_bound + upper_bound) / 2
                G_test = calculate_edge_lengths(
                    G.copy(),
                    compute_sphere=compute_sphere,
                    include_edge_weight=include_edge_weight,
                    surface_depth=mid_surface_depth,
                )
                with _suppress_print():
                    neighborhoods_test = get_network_neighborhoods(
                        network=G_test,
                        distance_metric=distance_metric,
                        neighborhood_diameter=neighborhood_diameter,
                        compute_sphere=compute_sphere,
                        louvain_resolution=louvain_resolution,
                        random_walk_length=random_walk_length,
                        random_walk_num=random_walk_num,
                    )
                max_value = np.max(neighborhoods_test)
                distance_matrix = max_value - neighborhoods_test
                np.fill_diagonal(distance_matrix, 0)
                distance_matrix = np.maximum(distance_matrix, 0)
                score_test = _compute_silhouette_score(distance_matrix)

                if score_test > max_score:
                    max_score = score_test
                    best_surface_depth = mid_surface_depth

                if best_surface_depth == mid_surface_depth + 1:
                    lower_bound = mid_surface_depth
                else:
                    upper_bound = mid_surface_depth

            current_iteration += 1
            pbar.update(1)

    print(f"Optimal surface depth: {best_surface_depth}")
    return best_surface_depth


def calculate_edge_lengths(G, compute_sphere=True, include_edge_weight=False, surface_depth=0.0):
    # Normalize graph coordinates
    _normalize_graph_coordinates(G)
    # Normalize weights
    _normalize_weights(G)
    # Conditionally map nodes to a sphere based on `compute_sphere`
    if compute_sphere:
        _map_to_sphere(G)
        # Identify subclusters - TODO: play with this value and research optimal radius! So far, 1 works best
        neighborhood_radius = np.pi / 2  # (4 * 1.0 (normalized diameter))
        partition = _find_subclusters_with_shortest_path(G, neighborhood_radius)
        # Create surface depth
        _create_depth(G, partition, surface_depth=surface_depth)

    for u, v, edge_data in G.edges(data=True):
        if compute_sphere:
            u_coords = np.array([G.nodes[u]["x"], G.nodes[u]["y"], G.nodes[u].get("z", 0)])
            v_coords = np.array([G.nodes[v]["x"], G.nodes[v]["y"], G.nodes[v].get("z", 0)])
            # Normalize vectors
            u_coords /= np.linalg.norm(u_coords)
            v_coords /= np.linalg.norm(v_coords)
            # Calculate the spherical distance
            dist = np.arccos(np.clip(np.dot(u_coords, v_coords), -1.0, 1.0))
        else:
            # If not computing sphere, use only x, y for planar distance
            u_coords = np.array([G.nodes[u]["x"], G.nodes[u]["y"]])
            v_coords = np.array([G.nodes[v]["x"], G.nodes[v]["y"]])
            # Calculate the planar distance
            dist = np.linalg.norm(u_coords - v_coords)

        if include_edge_weight and "normalized_weight" in edge_data:
            # Invert the weight influence such that higher weights bring nodes closer
            G.edges[u, v]["length"] = dist / (
                edge_data["normalized_weight"] + 10e-12  # Avoid division by zero
            )
        else:
            G.edges[u, v]["length"] = dist  # Use calculated distance directly

    return G


def _map_to_sphere(G):
    # Extract x, y coordinates from the graph nodes
    xy_coords = np.array([[G.nodes[node]["x"], G.nodes[node]["y"]] for node in G.nodes()])

    # Normalize the coordinates between [0, 1]
    min_vals = np.min(xy_coords, axis=0)
    max_vals = np.max(xy_coords, axis=0)
    normalized_xy = (xy_coords - min_vals) / (max_vals - min_vals)

    # Map normalized coordinates to theta and phi on a sphere
    theta = normalized_xy[:, 0] * np.pi * 2
    phi = normalized_xy[:, 1] * np.pi

    # Convert spherical coordinates to Cartesian coordinates for 3D sphere
    for i, node in enumerate(G.nodes()):
        x = np.sin(phi[i]) * np.cos(theta[i])
        y = np.sin(phi[i]) * np.sin(theta[i])
        z = np.cos(phi[i])
        G.nodes[node]["x"] = x
        G.nodes[node]["y"] = y
        G.nodes[node]["z"] = z


def _find_subclusters_with_shortest_path(G, neighborhood_radius):
    # Compute Djikstra's shortest path for each pair of nodes within a specified cutoff
    all_shortest_paths = dict(
        nx.all_pairs_dijkstra_path_length(
            G,
            weight="weight",
            cutoff=neighborhood_radius,
        )
    )
    # Identify subclusters based on the shortest path lengths
    subclusters = {}
    for source, targets in all_shortest_paths.items():
        for target, length in targets.items():
            if length <= neighborhood_radius:
                if source not in subclusters:
                    subclusters[source] = set()
                subclusters[source].add(target)
    return subclusters


def _create_depth(G, subclusters, surface_depth=0.20):
    # Create a strength metric for subclusters (here using size)
    subcluster_strengths = {node: len(neighbors) for node, neighbors in subclusters.items()}
    # Normalize the subcluster strengths and apply depths
    max_strength = max(subcluster_strengths.values())
    for node, strength in subcluster_strengths.items():
        normalized_surface_depth = (strength / max_strength) * surface_depth
        x, y, z = G.nodes[node]["x"], G.nodes[node]["y"], G.nodes[node]["z"]
        norm = np.sqrt(x**2 + y**2 + z**2)
        G.nodes[node]["z"] -= (z / norm) * normalized_surface_depth  # Adjust Z for a depth


def _normalize_graph_coordinates(G):
    # Extract x, y coordinates from the graph nodes
    xy_coords = np.array([[G.nodes[node]["x"], G.nodes[node]["y"]] for node in G.nodes()])

    # Calculate min and max values for x and y
    min_vals = np.min(xy_coords, axis=0)
    max_vals = np.max(xy_coords, axis=0)

    # Normalize the coordinates to [0, 1]
    normalized_xy = (xy_coords - min_vals) / (max_vals - min_vals)

    # Update the node coordinates with the normalized values
    for i, node in enumerate(G.nodes()):
        G.nodes[node]["x"], G.nodes[node]["y"] = normalized_xy[i]


def _normalize_weights(G):
    weights = [data["weight"] for _, _, data in G.edges(data=True) if "weight" in data]
    if weights:  # Ensure there are weighted edges
        min_weight = min(weights)
        max_weight = max(weights)
        range_weight = max_weight - min_weight if max_weight > min_weight else 1
        for u, v, data in G.edges(data=True):
            if "weight" in data:
                data["normalized_weight"] = (data["weight"] - min_weight) / range_weight


def _compute_silhouette_score(neighborhoods, linkage_method="average"):
    """Compute the silhouette score for a given graph and neighborhoods.

    Args:
        neighborhoods (numpy.ndarray): Neighborhood matrix.
        linkage_method (str): The linkage method to use for clustering.

    Returns:
        float: The silhouette score.
    """
    # Calculate the maximum value in the neighborhoods matrix
    max_value = np.max(neighborhoods)
    # Compute the distance matrix by subtracting neighborhoods from the max value
    distance_matrix = max_value - neighborhoods
    # Ensure the diagonal elements are zero (distance to itself is zero)
    np.fill_diagonal(distance_matrix, 0)
    # Ensure all distance values are non-negative
    distance_matrix = np.maximum(distance_matrix, 0)
    # Compute the silhouette score using hierarchical clustering
    return _compute_silhouette_with_validation(distance_matrix, linkage_method)


def _compute_silhouette_with_validation(distance_matrix, linkage_method="average"):
    """Compute the silhouette score for hierarchical clustering.

    Args:
        distance_matrix (np.ndarray): The distance matrix.
        linkage_method (str): The linkage method to use for hierarchical clustering.

    Returns:
        float: The best silhouette score found.
    """
    # Ensure all values in the distance matrix are non-negative
    distance_matrix = np.maximum(distance_matrix, 0)

    # Perform hierarchical clustering
    Z = linkage(pdist(distance_matrix), method=linkage_method)

    # Initialize variables to keep track of the best score
    best_score = float("-inf")
    num_clusters = 2

    while num_clusters < len(distance_matrix):
        # Generate cluster labels for the current number of clusters
        labels = fcluster(Z, t=num_clusters, criterion="maxclust")

        # Ensure there is more than one cluster
        if len(set(labels)) > 1:
            try:
                # Compute the silhouette score for the current clustering
                score = silhouette_score(distance_matrix, labels, metric="precomputed")
                # Update the best score if the current score is better
                if score > best_score:
                    best_score = score
                break
            except ValueError:
                pass  # Continue to the next iteration if an error occurs

        num_clusters += 1

    # Assign a default score if no valid clustering is found
    if best_score == float("-inf"):
        best_score = 0.0
        print("Valid clustering could not be achieved. Returning default score of 0.0.")

    return best_score


@contextmanager
def _suppress_print():
    """Context manager to suppress print statements."""
    original_stdout = sys.stdout
    try:
        with open(os.devnull, "w") as devnull:
            sys.stdout = devnull
            yield
    finally:
        sys.stdout = original_stdout
