"""
risk/neighborhoods/neighborhoods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import warnings
from typing import Any, Dict, Tuple

import networkx as nx
import numpy as np
from sklearn.exceptions import DataConversionWarning

from risk.neighborhoods.community import (
    calculate_dijkstra_neighborhoods,
    calculate_label_propagation_neighborhoods,
    calculate_louvain_neighborhoods,
    calculate_markov_clustering_neighborhoods,
    calculate_spinglass_neighborhoods,
    calculate_walktrap_neighborhoods,
)

# Suppress DataConversionWarning
warnings.filterwarnings(action="ignore", category=DataConversionWarning)


def get_network_neighborhoods(
    network: nx.Graph,
    distance_metric: str = "dijkstra",
    edge_length_threshold: float = 1.0,
    louvain_resolution: float = 1.0,
    random_seed: int = 888,
) -> np.ndarray:
    """Calculate the neighborhoods for each node in the network based on the specified distance metric.

    Args:
        network (nx.Graph): The network graph.
        distance_metric (str): The distance metric to use ('euclidean', 'dijkstra', 'louvain', 'affinity_propagation',
            'label_propagation', 'markov_clustering', 'walktrap', 'spinglass').
        edge_length_threshold (float): The edge length threshold for the neighborhoods.
        louvain_resolution (float, optional): Resolution parameter for the Louvain method. Defaults to 1.0.
        random_seed (int, optional): Random seed for methods requiring random initialization. Defaults to 888.

    Returns:
        np.ndarray: Neighborhood matrix calculated based on the selected distance metric.
    """
    network = _create_percentile_limited_subgraph(network, edge_length_threshold)

    if distance_metric == "dijkstra":
        return calculate_dijkstra_neighborhoods(network)
    if distance_metric == "louvain":
        return calculate_louvain_neighborhoods(network, louvain_resolution, random_seed=random_seed)
    if distance_metric == "label_propagation":
        return calculate_label_propagation_neighborhoods(network)
    if distance_metric == "markov_clustering":
        return calculate_markov_clustering_neighborhoods(network)
    if distance_metric == "walktrap":
        return calculate_walktrap_neighborhoods(network)
    if distance_metric == "spinglass":
        return calculate_spinglass_neighborhoods(network)

    raise ValueError(
        "Incorrect distance metric specified. Please choose from 'dijkstra', 'louvain',"
        "'label_propagation', 'markov_clustering', 'walktrap', 'spinglass'."
    )


def _create_percentile_limited_subgraph(G: nx.Graph, edge_length_percentile: float) -> nx.Graph:
    """Calculate the edge length corresponding to the given percentile of edge lengths in the graph
    and create a subgraph with all nodes and edges below this length.

    Args:
        G (nx.Graph): The input graph.
        edge_length_percentile (float): The percentile to calculate (between 0 and 1).

    Returns:
        nx.Graph: A subgraph with all nodes and edges below the edge length corresponding to the given percentile.
    """
    # Extract edge lengths from the graph
    edge_lengths = [d["length"] for _, _, d in G.edges(data=True) if "length" in d]
    # Calculate the specific edge length for the given percentile
    percentile_length = np.percentile(edge_lengths, edge_length_percentile * 100)
    # Create a new graph with all nodes from the original graph
    subgraph = nx.Graph()
    subgraph.add_nodes_from(G.nodes(data=True))
    # Add edges to the subgraph if they are below the specified percentile length
    for u, v, d in G.edges(data=True):
        if d.get("length", 1) <= percentile_length:
            subgraph.add_edge(u, v, **d)

    return subgraph


def process_neighborhoods(
    network: nx.Graph,
    neighborhoods: Dict[str, Any],
    impute_depth: int = 1,
    prune_threshold: float = 0.0,
) -> Dict[str, Any]:
    """Process neighborhoods based on the imputation and pruning settings.

    Args:
        network (nx.Graph): The network data structure used for imputing and pruning neighbors.
        neighborhoods (dict): Dictionary containing 'enrichment_matrix', 'binary_enrichment_matrix', and 'significant_enrichment_matrix'.
        impute_depth (int, optional): Depth for imputing neighbors. Defaults to 1.
        prune_threshold (float, optional): Distance threshold for pruning neighbors. Defaults to 0.0.

    Returns:
        dict: Processed neighborhoods data, including the updated matrices and enrichment counts.
    """
    enrichment_matrix = neighborhoods["enrichment_matrix"]
    binary_enrichment_matrix = neighborhoods["binary_enrichment_matrix"]
    significant_enrichment_matrix = neighborhoods["significant_enrichment_matrix"]
    print(f"Imputation depth: {impute_depth}")
    if impute_depth:
        (
            enrichment_matrix,
            binary_enrichment_matrix,
            significant_enrichment_matrix,
        ) = _impute_neighbors(
            network,
            enrichment_matrix,
            binary_enrichment_matrix,
            max_depth=impute_depth,
        )

    print(f"Pruning threshold: {prune_threshold}")
    if prune_threshold:
        (
            enrichment_matrix,
            binary_enrichment_matrix,
            significant_enrichment_matrix,
        ) = _prune_neighbors(
            network,
            enrichment_matrix,
            binary_enrichment_matrix,
            distance_threshold=prune_threshold,
        )

    neighborhood_enrichment_counts = np.sum(binary_enrichment_matrix, axis=0)
    node_enrichment_sums = np.sum(enrichment_matrix, axis=1)
    return {
        "enrichment_matrix": enrichment_matrix,
        "binary_enrichment_matrix": binary_enrichment_matrix,
        "significant_enrichment_matrix": significant_enrichment_matrix,
        "neighborhood_enrichment_counts": neighborhood_enrichment_counts,
        "node_enrichment_sums": node_enrichment_sums,
    }


def _impute_neighbors(
    network: nx.Graph,
    enrichment_matrix: np.ndarray,
    binary_enrichment_matrix: np.ndarray,
    max_depth: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Impute rows with sums of zero in the enrichment matrix based on the closest non-zero neighbors in the network graph.

    Args:
        network (nx.Graph): The network graph with nodes having IDs matching the matrix indices.
        enrichment_matrix (np.ndarray): The enrichment matrix with rows to be imputed.
        binary_enrichment_matrix (np.ndarray): The alpha threshold matrix to be imputed similarly.
        max_depth (int): Maximum depth of nodes to traverse for imputing values.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: The imputed enrichment matrix.
            - np.ndarray: The imputed alpha threshold matrix.
            - np.ndarray: The significant enrichment matrix with non-significant entries set to zero.
    """
    # Calculate shortest distances for each node to determine the distance threshold
    shortest_distances = []
    for node in network.nodes():
        try:
            neighbors = [
                n for n in network.neighbors(node) if binary_enrichment_matrix[n].sum() != 0
            ]
        except IndexError as e:
            raise IndexError(
                f"Failed to find neighbors for node '{node}': Ensure that the node exists in the network and that the binary enrichment matrix is correctly indexed."
            ) from e

        # Calculate the shortest distance to a neighbor
        if neighbors:
            shortest_distance = min([_get_euclidean_distance(node, n, network) for n in neighbors])
            shortest_distances.append(shortest_distance)

    depth = 1
    rows_to_impute = np.where(binary_enrichment_matrix.sum(axis=1) == 0)[0]
    while len(rows_to_impute) and depth <= max_depth:
        next_rows_to_impute = []
        for row_index in rows_to_impute:
            neighbors = nx.single_source_shortest_path_length(network, row_index, cutoff=depth)
            valid_neighbors = [
                n
                for n in neighbors
                if n != row_index
                and binary_enrichment_matrix[n].sum() != 0
                and enrichment_matrix[n].sum() != 0
            ]
            if valid_neighbors:
                closest_neighbor = min(
                    valid_neighbors, key=lambda n: _get_euclidean_distance(row_index, n, network)
                )
                # Impute the row with the closest valid neighbor's data
                enrichment_matrix[row_index] = enrichment_matrix[closest_neighbor]
                binary_enrichment_matrix[row_index] = binary_enrichment_matrix[
                    closest_neighbor
                ] / np.sqrt(depth + 1)
            else:
                next_rows_to_impute.append(row_index)

        rows_to_impute = next_rows_to_impute
        depth += 1

    # Create a matrix where non-significant entries are set to zero
    significant_enrichment_matrix = np.where(binary_enrichment_matrix == 1, enrichment_matrix, 0)

    return enrichment_matrix, binary_enrichment_matrix, significant_enrichment_matrix


def _prune_neighbors(
    network: nx.Graph,
    enrichment_matrix: np.ndarray,
    binary_enrichment_matrix: np.ndarray,
    distance_threshold: float = 0.9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remove outliers based on their rank for edge lengths.

    Args:
        network (nx.Graph): The network graph with nodes having IDs matching the matrix indices.
        enrichment_matrix (np.ndarray): The enrichment matrix.
        binary_enrichment_matrix (np.ndarray): The alpha threshold matrix.
        distance_threshold (float): Rank threshold (0 to 1) to determine outliers.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: The updated enrichment matrix with outliers set to zero.
            - np.ndarray: The updated alpha threshold matrix with outliers set to zero.
            - np.ndarray: The significant enrichment matrix, where non-significant entries are set to zero.
    """
    # Identify indices with non-zero rows in the binary enrichment matrix
    non_zero_indices = np.where(binary_enrichment_matrix.sum(axis=1) != 0)[0]
    average_distances = []
    for node in non_zero_indices:
        neighbors = [n for n in network.neighbors(node) if binary_enrichment_matrix[n].sum() != 0]
        if neighbors:
            average_distance = np.mean(
                [_get_euclidean_distance(node, n, network) for n in neighbors]
            )
            average_distances.append(average_distance)

    # Calculate the distance threshold value based on rank
    distance_threshold_value = _calculate_threshold(average_distances, 1 - distance_threshold)
    # Prune nodes that are outliers based on the distance threshold
    for row_index in non_zero_indices:
        neighbors = [
            n for n in network.neighbors(row_index) if binary_enrichment_matrix[n].sum() != 0
        ]
        if neighbors:
            average_distance = np.mean(
                [_get_euclidean_distance(row_index, n, network) for n in neighbors]
            )
            if average_distance >= distance_threshold_value:
                enrichment_matrix[row_index] = 0
                binary_enrichment_matrix[row_index] = 0

    # Create a matrix where non-significant entries are set to zero
    significant_enrichment_matrix = np.where(binary_enrichment_matrix == 1, enrichment_matrix, 0)

    return enrichment_matrix, binary_enrichment_matrix, significant_enrichment_matrix


def _get_euclidean_distance(node1: Any, node2: Any, network: nx.Graph) -> float:
    """Calculate the Euclidean distance between two nodes in the network.

    Args:
        node1 (Any): The first node.
        node2 (Any): The second node.
        network (nx.Graph): The network graph containing the nodes.

    Returns:
        float: The Euclidean distance between the two nodes.
    """
    pos1 = _get_node_position(network, node1)
    pos2 = _get_node_position(network, node2)
    return np.linalg.norm(pos1 - pos2)


def _get_node_position(network: nx.Graph, node: Any) -> np.ndarray:
    """Retrieve the position of a node in the network as a numpy array.

    Args:
        network (nx.Graph): The network graph containing node positions.
        node (Any): The node for which the position is being retrieved.

    Returns:
        np.ndarray: A numpy array representing the position of the node in the format [x, y, z].
    """
    return np.array(
        [
            network.nodes[node].get(coord, 0)
            for coord in ["x", "y", "z"]
            if coord in network.nodes[node]
        ]
    )


def _calculate_threshold(average_distances: list, distance_threshold: float) -> float:
    """Calculate the distance threshold based on the given average distances and a percentile threshold.

    Args:
        average_distances (list): An array of average distances.
        distance_threshold (float): A percentile threshold (0 to 1) used to determine the distance cutoff.

    Returns:
        float: The calculated distance threshold value.
    """
    # Sort the average distances
    sorted_distances = np.sort(average_distances)
    # Compute the rank percentiles for the sorted distances
    rank_percentiles = np.linspace(0, 1, len(sorted_distances))
    # Interpolating the ranks to 1000 evenly spaced percentiles
    interpolated_percentiles = np.linspace(0, 1, 1000)
    try:
        smoothed_distances = np.interp(interpolated_percentiles, rank_percentiles, sorted_distances)
    except ValueError as e:
        raise ValueError("No significant annotations found.") from e

    # Determine the index corresponding to the distance threshold
    threshold_index = int(np.ceil(distance_threshold * len(smoothed_distances))) - 1
    # Return the smoothed distance at the calculated index
    return smoothed_distances[threshold_index]
