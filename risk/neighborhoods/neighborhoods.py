"""
risk/neighborhoods/neighborhoods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import random
import warnings
from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np
from sklearn.exceptions import DataConversionWarning
from sklearn.metrics.pairwise import cosine_similarity

from risk.neighborhoods.community import (
    calculate_greedy_modularity_neighborhoods,
    calculate_label_propagation_neighborhoods,
    calculate_louvain_neighborhoods,
    calculate_markov_clustering_neighborhoods,
    calculate_spinglass_neighborhoods,
    calculate_walktrap_neighborhoods,
)
from risk.log import logger

# Suppress DataConversionWarning
warnings.filterwarnings(action="ignore", category=DataConversionWarning)


def get_network_neighborhoods(
    network: nx.Graph,
    distance_metric: str = "louvain",
    edge_length_threshold: float = 1.0,
    louvain_resolution: float = 1.0,
    random_seed: int = 888,
) -> np.ndarray:
    """Calculate the neighborhoods for each node in the network based on the specified distance metric.

    Args:
        network (nx.Graph): The network graph.
        distance_metric (str): The distance metric to use ('greedy_modularity', 'louvain', 'label_propagation',
            'markov_clustering', 'walktrap', 'spinglass').
        edge_length_threshold (float): The edge length threshold for the neighborhoods.
        louvain_resolution (float, optional): Resolution parameter for the Louvain method. Defaults to 1.0.
        random_seed (int, optional): Random seed for methods requiring random initialization. Defaults to 888.

    Returns:
        np.ndarray: Neighborhood matrix calculated based on the selected distance metric.
    """
    # Set random seed for reproducibility in all methods besides Louvain, which requires a separate seed
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Create a subgraph based on the edge length percentile threshold
    network = _create_percentile_limited_subgraph(
        network, edge_length_percentile=edge_length_threshold
    )

    if distance_metric == "louvain":
        return calculate_louvain_neighborhoods(network, louvain_resolution, random_seed=random_seed)
    if distance_metric == "greedy_modularity":
        return calculate_greedy_modularity_neighborhoods(network)
    if distance_metric == "label_propagation":
        return calculate_label_propagation_neighborhoods(network)
    if distance_metric == "markov_clustering":
        return calculate_markov_clustering_neighborhoods(network)
    if distance_metric == "walktrap":
        return calculate_walktrap_neighborhoods(network)
    if distance_metric == "spinglass":
        return calculate_spinglass_neighborhoods(network)

    raise ValueError(
        "Incorrect distance metric specified. Please choose from 'greedy_modularity', 'louvain',"
        "'label_propagation', 'markov_clustering', 'walktrap', 'spinglass'."
    )


def _create_percentile_limited_subgraph(G: nx.Graph, edge_length_percentile: float) -> nx.Graph:
    """Create a subgraph containing all nodes and edges where the edge length is below the
    specified percentile of all edge lengths in the input graph.

    Args:
        G (nx.Graph): The input graph with 'length' attributes on edges.
        edge_length_percentile (float): The percentile (between 0 and 1) to filter edges by length.

    Returns:
        nx.Graph: A subgraph with all nodes and edges where the edge length is below the
        calculated threshold length.
    """
    # Extract edge lengths and handle missing lengths
    edge_lengths = [d["length"] for _, _, d in G.edges(data=True) if "length" in d]
    if not edge_lengths:
        raise ValueError(
            "No edge lengths found in the graph. Ensure edges have 'length' attributes."
        )

    # Calculate the specific edge length for the given percentile
    percentile_length = np.percentile(edge_lengths, edge_length_percentile * 100)
    # Create the subgraph by directly filtering edges during iteration
    subgraph = nx.Graph()
    subgraph.add_nodes_from(G.nodes(data=True))  # Retain all nodes from the original graph
    # Add edges below the specified percentile length in a single pass
    for u, v, d in G.edges(data=True):
        if d.get("length", 1) <= percentile_length:
            subgraph.add_edge(u, v, **d)

    # Return the subgraph; optionally check if it's too sparse
    if subgraph.number_of_edges() == 0:
        raise Warning("The resulting subgraph has no edges. Consider adjusting the percentile.")

    return subgraph


def process_neighborhoods(
    network: nx.Graph,
    neighborhoods: Dict[str, Any],
    impute_depth: int = 0,
    prune_threshold: float = 0.0,
) -> Dict[str, Any]:
    """Process neighborhoods based on the imputation and pruning settings.

    Args:
        network (nx.Graph): The network data structure used for imputing and pruning neighbors.
        neighborhoods (dict): Dictionary containing 'enrichment_matrix', 'binary_enrichment_matrix', and 'significant_enrichment_matrix'.
        impute_depth (int, optional): Depth for imputing neighbors. Defaults to 0.
        prune_threshold (float, optional): Distance threshold for pruning neighbors. Defaults to 0.0.

    Returns:
        dict: Processed neighborhoods data, including the updated matrices and enrichment counts.
    """
    enrichment_matrix = neighborhoods["enrichment_matrix"]
    binary_enrichment_matrix = neighborhoods["binary_enrichment_matrix"]
    significant_enrichment_matrix = neighborhoods["significant_enrichment_matrix"]
    logger.debug(f"Imputation depth: {impute_depth}")
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

    logger.debug(f"Pruning threshold: {prune_threshold}")
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
    # Calculate the distance threshold value based on the shortest distances
    enrichment_matrix, binary_enrichment_matrix = _impute_neighbors_with_similarity(
        network, enrichment_matrix, binary_enrichment_matrix, max_depth=max_depth
    )
    # Create a matrix where non-significant entries are set to zero
    significant_enrichment_matrix = np.where(binary_enrichment_matrix == 1, enrichment_matrix, 0)

    return enrichment_matrix, binary_enrichment_matrix, significant_enrichment_matrix


def _impute_neighbors_with_similarity(
    network: nx.Graph,
    enrichment_matrix: np.ndarray,
    binary_enrichment_matrix: np.ndarray,
    max_depth: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Impute non-enriched nodes based on the closest enriched neighbors' profiles and their similarity.

    Args:
        network (nx.Graph): The network graph with nodes having IDs matching the matrix indices.
        enrichment_matrix (np.ndarray): The enrichment matrix with rows to be imputed.
        binary_enrichment_matrix (np.ndarray): The alpha threshold matrix to be imputed similarly.
        max_depth (int): Maximum depth of nodes to traverse for imputing values.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - The imputed enrichment matrix.
            - The imputed alpha threshold matrix.
    """
    depth = 1
    rows_to_impute = np.where(binary_enrichment_matrix.sum(axis=1) == 0)[0]
    while len(rows_to_impute) and depth <= max_depth:
        # Iterate over all enriched nodes
        for row_index in range(binary_enrichment_matrix.shape[0]):
            if binary_enrichment_matrix[row_index].sum() != 0:
                enrichment_matrix, binary_enrichment_matrix = _process_node_imputation(
                    row_index, network, enrichment_matrix, binary_enrichment_matrix, depth
                )

        # Update rows to impute for the next iteration
        rows_to_impute = np.where(binary_enrichment_matrix.sum(axis=1) == 0)[0]
        depth += 1

    return enrichment_matrix, binary_enrichment_matrix


def _process_node_imputation(
    row_index: int,
    network: nx.Graph,
    enrichment_matrix: np.ndarray,
    binary_enrichment_matrix: np.ndarray,
    depth: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Process the imputation for a single node based on its enriched neighbors.

    Args:
        row_index (int): The index of the enriched node being processed.
        network (nx.Graph): The network graph with nodes having IDs matching the matrix indices.
        enrichment_matrix (np.ndarray): The enrichment matrix with rows to be imputed.
        binary_enrichment_matrix (np.ndarray): The alpha threshold matrix to be imputed similarly.
        depth (int): Current depth for traversal.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The modified enrichment matrix and binary threshold matrix.
    """
    # Check neighbors at the current depth
    neighbors = nx.single_source_shortest_path_length(network, row_index, cutoff=depth)
    # Filter annotated neighbors (already enriched)
    annotated_neighbors = [
        n
        for n in neighbors
        if n != row_index
        and binary_enrichment_matrix[n].sum() != 0
        and enrichment_matrix[n].sum() != 0
    ]
    # Filter non-enriched neighbors
    valid_neighbors = [
        n
        for n in neighbors
        if n != row_index
        and binary_enrichment_matrix[n].sum() == 0
        and enrichment_matrix[n].sum() == 0
    ]
    # If there are valid non-enriched neighbors
    if valid_neighbors and annotated_neighbors:
        # Calculate distances to annotated neighbors
        distances_to_annotated = [
            _get_euclidean_distance(row_index, n, network) for n in annotated_neighbors
        ]
        # Calculate the IQR to identify outliers
        q1, q3 = np.percentile(distances_to_annotated, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        # Filter valid non-enriched neighbors that fall within the IQR bounds
        valid_neighbors_within_iqr = [
            n
            for n in valid_neighbors
            if lower_bound <= _get_euclidean_distance(row_index, n, network) <= upper_bound
        ]
        # If there are any valid neighbors within the IQR
        if valid_neighbors_within_iqr:
            # If more than one valid neighbor is within the IQR, compute pairwise cosine similarities
            if len(valid_neighbors_within_iqr) > 1:
                # Find the most similar neighbor based on pairwise cosine similarities
                def sum_pairwise_cosine_similarities(neighbor):
                    return sum(
                        cosine_similarity(
                            enrichment_matrix[neighbor].reshape(1, -1),
                            enrichment_matrix[other_neighbor].reshape(1, -1),
                        )[0][0]
                        for other_neighbor in valid_neighbors_within_iqr
                        if other_neighbor != neighbor
                    )

                most_similar_neighbor = max(
                    valid_neighbors_within_iqr, key=sum_pairwise_cosine_similarities
                )
            else:
                most_similar_neighbor = valid_neighbors_within_iqr[0]

            # Impute the most similar non-enriched neighbor with the enriched node's data, scaled by depth
            enrichment_matrix[most_similar_neighbor] = enrichment_matrix[row_index] / np.sqrt(
                depth + 1
            )
            binary_enrichment_matrix[most_similar_neighbor] = binary_enrichment_matrix[row_index]

    return enrichment_matrix, binary_enrichment_matrix


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
    median_distances = []
    for node in non_zero_indices:
        neighbors = [n for n in network.neighbors(node) if binary_enrichment_matrix[n].sum() != 0]
        if neighbors:
            median_distance = np.median(
                [_get_euclidean_distance(node, n, network) for n in neighbors]
            )
            median_distances.append(median_distance)

    # Calculate the distance threshold value based on rank
    distance_threshold_value = _calculate_threshold(median_distances, 1 - distance_threshold)
    # Prune nodes that are outliers based on the distance threshold
    for row_index in non_zero_indices:
        neighbors = [
            n for n in network.neighbors(row_index) if binary_enrichment_matrix[n].sum() != 0
        ]
        if neighbors:
            median_distance = np.median(
                [_get_euclidean_distance(row_index, n, network) for n in neighbors]
            )
            if median_distance >= distance_threshold_value:
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


def _calculate_threshold(median_distances: List, distance_threshold: float) -> float:
    """Calculate the distance threshold based on the given median distances and a percentile threshold.

    Args:
        median_distances (list): An array of median distances.
        distance_threshold (float): A percentile threshold (0 to 1) used to determine the distance cutoff.

    Returns:
        float: The calculated distance threshold value.
    """
    # Sort the median distances
    sorted_distances = np.sort(median_distances)
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
