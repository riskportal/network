"""
risk/_neighborhoods/_neighborhoods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import random
import warnings
from typing import Any, Dict, List, Tuple, Union

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.exceptions import DataConversionWarning
from sklearn.metrics.pairwise import cosine_similarity

from .._log import logger
from ._community import (
    calculate_greedy_modularity_neighborhoods,
    calculate_label_propagation_neighborhoods,
    calculate_leiden_neighborhoods,
    calculate_louvain_neighborhoods,
    calculate_markov_clustering_neighborhoods,
    calculate_spinglass_neighborhoods,
    calculate_walktrap_neighborhoods,
)

# Suppress DataConversionWarning
warnings.filterwarnings(action="ignore", category=DataConversionWarning)


def get_network_neighborhoods(
    network: nx.Graph,
    distance_metric: Union[str, List, Tuple, np.ndarray] = "louvain",
    fraction_shortest_edges: Union[float, List, Tuple, np.ndarray] = 1.0,
    louvain_resolution: float = 0.1,
    leiden_resolution: float = 1.0,
    random_seed: int = 888,
) -> csr_matrix:
    """
    Calculate the combined neighborhoods for each node using sparse matrices.

    Args:
        network (nx.Graph): The network graph.
        distance_metric (str, List, Tuple, or np.ndarray, optional): The distance metric(s) to use.
        fraction_shortest_edges (float, List, Tuple, or np.ndarray, optional): Shortest edge rank fraction thresholds.
        louvain_resolution (float, optional): Resolution parameter for the Louvain method.
        leiden_resolution (float, optional): Resolution parameter for the Leiden method.
        random_seed (int, optional): Random seed for methods requiring random initialization.

    Returns:
        csr_matrix: The combined neighborhood matrix.

    Raises:
        ValueError: If the number of distance metrics does not match the number of edge length thresholds.
    """
    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Ensure distance_metric is a list for multi-algorithm handling
    if isinstance(distance_metric, (str, np.ndarray)):
        distance_metric = [distance_metric]
    # Ensure fraction_shortest_edges is a list for multi-threshold handling
    if isinstance(fraction_shortest_edges, (float, int)):
        fraction_shortest_edges = [fraction_shortest_edges] * len(distance_metric)
    # Validate matching lengths of distance metrics and thresholds
    if len(distance_metric) != len(fraction_shortest_edges):
        raise ValueError(
            "The number of distance metrics must match the number of edge length thresholds."
        )

    # Initialize a sparse LIL matrix for incremental updates
    num_nodes = network.number_of_nodes()
    # Initialize a sparse matrix with the same shape as the network
    combined_neighborhoods = csr_matrix((num_nodes, num_nodes), dtype=np.uint8)
    # Loop through each distance metric and corresponding edge rank fraction
    for metric, percentile in zip(distance_metric, fraction_shortest_edges):
        # Compute neighborhoods for the specified metric
        if metric == "greedy_modularity":
            neighborhoods = calculate_greedy_modularity_neighborhoods(
                network, fraction_shortest_edges=percentile
            )
        elif metric == "label_propagation":
            neighborhoods = calculate_label_propagation_neighborhoods(
                network, fraction_shortest_edges=percentile
            )
        elif metric == "leiden":
            neighborhoods = calculate_leiden_neighborhoods(
                network,
                resolution=leiden_resolution,
                fraction_shortest_edges=percentile,
                random_seed=random_seed,
            )
        elif metric == "louvain":
            neighborhoods = calculate_louvain_neighborhoods(
                network,
                resolution=louvain_resolution,
                fraction_shortest_edges=percentile,
                random_seed=random_seed,
            )
        elif metric == "markov_clustering":
            neighborhoods = calculate_markov_clustering_neighborhoods(
                network, fraction_shortest_edges=percentile
            )
        elif metric == "spinglass":
            neighborhoods = calculate_spinglass_neighborhoods(
                network, fraction_shortest_edges=percentile
            )
        elif metric == "walktrap":
            neighborhoods = calculate_walktrap_neighborhoods(
                network, fraction_shortest_edges=percentile
            )
        else:
            raise ValueError(
                "Invalid distance metric. Choose from: 'greedy_modularity', 'label_propagation',"
                "'leiden', 'louvain', 'markov_clustering', 'spinglass', 'walktrap'."
            )

        # Add the sparse neighborhood matrix
        combined_neighborhoods += neighborhoods

    # Ensure maximum value in each row is set to 1
    combined_neighborhoods = _set_max_row_value_to_one_sparse(combined_neighborhoods)

    return combined_neighborhoods


def _set_max_row_value_to_one_sparse(matrix: csr_matrix) -> csr_matrix:
    """
    Set the maximum value in each row of a sparse matrix to 1.

    Args:
        matrix (csr_matrix): The input sparse matrix.

    Returns:
        csr_matrix: The modified sparse matrix where only the maximum value in each row is set to 1.
    """
    # Iterate over each row and set the maximum value to 1
    for i in range(matrix.shape[0]):
        row_data = matrix[i].data
        if len(row_data) > 0:
            row_data[:] = (row_data == max(row_data)).astype(int)

    return matrix


def process_neighborhoods(
    network: nx.Graph,
    neighborhoods: Dict[str, Any],
    impute_depth: int = 0,
    prune_threshold: float = 0.0,
) -> Dict[str, Any]:
    """
    Process neighborhoods based on the imputation and pruning settings.

    Args:
        network (nx.Graph): The network data structure used for imputing and pruning neighbors.
        neighborhoods (Dict[str, Any]): Dictionary containing 'significance_matrix', 'significant_binary_significance_matrix', and 'significant_significance_matrix'.
        impute_depth (int, optional): Depth for imputing neighbors. Defaults to 0.
        prune_threshold (float, optional): Distance threshold for pruning neighbors. Defaults to 0.0.

    Returns:
        Dict[str, Any]: Processed neighborhoods data, including the updated matrices and significance counts.
    """
    significance_matrix = neighborhoods["significance_matrix"]
    significant_binary_significance_matrix = neighborhoods["significant_binary_significance_matrix"]
    significant_significance_matrix = neighborhoods["significant_significance_matrix"]
    logger.debug(f"Imputation depth: {impute_depth}")
    if impute_depth:
        (
            significance_matrix,
            significant_binary_significance_matrix,
            significant_significance_matrix,
        ) = _impute_neighbors(
            network,
            significance_matrix,
            significant_binary_significance_matrix,
            max_depth=impute_depth,
        )

    logger.debug(f"Pruning threshold: {prune_threshold}")
    if prune_threshold:
        (
            significance_matrix,
            significant_binary_significance_matrix,
            significant_significance_matrix,
        ) = _prune_neighbors(
            network,
            significance_matrix,
            significant_binary_significance_matrix,
            distance_threshold=prune_threshold,
        )

    neighborhood_significance_counts = np.sum(significant_binary_significance_matrix, axis=0)
    node_significance_sums = np.sum(significance_matrix, axis=1)
    return {
        "significance_matrix": significance_matrix,
        "significant_binary_significance_matrix": significant_binary_significance_matrix,
        "significant_significance_matrix": significant_significance_matrix,
        "neighborhood_significance_counts": neighborhood_significance_counts,
        "node_significance_sums": node_significance_sums,
    }


def _impute_neighbors(
    network: nx.Graph,
    significance_matrix: np.ndarray,
    significant_binary_significance_matrix: np.ndarray,
    max_depth: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Impute rows with sums of zero in the significance matrix based on the closest non-zero neighbors in the network graph.

    Args:
        network (nx.Graph): The network graph with nodes having IDs matching the matrix indices.
        significance_matrix (np.ndarray): The significance matrix with rows to be imputed.
        significant_binary_significance_matrix (np.ndarray): The alpha threshold matrix to be imputed similarly.
        max_depth (int): Maximum depth of nodes to traverse for imputing values.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - np.ndarray: The imputed significance matrix.
            - np.ndarray: The imputed alpha threshold matrix.
            - np.ndarray: The significant significance matrix with non-significant entries set to zero.
    """
    # Calculate the distance threshold value based on the shortest distances
    significance_matrix, significant_binary_significance_matrix = _impute_neighbors_with_similarity(
        network, significance_matrix, significant_binary_significance_matrix, max_depth=max_depth
    )
    # Create a matrix where non-significant entries are set to zero
    significant_significance_matrix = np.where(
        significant_binary_significance_matrix == 1, significance_matrix, 0
    )

    return (
        significance_matrix,
        significant_binary_significance_matrix,
        significant_significance_matrix,
    )


def _impute_neighbors_with_similarity(
    network: nx.Graph,
    significance_matrix: np.ndarray,
    significant_binary_significance_matrix: np.ndarray,
    max_depth: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Impute non-significant nodes based on the closest significant neighbors' profiles and their similarity.

    Args:
        network (nx.Graph): The network graph with nodes having IDs matching the matrix indices.
        significance_matrix (np.ndarray): The significance matrix with rows to be imputed.
        significant_binary_significance_matrix (np.ndarray): The alpha threshold matrix to be imputed similarly.
        max_depth (int): Maximum depth of nodes to traverse for imputing values.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - The imputed significance matrix.
            - The imputed alpha threshold matrix.
    """
    depth = 1
    rows_to_impute = np.where(significant_binary_significance_matrix.sum(axis=1) == 0)[0]
    while len(rows_to_impute) and depth <= max_depth:
        # Iterate over all significant nodes
        for row_index in range(significant_binary_significance_matrix.shape[0]):
            if significant_binary_significance_matrix[row_index].sum() != 0:
                (
                    significance_matrix,
                    significant_binary_significance_matrix,
                ) = _process_node_imputation(
                    row_index,
                    network,
                    significance_matrix,
                    significant_binary_significance_matrix,
                    depth,
                )

        # Update rows to impute for the next iteration
        rows_to_impute = np.where(significant_binary_significance_matrix.sum(axis=1) == 0)[0]
        depth += 1

    return significance_matrix, significant_binary_significance_matrix


def _process_node_imputation(
    row_index: int,
    network: nx.Graph,
    significance_matrix: np.ndarray,
    significant_binary_significance_matrix: np.ndarray,
    depth: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process the imputation for a single node based on its significant neighbors.

    Args:
        row_index (int): The index of the significant node being processed.
        network (nx.Graph): The network graph with nodes having IDs matching the matrix indices.
        significance_matrix (np.ndarray): The significance matrix with rows to be imputed.
        significant_binary_significance_matrix (np.ndarray): The alpha threshold matrix to be imputed similarly.
        depth (int): Current depth for traversal.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The modified significance matrix and binary threshold matrix.
    """
    # Check neighbors at the current depth
    neighbors = nx.single_source_shortest_path_length(network, row_index, cutoff=depth)
    # Filter annotated neighbors (already significant)
    annotated_neighbors = [
        n
        for n in neighbors
        if n != row_index
        and significant_binary_significance_matrix[n].sum() != 0
        and significance_matrix[n].sum() != 0
    ]
    # Filter non-significant neighbors
    valid_neighbors = [
        n
        for n in neighbors
        if n != row_index
        and significant_binary_significance_matrix[n].sum() == 0
        and significance_matrix[n].sum() == 0
    ]
    # If there are valid non-significant neighbors
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
        # Filter valid non-significant neighbors that fall within the IQR bounds
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
                            significance_matrix[neighbor].reshape(1, -1),
                            significance_matrix[other_neighbor].reshape(1, -1),
                        )[0][0]
                        for other_neighbor in valid_neighbors_within_iqr
                        if other_neighbor != neighbor
                    )

                most_similar_neighbor = max(
                    valid_neighbors_within_iqr, key=sum_pairwise_cosine_similarities
                )
            else:
                most_similar_neighbor = valid_neighbors_within_iqr[0]

            # Impute the most similar non-significant neighbor with the significant node's data, scaled by depth
            significance_matrix[most_similar_neighbor] = significance_matrix[row_index] / np.sqrt(
                depth + 1
            )
            significant_binary_significance_matrix[most_similar_neighbor] = (
                significant_binary_significance_matrix[row_index]
            )

    return significance_matrix, significant_binary_significance_matrix


def _prune_neighbors(
    network: nx.Graph,
    significance_matrix: np.ndarray,
    significant_binary_significance_matrix: np.ndarray,
    distance_threshold: float = 0.9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Remove outliers based on their rank for edge lengths.

    Args:
        network (nx.Graph): The network graph with nodes having IDs matching the matrix indices.
        significance_matrix (np.ndarray): The significance matrix.
        significant_binary_significance_matrix (np.ndarray): The alpha threshold matrix.
        distance_threshold (float): Rank threshold (0 to 1) to determine outliers.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - np.ndarray: The updated significance matrix with outliers set to zero.
            - np.ndarray: The updated alpha threshold matrix with outliers set to zero.
            - np.ndarray: The significant significance matrix, where non-significant entries are set to zero.
    """
    # Identify indices with non-zero rows in the binary significance matrix
    non_zero_indices = np.where(significant_binary_significance_matrix.sum(axis=1) != 0)[0]
    median_distances = []
    for node in non_zero_indices:
        neighbors = [
            n
            for n in network.neighbors(node)
            if significant_binary_significance_matrix[n].sum() != 0
        ]
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
            n
            for n in network.neighbors(row_index)
            if significant_binary_significance_matrix[n].sum() != 0
        ]
        if neighbors:
            median_distance = np.median(
                [_get_euclidean_distance(row_index, n, network) for n in neighbors]
            )
            if median_distance >= distance_threshold_value:
                significance_matrix[row_index] = 0
                significant_binary_significance_matrix[row_index] = 0

    # Create a matrix where non-significant entries are set to zero
    significant_significance_matrix = np.where(
        significant_binary_significance_matrix == 1, significance_matrix, 0
    )

    return (
        significance_matrix,
        significant_binary_significance_matrix,
        significant_significance_matrix,
    )


def _get_euclidean_distance(node1: Any, node2: Any, network: nx.Graph) -> float:
    """
    Calculate the Euclidean distance between two nodes in the network.

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
    """
    Retrieve the position of a node in the network as a numpy array.

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
    """
    Calculate the distance threshold based on the given median distances and a percentile threshold.

    Args:
        median_distances (List): An array of median distances.
        distance_threshold (float): A percentile threshold (0 to 1) used to determine the distance cutoff.

    Returns:
        float: The calculated distance threshold value.

    Raises:
        ValueError: If no significant annotation is found in the median distances.
    """
    # Sort the median distances
    sorted_distances = np.sort(median_distances)
    # Compute the rank fractions for the sorted distances
    rank_percentiles = np.linspace(0, 1, len(sorted_distances))
    # Interpolating the ranks to 1000 evenly spaced percentiles
    interpolated_percentiles = np.linspace(0, 1, 1000)
    try:
        smoothed_distances = np.interp(interpolated_percentiles, rank_percentiles, sorted_distances)
    except ValueError as e:
        raise ValueError("No significant annotation found.") from e

    # Determine the index corresponding to the distance threshold
    threshold_index = int(np.ceil(distance_threshold * len(smoothed_distances))) - 1
    # Return the smoothed distance at the calculated index
    return smoothed_distances[threshold_index]
