"""
risk/neighborhoods/neighborhoods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import random
import warnings
from typing import Any, Dict, List, Tuple, Union

import networkx as nx
import numpy as np
from sklearn.exceptions import DataConversionWarning
from sklearn.metrics.pairwise import cosine_similarity

from risk.neighborhoods.community import (
    calculate_greedy_modularity_neighborhoods,
    calculate_label_propagation_neighborhoods,
    calculate_leiden_neighborhoods,
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
    distance_metric: Union[str, List, Tuple, np.ndarray] = "louvain",
    edge_rank_percentile: Union[float, List, Tuple, np.ndarray] = 1.0,
    louvain_resolution: float = 0.1,
    leiden_resolution: float = 1.0,
    random_seed: int = 888,
) -> np.ndarray:
    """Calculate the combined neighborhoods for each node based on the specified community detection algorithm(s).

    Args:
        network (nx.Graph): The network graph.
        distance_metric (str, List, Tuple, or np.ndarray, optional): The distance metric(s) to use.
        edge_rank_percentile (float, List, Tuple, or np.ndarray, optional): Shortest edge rank percentile threshold(s) for creating subgraphs.
        louvain_resolution (float, optional): Resolution parameter for the Louvain method.
        leiden_resolution (float, optional): Resolution parameter for the Leiden method.
        random_seed (int, optional): Random seed for methods requiring random initialization.

    Returns:
        np.ndarray: Summed neighborhood matrix from all selected algorithms.
    """
    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Ensure distance_metric and edge_rank_percentile are lists
    if isinstance(distance_metric, (str, np.ndarray)):
        distance_metric = [distance_metric]
    if isinstance(edge_rank_percentile, (float, int)):
        edge_rank_percentile = [edge_rank_percentile] * len(distance_metric)

    if len(distance_metric) != len(edge_rank_percentile):
        raise ValueError(
            "The number of distance metrics must match the number of edge length thresholds."
        )

    # Initialize combined neighborhood matrix
    num_nodes = network.number_of_nodes()
    combined_neighborhoods = np.zeros((num_nodes, num_nodes), dtype=int)

    # Loop through each distance metric and corresponding edge length threshold
    for metric, threshold in zip(distance_metric, edge_rank_percentile):
        # Create a subgraph based on the edge length threshold
        subgraph = _create_percentile_limited_subgraph(network, edge_rank_percentile=threshold)
        subgraph_nodes = list(subgraph.nodes)
        # Calculate neighborhoods based on the specified metric
        if metric == "greedy_modularity":
            neighborhoods = calculate_greedy_modularity_neighborhoods(subgraph)
        elif metric == "label_propagation":
            neighborhoods = calculate_label_propagation_neighborhoods(subgraph)
        elif metric == "leiden":
            neighborhoods = calculate_leiden_neighborhoods(
                subgraph, leiden_resolution, random_seed=random_seed
            )
        elif metric == "louvain":
            neighborhoods = calculate_louvain_neighborhoods(
                subgraph, louvain_resolution, random_seed=random_seed
            )
        elif metric == "markov_clustering":
            neighborhoods = calculate_markov_clustering_neighborhoods(subgraph)
        elif metric == "spinglass":
            neighborhoods = calculate_spinglass_neighborhoods(subgraph)
        elif metric == "walktrap":
            neighborhoods = calculate_walktrap_neighborhoods(subgraph)
        else:
            raise ValueError(
                "Invalid distance metric specified. Please choose from 'greedy_modularity', 'label_propagation',"
                "'leiden', 'louvain', 'markov_clustering', 'spinglass', 'walktrap'."
            )

        # Expand the neighborhood matrix to match the original network's size
        expanded_neighborhoods = expand_neighborhood_matrix(
            neighborhoods, subgraph_nodes, num_nodes
        )
        # Sum the expanded neighborhood matrices
        combined_neighborhoods += expanded_neighborhoods

    # Convert combined_neighborhoods to binary: values > 0 are set to 1
    combined_neighborhoods = (combined_neighborhoods > 0).astype(int)

    return combined_neighborhoods


def expand_neighborhood_matrix(
    subgraph_matrix: np.ndarray, subgraph_nodes: list, original_size: int
) -> np.ndarray:
    """Expand a subgraph neighborhood matrix back to the size of the original graph.

    Args:
        subgraph_matrix (np.ndarray): The neighborhood matrix for the subgraph.
        subgraph_nodes (list): List of nodes in the subgraph, corresponding to rows/columns in subgraph_matrix.
        original_size (int): The number of nodes in the original graph.

    Returns:
        np.ndarray: The expanded matrix with the original size, with subgraph values mapped correctly.
    """
    expanded_matrix = np.zeros((original_size, original_size), dtype=int)
    for i, node_i in enumerate(subgraph_nodes):
        for j, node_j in enumerate(subgraph_nodes):
            expanded_matrix[node_i, node_j] = subgraph_matrix[i, j]
    return expanded_matrix


def _create_percentile_limited_subgraph(G: nx.Graph, edge_rank_percentile: float) -> nx.Graph:
    """Create a subgraph containing all nodes and edges where the edge length is within the
    specified rank percentile of all edges in the input graph. Isolated nodes are removed.

    Args:
        G (nx.Graph): The input graph with 'length' attributes on edges.
        edge_rank_percentile (float): The rank percentile (between 0 and 1) to filter edges.

    Returns:
        nx.Graph: A subgraph with nodes and edges where the edge length is within the
        specified percentile, with isolated nodes removed, retaining all original attributes.
    """
    # Extract edges with their lengths
    edges_with_length = [(u, v, d) for u, v, d in G.edges(data=True) if "length" in d]
    if not edges_with_length:
        raise ValueError("No edge lengths found. Ensure edges have 'length' attributes.")

    # Sort edges by length in ascending order
    edges_with_length.sort(key=lambda x: x[2]["length"])
    # Calculate the cutoff based on the specified rank percentile
    cutoff_index = int(edge_rank_percentile * len(edges_with_length))
    if cutoff_index == 0:
        raise ValueError("The rank percentile is too low, resulting in no edges being included.")

    # Keep only the edges within the specified percentile
    selected_edges = edges_with_length[:cutoff_index]
    # Create a new subgraph with the selected edges, retaining all attributes
    subgraph = nx.Graph()
    subgraph.add_edges_from((u, v, d) for u, v, d in selected_edges)
    # Copy over all node attributes from the original graph
    subgraph.add_nodes_from((node, G.nodes[node]) for node in subgraph.nodes())

    # Remove isolated nodes (if any)
    isolated_nodes = [node for node, degree in subgraph.degree() if degree == 0]
    subgraph.remove_nodes_from(isolated_nodes)
    # Check if the resulting subgraph has no edges
    if subgraph.number_of_edges() == 0:
        raise ValueError("The resulting subgraph has no edges. Adjust the rank percentile.")

    return subgraph


def _set_max_row_value_to_one(matrix: np.ndarray) -> np.ndarray:
    """For each row in the input matrix, set the maximum value(s) to 1 and all other values to 0. This is particularly
    useful for neighborhood matrices that have undergone multiple neighborhood detection algorithms, where the
    maximum value in each row represents the most significant relationship per node in the combined neighborhoods.

    Args:
        matrix (np.ndarray): A 2D numpy array representing the neighborhood matrix.

    Returns:
        np.ndarray: The modified matrix where only the maximum value(s) in each row is set to 1, and others are set to 0.
    """
    # Find the maximum value in each row (column-wise max operation)
    max_values = np.max(matrix, axis=1, keepdims=True)
    # Create a boolean mask where elements are True if they are the max value in their row
    max_mask = matrix == max_values
    # Set all elements to 0, and then set the maximum value positions to 1
    matrix[:] = 0  # Set everything to 0
    matrix[max_mask] = 1  # Set only the max values to 1
    return matrix


def process_neighborhoods(
    network: nx.Graph,
    neighborhoods: Dict[str, Any],
    impute_depth: int = 0,
    prune_threshold: float = 0.0,
) -> Dict[str, Any]:
    """Process neighborhoods based on the imputation and pruning settings.

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
    """Impute rows with sums of zero in the significance matrix based on the closest non-zero neighbors in the network graph.

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
    """Impute non-significant nodes based on the closest significant neighbors' profiles and their similarity.

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
                significance_matrix, significant_binary_significance_matrix = (
                    _process_node_imputation(
                        row_index,
                        network,
                        significance_matrix,
                        significant_binary_significance_matrix,
                        depth,
                    )
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
    """Process the imputation for a single node based on its significant neighbors.

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
    """Remove outliers based on their rank for edge lengths.

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
        median_distances (List): An array of median distances.
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
