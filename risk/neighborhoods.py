"""
risk/network/neighborhoods
~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import warnings
from contextlib import suppress
from tqdm import tqdm

import community as community_louvain
import networkx as nx
import numpy as np
import markov_clustering as mc
import pandas as pd
from networkx.algorithms.community import asyn_lpa_communities
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AffinityPropagation
from sklearn.exceptions import DataConversionWarning
from sklearn.metrics import silhouette_score

from risk.annotations import get_description
from risk.constants import GROUP_LINKAGE_METHODS, GROUP_DISTANCE_METRICS

# Suppress DataConversionWarning
warnings.filterwarnings(action="ignore", category=DataConversionWarning)


def get_network_neighborhoods(
    network,
    distance_metric,
    neighborhood_diameter,
    compute_sphere=False,
    louvain_resolution=1.0,
    random_walk_length=3,
    random_walk_num=250,
    random_seed=888,
):
    """Calculate the neighborhoods for each node in the network based on the specified distance metric.

    Args:
        network (nx.Graph): The network graph.
        distance_metric (str): The distance metric to use ('euclidean', 'dijkstra', 'louvain', 'affinity_propagation',
        'label_propagation').
        neighborhood_diameter (float): The neighborhood_diameter of the neighborhoods.
        compute_sphere (bool, optional): Whether to compute the neighborhoods considering a spherical surface. Defaults to False.
        louvain_resolution (float, optional): Resolution parameter for the Louvain method. Defaults to 1.0.

    Returns:
        np.ndarray: Neighborhood matrix.
    """
    radius = (neighborhood_diameter / 2) * (4 if compute_sphere else 1)

    if distance_metric == "dijkstra":
        return _calculate_dijkstra_neighborhoods(network, radius)
    if distance_metric == "louvain":
        return _calculate_louvain_neighborhoods(
            network, louvain_resolution, random_seed=random_seed
        )
    if distance_metric == "affinity_propagation":
        return _calculate_affinity_propagation_neighborhoods(network, random_seed=random_seed)
    if distance_metric == "label_propagation":
        return _calculate_label_propagation_neighborhoods(network)
    if distance_metric == "random_walk":
        return _calculate_random_walk_neighborhoods(
            network, walk_length=random_walk_length, num_walks=random_walk_num
        )
    if distance_metric == "markov_clustering":
        return _calculate_markov_clustering_neighborhoods(network)
    if distance_metric == "walktrap":
        return _calculate_walktrap_neighborhoods(network)
    if distance_metric == "spinglass":
        return _calculate_spinglass_neighborhoods(network)
    if distance_metric == "chinese_whispers":
        return _calculate_chinese_whispers_neighborhoods(network)

    raise ValueError(
        "Incorrect distance metric specified. Please choose from 'dijkstra', 'louvain', 'affinity_propagation'."
        "label_propagation', 'random_walk', 'markov_clustering', 'walktrap', spinglass', 'chinese_whispers'."
    )


def _calculate_dijkstra_neighborhoods(network, radius):
    """Helper function to calculate neighborhoods using Dijkstra's distances.

    Args:
        network (nx.Graph): The network graph.
        radius (float): The radius for neighborhood calculation.

    Returns:
        np.ndarray: Neighborhood matrix based on Dijkstra's distances.
    """
    # Compute Dijkstra's distance within the specified radius
    all_dijkstra_paths = dict(
        nx.all_pairs_dijkstra_path_length(network, weight="length", cutoff=radius)
    )
    neighborhoods = np.zeros((network.number_of_nodes(), network.number_of_nodes()), dtype=int)

    # Populate neighborhoods matrix based on Dijkstra's
    for source, targets in all_dijkstra_paths.items():
        for target, length in targets.items():
            neighborhoods[source, target] = (
                1 if np.isnan(length) or length == 0 else np.sqrt(1 / length)
            )

    return neighborhoods


def _calculate_louvain_neighborhoods(network, resolution, random_seed=888):
    """Helper function to calculate neighborhoods using the Louvain method.

    Args:
        network (nx.Graph): The network graph.
        resolution (float): Resolution parameter for the Louvain method.

    Returns:
        np.ndarray: Neighborhood matrix based on the Louvain method.
    """
    # Apply Louvain method to partition the network
    partition = community_louvain.best_partition(
        network, resolution=resolution, random_state=random_seed
    )
    neighborhoods = np.zeros((network.number_of_nodes(), network.number_of_nodes()), dtype=int)

    # Assign neighborhoods based on community partitions
    for node_i, community_i in partition.items():
        for node_j, community_j in partition.items():
            if community_i == community_j:
                neighborhoods[node_i, node_j] = 1

    return neighborhoods


def _calculate_affinity_propagation_neighborhoods(network, random_seed=888):
    """Helper function to calculate neighborhoods using Affinity Propagation.

    Args:
        network (nx.Graph): The network graph.

    Returns:
        np.ndarray: Neighborhood matrix based on Affinity Propagation clustering.
    """
    # Compute Dijkstra's to form a distance matrix
    distance_matrix = nx.floyd_warshall_numpy(network)

    # Convert distances to similarities
    similarity_matrix = -distance_matrix

    # Apply Affinity Propagation clustering
    clustering = AffinityPropagation(affinity="precomputed", random_state=random_seed)
    clustering.fit(similarity_matrix)
    labels = clustering.labels_
    neighborhoods = np.zeros((network.number_of_nodes(), network.number_of_nodes()), dtype=int)

    # Assign neighborhoods based on clustering results
    for i, label_i in enumerate(labels):
        for j, label_j in enumerate(labels):
            if label_i == label_j:
                neighborhoods[i, j] = 1

    return neighborhoods


def _calculate_label_propagation_neighborhoods(network):
    """Apply Label Propagation to the network to detect communities.

    Args:
        network (nx.Graph): The network graph.

    Returns:
        np.ndarray: Neighborhood matrix based on Label Propagation.
    """
    # Apply Label Propagation
    communities = nx.algorithms.community.label_propagation.label_propagation_communities(network)

    # Create a mapping from node to community
    community_dict = {}
    for community_id, community in enumerate(communities):
        for node in community:
            community_dict[node] = community_id

    # Create a neighborhood matrix
    num_nodes = network.number_of_nodes()
    neighborhoods = np.zeros((num_nodes, num_nodes), dtype=int)

    # Assign neighborhoods based on community labels
    for node_i, community_i in community_dict.items():
        for node_j, community_j in community_dict.items():
            if community_i == community_j:
                neighborhoods[node_i, node_j] = 1

    return neighborhoods


def _calculate_random_walk_neighborhoods(network, walk_length=3, num_walks=100):
    """Apply Random Walk Clustering to the network to detect communities.

    Args:
        network (nx.Graph): The network graph.
        walk_length (int): The length of each random walk.
        num_walks (int): The number of random walks to perform from each node.

    Returns:
        np.ndarray: Neighborhood matrix based on Random Walk Clustering.
    """

    def random_walk(graph, start_node, walk_length):
        """Perform a random walk of specified length starting from the start_node."""
        walk = [start_node]
        for _ in range(walk_length - 1):
            neighbors = list(graph.neighbors(walk[-1]))
            if neighbors:
                walk.append(np.random.choice(neighbors))
            else:
                break
        return walk

    # Perform random walks from each node
    walks = []
    for node in network.nodes:
        for _ in range(num_walks):
            walks.append(random_walk(network, node, walk_length))

    # Count co-occurrences in walks to create a similarity matrix
    co_occurrences = np.zeros((network.number_of_nodes(), network.number_of_nodes()), dtype=int)
    for walk in walks:
        for i in range(len(walk)):
            for j in range(i + 1, len(walk)):
                co_occurrences[walk[i], walk[j]] += 1
                co_occurrences[walk[j], walk[i]] += 1

    # Normalize co-occurrences to get a neighborhood matrix
    max_co_occurrence = np.max(co_occurrences)
    neighborhoods = (co_occurrences / max_co_occurrence) > 0.1  # Threshold can be adjusted

    return neighborhoods.astype(int)


def _calculate_markov_clustering_neighborhoods(network):
    """Apply Markov Clustering (MCL) to the network.

    Args:
        network (nx.Graph): The network graph.

    Returns:
        np.ndarray: Neighborhood matrix based on Markov Clustering.
    """
    # Convert the graph to an adjacency matrix
    adjacency_matrix = nx.to_numpy_array(network)
    # Run Markov Clustering
    result = mc.run_mcl(adjacency_matrix)  # run MCL with default parameters
    # Get clusters
    clusters = mc.get_clusters(result)
    # Create a community label for each node
    community_dict = {}
    for community_id, community in enumerate(clusters):
        for node in community:
            community_dict[node] = community_id

    # Create a neighborhood matrix
    num_nodes = network.number_of_nodes()
    neighborhoods = np.zeros((num_nodes, num_nodes), dtype=int)

    # Assign neighborhoods based on community labels
    for node_i, community_i in community_dict.items():
        for node_j, community_j in community_dict.items():
            if community_i == community_j:
                neighborhoods[node_i, node_j] = 1

    return neighborhoods


def _calculate_walktrap_neighborhoods(network):
    """Apply Walktrap Community Detection to the network.

    Args:
        network (nx.Graph): The network graph.

    Returns:
        np.ndarray: Neighborhood matrix based on Walktrap communities.
    """
    # Use the asynchronous label propagation algorithm as a proxy for Walktrap
    communities = asyn_lpa_communities(network)

    # Create a community label for each node
    community_dict = {}
    for community_id, community in enumerate(communities):
        for node in community:
            community_dict[node] = community_id

    # Create a neighborhood matrix
    num_nodes = network.number_of_nodes()
    neighborhoods = np.zeros((num_nodes, num_nodes), dtype=int)

    # Assign neighborhoods based on community labels
    for node_i, community_i in community_dict.items():
        for node_j, community_j in community_dict.items():
            if community_i == community_j:
                neighborhoods[node_i, node_j] = 1

    return neighborhoods


def _calculate_spinglass_neighborhoods(network):
    """Apply Spin Glass Community Detection to the network.

    Args:
        network (nx.Graph): The network graph.

    Returns:
        np.ndarray: Neighborhood matrix based on Spin Glass communities.
    """
    # Use the asynchronous label propagation algorithm as a proxy for Spin Glass
    communities = asyn_lpa_communities(network)

    # Create a community label for each node
    community_dict = {}
    for community_id, community in enumerate(communities):
        for node in community:
            community_dict[node] = community_id

    # Create a neighborhood matrix
    num_nodes = network.number_of_nodes()
    neighborhoods = np.zeros((num_nodes, num_nodes), dtype=int)

    # Assign neighborhoods based on community labels
    for node_i, community_i in community_dict.items():
        for node_j, community_j in community_dict.items():
            if community_i == community_j:
                neighborhoods[node_i, node_j] = 1

    return neighborhoods


def _calculate_chinese_whispers_neighborhoods(network):
    """Apply Chinese Whispers Community Detection to the network.

    Args:
        network (nx.Graph): The network graph.

    Returns:
        np.ndarray: Neighborhood matrix based on Chinese Whispers communities.
    """
    # Initialize node labels
    for node in network.nodes:
        network.nodes[node]["label"] = node

    for _ in range(20):  # Number of iterations
        nodes = list(network.nodes)
        np.random.shuffle(nodes)
        for node in nodes:
            label_freq = {}
            for neighbor in network.neighbors(node):
                label = network.nodes[neighbor]["label"]
                if label in label_freq:
                    label_freq[label] += 1
                else:
                    label_freq[label] = 1
            most_frequent_label = max(label_freq, key=label_freq.get)
            network.nodes[node]["label"] = most_frequent_label

    # Create a community label for each node
    community_dict = {}
    for node in network.nodes:
        community_dict[node] = network.nodes[node]["label"]

    # Create a neighborhood matrix
    num_nodes = network.number_of_nodes()
    neighborhoods = np.zeros((num_nodes, num_nodes), dtype=int)

    # Assign neighborhoods based on community labels
    for node_i, community_i in community_dict.items():
        for node_j, community_j in community_dict.items():
            if community_i == community_j:
                neighborhoods[node_i, node_j] = 1

    return neighborhoods


def process_neighborhoods(
    network,
    neighborhoods,
    impute_depth=1,
    prune_threshold=0.0,
):
    """Process neighborhoods based on the imputation and pruning settings.

    Args:
        network: The network data structure used for imputing and pruning neighbors.
        enrichment_matrix (numpy.ndarray): Enrichment matrix data.
        binary_enrichment_matrix (numpy.ndarray): Binary enrichment matrix data.
        impute_threshold (float): Distance threshold for imputing neighbors.
        impute_depth (int): Depth for imputing neighbors.
        prune_threshold (float): Distance threshold for pruning neighbors.

    Returns:
        dict: Processed neighborhoods data.
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


def _impute_neighbors(network, enrichment_matrix, binary_enrichment_matrix, max_depth=3):
    """
    Impute rows with sums of zero in the enrichment matrix based on the closest non-zero neighbors in the network graph.

    Args:
        network (NetworkX graph): The network graph with nodes having IDs matching the matrix indices.
        enrichment_matrix (np.ndarray): The enrichment matrix with rows to be imputed.
        binary_enrichment_matrix (np.ndarray): The alpha threshold matrix to be imputed similarly.
        max_depth (int): Maximum depth of nodes to traverse for imputing values.

    Returns:
        tuple: The imputed enrichment matrix and the imputed alpha threshold matrix.
    """
    # Calculate shortest instances for each node and determine the distance threshold
    shortest_distances = []
    for node in network.nodes():
        neighbors = [n for n in network.neighbors(node) if binary_enrichment_matrix[n].sum() != 0]
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
                enrichment_matrix[row_index] = enrichment_matrix[closest_neighbor]
                binary_enrichment_matrix[row_index] = binary_enrichment_matrix[
                    closest_neighbor
                ] / np.sqrt(depth + 1)
            else:
                next_rows_to_impute.append(row_index)

        rows_to_impute = next_rows_to_impute
        depth += 1

    significant_enrichment_matrix = np.where(binary_enrichment_matrix == 1, enrichment_matrix, 0)

    return enrichment_matrix, binary_enrichment_matrix, significant_enrichment_matrix


def _prune_neighbors(network, enrichment_matrix, binary_enrichment_matrix, distance_threshold=0.9):
    """
    Remove outliers based on their rank for edge lengths.

    Args:
        network (NetworkX graph): The network graph with nodes having IDs matching the matrix indices.
        enrichment_matrix (np.ndarray): The enrichment matrix.
        binary_enrichment_matrix (np.ndarray): The alpha threshold matrix.
        distance_threshold (float): Rank threshold (0 to 1) to determine outliers.

    Returns:
        tuple: The updated enrichment matrix and alpha threshold matrix with outliers set to zero.
    """
    """
    Remove outliers based on their rank for edge lengths.

    Args:
        network (NetworkX graph): The network graph with nodes having IDs matching the matrix indices.
        enrichment_matrix (np.ndarray): The enrichment matrix.
        binary_enrichment_matrix (np.ndarray): The alpha threshold matrix.
        distance_threshold (float): Rank threshold (0 to 1) to determine outliers.

    Returns:
        tuple: The updated enrichment matrix and alpha threshold matrix with outliers set to zero.
    """
    non_zero_indices = np.where(binary_enrichment_matrix.sum(axis=1) != 0)[0]

    average_distances = []
    for node in non_zero_indices:
        neighbors = [n for n in network.neighbors(node) if binary_enrichment_matrix[n].sum() != 0]
        if neighbors:
            average_distance = np.mean(
                [_get_euclidean_distance(node, n, network) for n in neighbors]
            )
            average_distances.append(average_distance)

    distance_threshold_value = _calculate_threshold(average_distances, 1 - distance_threshold)

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

    significant_enrichment_matrix = np.where(binary_enrichment_matrix == 1, enrichment_matrix, 0)

    return enrichment_matrix, binary_enrichment_matrix, significant_enrichment_matrix


def _get_euclidean_distance(node1, node2, network):
    pos1 = _get_node_position(network, node1)
    pos2 = _get_node_position(network, node2)
    return np.linalg.norm(pos1 - pos2)


def _get_node_position(network, node):
    return np.array(
        [
            network.nodes[node].get(coord, 0)
            for coord in ["x", "y", "z"]
            if coord in network.nodes[node]
        ]
    )


def _calculate_threshold(average_distances, distance_threshold):
    sorted_distances = np.sort(average_distances)
    rank_percentiles = np.linspace(0, 1, len(sorted_distances))
    # Interpolating the ranks to 1000 evenly spaced percentiles
    interpolated_percentiles = np.linspace(0, 1, 1000)
    smoothed_distances = np.interp(interpolated_percentiles, rank_percentiles, sorted_distances)
    threshold_index = int(np.ceil(distance_threshold * len(smoothed_distances))) - 1
    return smoothed_distances[threshold_index]


def define_domains(
    top_annotations,
    significant_neighborhoods_enrichment,
    linkage_criterion,
    linkage_method,
    linkage_metric,
):
    """Define domains and assign nodes to these domains based on their enrichment scores and clustering.

    Args:
        neighborhoods (np.ndarray): The neighborhood enrichment matrix.
        top_annotations (pd.DataFrame): DataFrame of top annotations data for the network nodes.
        significant_enrichment (np.ndarray): The binary enrichment matrix below alpha.
        linkage_criterion (str): The clustering criterion for defining groups.
        linkage_method (str): The linkage method for clustering.
        linkage_metric (str): The linkage metric for clustering.

    Returns:
        pd.DataFrame: DataFrame with primary domain for each node.
    """
    # Perform hierarchical clustering on the binary enrichment matrix
    m = significant_neighborhoods_enrichment[:, top_annotations["top attributes"]].T
    best_linkage, best_metric, best_threshold = _optimize_silhouette_across_linkage_and_metrics(
        m, linkage_criterion, linkage_method, linkage_metric
    )
    try:
        Z = linkage(m, method=best_linkage, metric=best_metric)
    except ValueError as e:
        raise ValueError("No significant annotations found.") from e

    print(
        f"Linkage criterion: '{linkage_criterion}'\nLinkage method: '{best_linkage}'\nLinkage metric: '{best_metric}'"
    )
    print(f"Optimal linkage threshold: {round(best_threshold, 3)}")

    max_d_optimal = np.max(Z[:, 2]) * best_threshold
    domains = fcluster(Z, max_d_optimal, criterion=linkage_criterion)
    # Assign domains to annotations matrix
    top_annotations["domain"] = 0
    top_annotations.loc[top_annotations["top attributes"], "domain"] = domains

    # Create DataFrames to store domain information
    node_to_enrichment = pd.DataFrame(
        data=significant_neighborhoods_enrichment,
        columns=[top_annotations.index.values, top_annotations["domain"]],
    )
    node_to_domain = node_to_enrichment.groupby(level="domain", axis=1).sum()

    t_max = node_to_domain.loc[:, 1:].max(axis=1)
    t_idxmax = node_to_domain.loc[:, 1:].idxmax(axis=1)
    t_idxmax[t_max == 0] = 0

    # Assign primary domain and NES
    node_to_domain["primary domain"] = t_idxmax

    return node_to_domain


def trim_domains_and_top_annotations(
    domains, top_annotations, min_cluster_size=5, max_cluster_size=1000
):
    """Trim domains and top annotations that do not meet size criteria and find outliers.

    Args:
        domains (pd.DataFrame): DataFrame of domain data for the network nodes.
        top_annotations (pd.DataFrame): DataFrame of top annotations data for the network nodes.
        min_cluster_size (int): Minimum size of a cluster to be retained.
        max_cluster_size (int): Maximum size of a cluster to be retained.

    Returns:
        tuple: Trimmed annotations, domains, and a DataFrame with domain labels.
    """
    # Identify domains to remove based on size criteria
    domain_counts = domains["primary domain"].value_counts()
    to_remove = set(
        domain_counts[(domain_counts < min_cluster_size) | (domain_counts > max_cluster_size)].index
    )
    # Add invalid domain IDs
    invalid_domain_id = 888888
    invalid_domain_ids = {0, invalid_domain_id}
    # Mark domains to be removed
    top_annotations["domain"].replace(to_remove, invalid_domain_id, inplace=True)
    domains.loc[domains["primary domain"].isin(to_remove), ["primary domain"]] = invalid_domain_id
    # Normalize "num enriched neighborhoods" by percentile for each domain and scale to 0-10
    top_annotations["normalized_value"] = top_annotations.groupby("domain")[
        "neighborhood enrichment sums"
    ].transform(lambda x: (x.rank(pct=True) * 10).apply(np.ceil).astype(int))
    # Multiply 'name' column by normalized values
    top_annotations["words"] = top_annotations.apply(
        lambda row: " ".join([row["words"]] * row["normalized_value"]), axis=1
    )

    # Generate domain labels
    domain_labels = top_annotations.groupby("domain")["words"].apply(get_description).reset_index()
    trimmed_domains_matrix = domain_labels.rename(
        columns={"domain": "id", "words": "label"}
    ).set_index("id")

    # Remove invalid domains
    valid_annotations = top_annotations[~top_annotations["domain"].isin(invalid_domain_ids)].drop(
        columns=["normalized_value"]
    )
    valid_domains = domains[~domains["primary domain"].isin(invalid_domain_ids)]
    valid_trimmed_domains_matrix = trimmed_domains_matrix[
        ~trimmed_domains_matrix.index.isin(invalid_domain_ids)
    ]

    return valid_annotations, valid_domains, valid_trimmed_domains_matrix


def _find_best_silhouette_score(
    Z, m, linkage_metric, linkage_criterion, lower_bound=0.001, upper_bound=1.0, resolution=0.001
):
    """Find the best silhouette score using binary search.

    Args:
        Z (np.ndarray): Linkage matrix.
        m (np.ndarray): Data matrix.
        linkage_metric (str): Linkage metric for silhouette score calculation.
        linkage_criterion (str): Clustering criterion.
        lower_bound (float, optional): Lower bound for search. Defaults to 0.001.
        upper_bound (float, optional): Upper bound for search. Defaults to 1.0.
        resolution (float, optional): Desired resolution for the best threshold. Defaults to 0.01.

    Returns:
        tuple: Best threshold and best silhouette score.
    """
    best_score = -np.inf
    best_threshold = None

    # Test lower bound
    max_d_lower = np.max(Z[:, 2]) * lower_bound
    clusters_lower = fcluster(Z, max_d_lower, criterion=linkage_criterion)
    try:
        score_lower = silhouette_score(m, clusters_lower, metric=linkage_metric)
    except ValueError:
        score_lower = -np.inf

    # Test upper bound
    max_d_upper = np.max(Z[:, 2]) * upper_bound
    clusters_upper = fcluster(Z, max_d_upper, criterion=linkage_criterion)
    try:
        score_upper = silhouette_score(m, clusters_upper, metric=linkage_metric)
    except ValueError:
        score_upper = -np.inf

    # Determine initial bounds for binary search
    if score_lower > score_upper:
        best_score = score_lower
        best_threshold = lower_bound
        upper_bound = (lower_bound + upper_bound) / 2
    else:
        best_score = score_upper
        best_threshold = upper_bound
        lower_bound = (lower_bound + upper_bound) / 2

    # Binary search loop
    while upper_bound - lower_bound > resolution:
        mid_threshold = (upper_bound + lower_bound) / 2

        max_d_mid = np.max(Z[:, 2]) * mid_threshold
        clusters_mid = fcluster(Z, max_d_mid, criterion=linkage_criterion)
        try:
            score_mid = silhouette_score(m, clusters_mid, metric=linkage_metric)
        except ValueError:
            score_mid = -np.inf
        # Update best score and threshold if mid-point is better
        if score_mid > best_score:
            best_score = score_mid
            best_threshold = mid_threshold

        # Adjust bounds based on the scores
        if score_lower > score_upper:
            upper_bound = mid_threshold
        else:
            lower_bound = mid_threshold

    return best_threshold, best_score


def _optimize_silhouette_across_linkage_and_metrics(
    m, linkage_criterion, linkage_method, linkage_metric
):
    """Optimize silhouette score across different linkage methods and distance metrics.

    Args:
        m (np.ndarray): Data matrix.
        linkage_criterion (str): Clustering criterion.
        linkage_method (str): Linkage method for clustering.
        linkage_metric (str): Linkage metric for clustering.

    Returns:
        tuple: Best linkage method, linkage metric, and threshold.
    """
    best_overall_method = linkage_method
    best_overall_metric = linkage_metric
    best_overall_score = -np.inf
    best_overall_threshold = 1

    linkage_methods = GROUP_LINKAGE_METHODS if linkage_method == "auto" else [linkage_method]
    linkage_metrics = GROUP_DISTANCE_METRICS if linkage_metric == "auto" else [linkage_metric]
    total_combinations = len(linkage_methods) * len(linkage_metrics)

    # Evaluating optimal linkage method and metric
    for method in tqdm(
        linkage_methods,
        desc="Evaluating optimal linkage method and metric",
        total=total_combinations,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    ):
        for metric in linkage_metrics:
            with suppress(Exception):
                Z = linkage(m, method=method, metric=metric)
                threshold, score = _find_best_silhouette_score(Z, m, metric, linkage_criterion)
                if score > best_overall_score:
                    best_overall_score = score
                    best_overall_threshold = threshold
                    best_overall_method = method
                    best_overall_metric = metric

    return best_overall_method, best_overall_metric, best_overall_threshold
