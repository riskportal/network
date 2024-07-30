"""
risk/network/neighborhoods
~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import warnings

from collections import Counter
import community as community_louvain
import networkx as nx
import numpy as np
import pandas as pd
from rich import print
from rich.progress import Progress
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
    network, distance_metric, neighborhood_diameter, compute_sphere=False, louvain_resolution=1.0
):
    """Calculate the neighborhoods for each node in the network based on the specified distance metric.

    Args:
        network (nx.Graph): The network graph.
        distance_metric (str): The distance metric to use ('euclidean', 'dijkstra', 'louvain', 'affinity_propagation').
        neighborhood_diameter (float): The neighborhood_diameter of the neighborhoods.
        compute_sphere (bool, optional): Whether to compute the neighborhoods considering a spherical surface. Defaults to False.
        louvain_resolution (float, optional): Resolution parameter for the Louvain method. Defaults to 1.0.

    Returns:
        np.ndarray: Neighborhood matrix.
    """
    radius = (neighborhood_diameter / 2) * (4 if compute_sphere else 1)

    if distance_metric == "euclidean":
        return _calculate_euclidean_neighborhoods(network, radius)
    if distance_metric == "dijkstra":
        return _calculate_dijkstra_neighborhoods(network, radius)
    if distance_metric == "louvain":
        return _calculate_louvain_neighborhoods(network, louvain_resolution)
    if distance_metric == "affinity_propagation":
        return _calculate_affinity_propagation_neighborhoods(network)

    raise ValueError(
        "Incorrect distance metric specified. Please choose from 'euclidean', 'dijkstra', 'louvain', or 'affinity_propagation'."
    )


def _calculate_euclidean_neighborhoods(network, radius):
    """Helper function to calculate neighborhoods using Euclidean distances.

    Args:
        network (nx.Graph): The network graph.
        radius (float): The radius for neighborhood calculation.

    Returns:
        np.ndarray: Neighborhood matrix based on Euclidean distances.
    """
    # Extract x and y coordinates from the network nodes
    x = np.array(list(dict(network.nodes.data("x")).values()))
    y = np.array(list(dict(network.nodes.data("y")).values()))
    node_coordinates = np.stack((x, y), axis=1)

    # Calculate Euclidean distances between all node pairs
    node_distances = squareform(pdist(node_coordinates, "euclidean"))

    # Determine neighborhoods based on the radius
    neighborhoods = np.zeros_like(node_distances, dtype=int)
    neighborhoods[node_distances < radius] = 1
    return neighborhoods


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


def _calculate_louvain_neighborhoods(network, resolution):
    """Helper function to calculate neighborhoods using the Louvain method.

    Args:
        network (nx.Graph): The network graph.
        resolution (float): Resolution parameter for the Louvain method.

    Returns:
        np.ndarray: Neighborhood matrix based on the Louvain method.
    """
    # Apply Louvain method to partition the network
    partition = community_louvain.best_partition(network, resolution=resolution)
    neighborhoods = np.zeros((network.number_of_nodes(), network.number_of_nodes()), dtype=int)

    # Assign neighborhoods based on community partitions
    for node_i, community_i in partition.items():
        for node_j, community_j in partition.items():
            if community_i == community_j:
                neighborhoods[node_i, node_j] = 1

    return neighborhoods


def _calculate_affinity_propagation_neighborhoods(network):
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
    clustering = AffinityPropagation(affinity="precomputed", random_state=5)
    clustering.fit(similarity_matrix)
    labels = clustering.labels_
    neighborhoods = np.zeros((network.number_of_nodes(), network.number_of_nodes()), dtype=int)

    # Assign neighborhoods based on clustering results
    for i, label_i in enumerate(labels):
        for j, label_j in enumerate(labels):
            if label_i == label_j:
                neighborhoods[i, j] = 1

    return neighborhoods


def define_domains(
    top_annotations,
    neighborhoods_enrichment,
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
        pd.DataFrame: DataFrame with primary domain and primary NES for each node.
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
        f"[cyan]Using [blue]clustering criterion[/blue] [yellow]'{linkage_criterion}'[/yellow] with [blue]linkage method[/blue] [yellow]'{best_linkage}'[/yellow] and [blue]linkage metric[/blue] [yellow]'{best_metric}'[/yellow]...[/cyan]"
    )
    print(f"[yellow]Optimal linkage threshold: [red]{round(best_threshold, 3)}[/red][/yellow]")

    max_d_optimal = np.max(Z[:, 2]) * best_threshold
    domains = fcluster(Z, max_d_optimal, criterion=linkage_criterion)
    # Assign domains to annotations matrix
    top_annotations["domain"] = 0
    top_annotations.loc[top_annotations["top attributes"], "domain"] = domains

    # Create DataFrames to store domain information
    node2nes = pd.DataFrame(
        data=neighborhoods_enrichment,
        columns=[top_annotations.index.values, top_annotations["domain"]],
    )
    node2nes_binary = pd.DataFrame(
        data=significant_neighborhoods_enrichment,
        columns=[top_annotations.index.values, top_annotations["domain"]],
    )
    node2domain = node2nes_binary.groupby(level="domain", axis=1).sum()

    t_max = node2domain.loc[:, 1:].max(axis=1)
    t_idxmax = node2domain.loc[:, 1:].idxmax(axis=1)
    t_idxmax[t_max == 0] = 0

    # Assign primary domain and NES
    node2domain["primary domain"] = t_idxmax
    o = node2nes.groupby(level="domain", axis=1).max()
    i = pd.Series(t_idxmax)
    node2domain["primary nes"] = o.values[i.index, i.values - 1]

    return node2domain


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
    to_remove.update(_find_outlier_domains(Counter(domains["primary domain"])))
    # Add invalid domain IDs
    invalid_domain_id = 888888
    invalid_domain_ids = {0, invalid_domain_id}
    to_remove.update(invalid_domain_ids)
    # Mark domains to be removed
    top_annotations["domain"].replace(to_remove, invalid_domain_id, inplace=True)
    domains.loc[
        domains["primary domain"].isin(to_remove), ["primary domain", "primary nes"]
    ] = invalid_domain_id
    # Normalize "num enriched neighborhoods" by percentile for each domain and scale to 0-10
    top_annotations["normalized_value"] = top_annotations.groupby("domain")[
        "num enriched neighborhoods"
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
    valid_annotations = top_annotations[top_annotations["domain"] != invalid_domain_id].drop(
        columns=["normalized_value"]
    )
    valid_domains = domains[domains["primary domain"] != invalid_domain_id]
    valid_trimmed_domains_matrix = trimmed_domains_matrix[
        ~trimmed_domains_matrix.index.isin(invalid_domain_ids)
    ]

    return valid_annotations, valid_domains, valid_trimmed_domains_matrix


def _find_outlier_domains(data_dict, z_score_threshold=3):
    """Identify outlier domains based on z-score.

    Args:
        data_dict (dict): Dictionary with domain counts.
        z_score_threshold (float, optional): Z-score threshold for identifying outliers. Defaults to 3.

    Returns:
        list: List of outlier domain keys.
    """
    values = np.array(list(data_dict.values()))
    mean = np.mean(values)
    std_dev = np.std(values)
    outlier_keys = [
        key for key, value in data_dict.items() if abs((value - mean) / std_dev) > z_score_threshold
    ]
    return outlier_keys


def _binary_search_silhouette_metric(
    Z, m, metric, linkage_criterion, lower_bound=0.0, upper_bound=1.0, tolerance=0.01
):
    """Perform binary search for the best silhouette score with a given metric and linkage method.

    Args:
        Z (np.ndarray): Linkage matrix.
        m (np.ndarray): Data matrix.
        metric (str): Distance metric for silhouette score calculation.
        linkage_criterion (str): Clustering criterion.
        lower_bound (float, optional): Lower bound for search. Defaults to 0.0.
        upper_bound (float, optional): Upper bound for search. Defaults to 1.0.
        tolerance (float, optional): Tolerance for search. Defaults to 0.01.

    Returns:
        tuple: Best threshold and best silhouette score.
    """
    best_threshold = lower_bound
    best_score = -np.inf

    while upper_bound - lower_bound > tolerance:
        mid = (lower_bound + upper_bound) / 2
        max_d_mid = np.max(Z[:, 2]) * mid
        clusters_mid = fcluster(Z, max_d_mid, criterion=linkage_criterion)

        max_d_high = np.max(Z[:, 2]) * (mid + tolerance)
        clusters_high = fcluster(Z, max_d_high, criterion=linkage_criterion)

        try:
            score_mid = silhouette_score(m, clusters_mid, metric=metric)
        except ValueError:
            score_mid = -np.inf

        try:
            score_high = silhouette_score(m, clusters_high, metric=metric)
        except ValueError:
            score_high = -np.inf

        if score_mid > best_score:
            best_score = score_mid
            best_threshold = mid

        if score_high > best_score:
            best_score = score_high
            best_threshold = mid + tolerance

        if score_high > score_mid:
            lower_bound = mid
        else:
            upper_bound = mid

    return best_threshold, best_score


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
    best_overall_score = -np.inf
    best_overall_metric = "cosine"
    best_overall_threshold = 1
    best_overall_linkage = "average"

    linkage_methods = GROUP_LINKAGE_METHODS if linkage_method is None else [linkage_method]
    linkage_metrics = GROUP_DISTANCE_METRICS if linkage_metric is None else [linkage_metric]
    total = len(linkage_methods) * len(linkage_metrics)

    with Progress() as progress:
        task = progress.add_task(
            "[cyan]Evaluating [yellow]optimal[/yellow] [blue]linkage method[/blue] and [blue]linkage metric[/blue]...",
            total=total,
        )
        for method in linkage_methods:
            for metric in linkage_metrics:
                try:
                    Z = linkage(m, method=method, metric=metric)
                    if len(linkage_methods) == 1 or len(linkage_metrics) == 1:
                        threshold, score = _find_best_silhouette_score(
                            Z, m, metric, linkage_criterion
                        )
                    else:
                        threshold, score = _binary_search_silhouette_metric(
                            Z, m, metric, linkage_criterion
                        )
                    if score > best_overall_score:
                        best_overall_score = score
                        best_overall_metric = metric
                        best_overall_threshold = threshold
                        best_overall_linkage = method
                except Exception:
                    pass
                finally:
                    progress.update(task, advance=1)

    return best_overall_linkage, best_overall_metric, best_overall_threshold
