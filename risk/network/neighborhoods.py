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

from risk.network.annotation import chop_and_filter
from risk.network.constants import GROUP_LINKAGE_METHODS, GROUP_DISTANCE_METRICS

# Suppress DataConversionWarning
warnings.filterwarnings(action="ignore", category=DataConversionWarning)


def get_network_neighborhoods(
    network,
    neighborhood_distance_algorithm,
    neighborhood_diameter,
    compute_sphere=False,
    louvain_resolution=1.0,
):
    # Take account the curvature of a sphere to sync neighborhood radius between 2D and 3D graphs
    neighborhood_radius = neighborhood_diameter * (np.pi if compute_sphere else 1) / 2
    # Initialize neighborhoods matrix
    neighborhoods = np.zeros((network.number_of_nodes(), network.number_of_nodes()), dtype=int)

    if neighborhood_distance_algorithm == "euclidean":
        x = np.array(list(dict(network.nodes.data("x")).values()))
        y = np.array(list(dict(network.nodes.data("y")).values()))
        node_coordinates = np.stack((x, y), axis=1)
        node_distances = squareform(pdist(node_coordinates, "euclidean"))
        neighborhoods[node_distances < neighborhood_radius] = 1

    elif neighborhood_distance_algorithm == "shortpath":
        # First, compute Djikstra's shortest path
        all_shortest_paths = dict(
            nx.all_pairs_dijkstra_path_length(
                network,
                weight="length",
                cutoff=neighborhood_radius,
            )
        )
        # Serves for better fine tuning... literally incorporate length for neighborhood score
        for source, targets in all_shortest_paths.items():
            for target, length in targets.items():
                # Scale the length of node distance after computing node distances on a normalized network (look at io file)
                neighborhoods[source, target] = (
                    1 if np.isnan(length) or length == 0 else np.sqrt(1 / length)
                )

    elif neighborhood_distance_algorithm == "louvain":
        partition = community_louvain.best_partition(network, resolution=louvain_resolution)
        for node_i, community_i in partition.items():
            for node_j, community_j in partition.items():
                if community_i == community_j:
                    neighborhoods[node_i, node_j] = 1

    elif neighborhood_distance_algorithm == "affinity_propagation":
        # Convert the network into a distance matrix
        distance_matrix = nx.floyd_warshall_numpy(network)
        # Affinity Propagation requires similarities, not distances, hence we negate the distances
        similarity_matrix = -distance_matrix
        # Apply Affinity Propagation
        clustering = AffinityPropagation(affinity="precomputed", random_state=5)
        clustering.fit(similarity_matrix)
        # Update neighborhoods matrix based on Affinity Propagation clustering results
        labels = clustering.labels_
        for i, label_i in enumerate(labels):
            for j, label_j in enumerate(labels):
                if label_i == label_j:
                    neighborhoods[i, j] = 1

    return neighborhoods


def define_domains(
    neighborhood_enrichment_matrix,
    binary_enrichment_matrix_below_alpha,
    annotation_matrix,
    group_distance_linkage,
    group_distance_metric,
):
    m = binary_enrichment_matrix_below_alpha[:, annotation_matrix["top attributes"]].T
    best_linkage, best_metric, best_threshold, _ = optimize_silhouette_across_linkage_and_metrics(
        m, group_distance_linkage, group_distance_metric
    )
    Z = linkage(m, method=best_linkage, metric=best_metric)
    print(
        f"[cyan]Using [blue]distance linkage[/blue] [yellow]'{best_linkage}'[/yellow] with [blue]distance metric[/blue] [yellow]'{best_metric}'[/yellow]...[/cyan]"
    )
    print(f"[yellow]Optimal distance threshold: [red]{round(best_threshold, 3)}[/red][/yellow]")

    max_d_optimal = np.max(Z[:, 2]) * best_threshold
    domains = fcluster(Z, max_d_optimal, criterion="distance")

    annotation_matrix["domain"] = 0
    annotation_matrix.loc[annotation_matrix["top attributes"], "domain"] = domains

    # Assign nodes to domains
    node2nes = pd.DataFrame(
        data=neighborhood_enrichment_matrix,
        columns=[annotation_matrix.index.values, annotation_matrix["domain"]],
    )
    node2nes_binary = pd.DataFrame(
        data=binary_enrichment_matrix_below_alpha,
        columns=[annotation_matrix.index.values, annotation_matrix["domain"]],
    )
    node2domain = node2nes_binary.groupby(level="domain", axis=1).sum()
    t_max = node2domain.loc[:, 1:].max(axis=1)
    t_idxmax = node2domain.loc[:, 1:].idxmax(axis=1)
    t_idxmax[t_max == 0] = 0

    node2domain["primary domain"] = t_idxmax
    # Get the max NES for the primary domain
    o = node2nes.groupby(level="domain", axis=1).max()
    i = pd.Series(t_idxmax)
    # Extract values from 'o' at the indices specified by 'i'
    # Subtrace i.values - 1, as i.values starts with 1 and we are looking at a 0-based index
    node2domain["primary nes"] = o.values[i.index, i.values - 1]

    return node2domain


def trim_domains(annotation_matrix, domains_matrix, min_cluster_size):
    # Remove domains that are the top choice for less than a certain number of neighborhoods
    domain_counts = domains_matrix["primary domain"].value_counts()
    to_remove = list(domain_counts[domain_counts < min_cluster_size].index)
    to_remove.extend(find_outlier_domains(Counter(domains_matrix["primary domain"])))
    annotation_matrix["domain"].replace(to_remove, 888888, inplace=True)
    domains_matrix.loc[
        domains_matrix["primary domain"].isin(to_remove), ["primary domain", "primary nes"]
    ] = 888888

    # Make labels for each domain
    domains_labels = (
        annotation_matrix.groupby("domain")["name"].apply(chop_and_filter).reset_index()
    )
    trimmed_domains_matrix = pd.DataFrame(domains_labels).rename(
        columns={"domain": "id", "name": "label"}
    )
    trimmed_domains_matrix.set_index("id", drop=False, inplace=True)

    return annotation_matrix, domains_matrix, trimmed_domains_matrix


def find_outlier_domains(data_dict, z_score_threshold=3):
    # Extract values from the dictionary
    values = np.array(list(data_dict.values()))
    # Calculate mean and standard deviation
    mean = np.mean(values)
    std_dev = np.std(values)
    # Identify outliers
    outlier_keys = []
    for key, value in data_dict.items():
        z_score = (value - mean) / std_dev
        if abs(z_score) > z_score_threshold:
            outlier_keys.append(key)

    return outlier_keys


# Function to perform binary search silhouette for a given metric and linkage method
def binary_search_silhouette(
    Z, m, group_distance_metric, lower_bound=0.0, upper_bound=1.0, tolerance=0.01
):
    best_threshold = lower_bound
    best_score = -np.inf

    while upper_bound - lower_bound > tolerance:
        mid = (lower_bound + upper_bound) / 2
        max_d_mid = np.max(Z[:, 2]) * mid
        clusters_mid = fcluster(Z, max_d_mid, criterion="distance")

        max_d_high = np.max(Z[:, 2]) * (mid + tolerance)
        clusters_high = fcluster(Z, max_d_high, criterion="distance")

        try:
            score_mid = silhouette_score(m, clusters_mid, metric=group_distance_metric)
        except ValueError:
            score_mid = -np.inf

        try:
            score_high = silhouette_score(m, clusters_high, metric=group_distance_metric)
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


def optimize_silhouette_across_linkage_and_metrics(
    m, group_distance_linkage, group_distance_metric
):
    best_overall_score = -np.inf
    best_overall_metric = None
    best_overall_threshold = None
    best_overall_linkage = None
    group_linkage_methods = (
        GROUP_LINKAGE_METHODS if group_distance_linkage == "auto" else [group_distance_linkage]
    )
    group_distance_metrics = (
        GROUP_DISTANCE_METRICS if group_distance_metric == "auto" else [group_distance_metric]
    )
    total = len(group_linkage_methods) * len(group_distance_metrics)

    with Progress() as progress:
        task = progress.add_task(
            "[cyan]Evaluating [yellow]optimal[/yellow] [blue]distance linkage[/blue] and [blue]distance metric[/blue]...",
            total=total,
        )

        for linkage_method in group_linkage_methods:
            for metric in group_distance_metrics:
                try:
                    Z = linkage(m, method=linkage_method, metric=metric)
                    threshold, score = binary_search_silhouette(Z, m, metric)
                    if score > best_overall_score:
                        best_overall_score = score
                        best_overall_metric = metric
                        best_overall_threshold = threshold
                        best_overall_linkage = linkage_method
                except Exception as e:
                    # Ignoring exceptions that arise due to incompatibility between metrics and linkage methods
                    pass
                finally:
                    progress.update(task, advance=1)

    return best_overall_linkage, best_overall_metric, best_overall_threshold, best_overall_score
