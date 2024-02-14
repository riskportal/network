from collections import Counter

import community as community_louvain
import networkx as nx
import numpy as np
import pandas as pd
from rich import print
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import silhouette_score

from spp.network.annotation import chop_and_filter


def get_network_neighborhoods(
    network, neighborhood_distance_algorithm, neighborhood_radius, louvain_resolution=1.0
):
    neighborhoods = np.zeros((network.number_of_nodes(), network.number_of_nodes()), dtype=int)
    all_x = list(dict(network.nodes.data("x")).values())

    if neighborhood_distance_algorithm == "euclidean":
        node_radius = neighborhood_radius * (np.max(all_x) - np.min(all_x))
        x = np.array(list(dict(network.nodes.data("x")).values()))
        y = np.array(list(dict(network.nodes.data("y")).values()))
        node_coordinates = np.stack((x, y), axis=1)
        node_distances = squareform(pdist(node_coordinates, "euclidean"))
        neighborhoods[node_distances < node_radius] = 1

    elif neighborhood_distance_algorithm in ["shortpath_weighted", "shortpath"]:
        network_radius = (
            neighborhood_radius * (np.max(all_x) - np.min(all_x))
            if neighborhood_distance_algorithm == "shortpath_weighted"
            else neighborhood_radius
        )
        all_shortest_paths = dict(
            nx.all_pairs_dijkstra_path_length(
                network,
                weight="length"
                if neighborhood_distance_algorithm == "shortpath_weighted"
                else None,
                cutoff=network_radius,
            )
        )
        for source, targets in all_shortest_paths.items():
            for target, _ in targets.items():
                neighborhoods[source, target] = 1

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
    group_distance_metric,
    group_distance_threshold=0.0,
):
    m = binary_enrichment_matrix_below_alpha[:, annotation_matrix["top attributes"]].T
    Z = linkage(m, method="average", metric=group_distance_metric)

    # Automatically compute a threshold if no threshold is provided
    if not group_distance_threshold:
        # Calculate silhouette scores for a range of thresholds
        thresholds = np.linspace(0.1, 0.9, 9)  # Example range of thresholds
        silhouette_scores = []
        for threshold in thresholds:
            max_d = np.max(Z[:, 2]) * threshold
            clusters = fcluster(Z, max_d, criterion="distance")
            score = silhouette_score(m, clusters, metric=group_distance_metric)
            silhouette_scores.append(score)
        # Find the threshold with the highest silhouette score
        group_distance_threshold = thresholds[np.argmax(silhouette_scores)]
        print(
            f"[yellow]Automatically computed threshold: [red]{round(group_distance_threshold, 1)}[/red][/yellow]"
        )

    max_d_optimal = np.max(Z[:, 2]) * group_distance_threshold
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
    # annotation_matrix = annotation_matrix[~annotation_matrix["domain"].isin(to_remove)]
    # domains_matrix = domains_matrix[~domains_matrix["primary domain"].isin(to_remove)]
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
    import numpy as np

    # Extract values from the dictionary
    values = np.array(list(data_dict.values()))

    # Calculate mean and standard deviation
    mean = np.mean(values)
    std_dev = np.std(values)

    # Function to calculate Z-score
    def calculate_z_score(value):
        return (value - mean) / std_dev

    # Identify outliers
    outlier_keys = []
    for key, value in data_dict.items():
        z_score = calculate_z_score(value)
        if abs(z_score) > z_score_threshold:
            outlier_keys.append(key)

    return outlier_keys


def preprocess_network(network, weight_threshold=None):
    """Remove edges below a certain weight threshold."""
    if weight_threshold is not None:
        edges_to_remove = [
            (u, v) for u, v, d in network.edges(data=True) if d["weight"] < weight_threshold
        ]
        network.remove_edges_from(edges_to_remove)
    return network
