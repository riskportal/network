import networkx as nx
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform

from spp.network.annotation import chop_and_filter


def get_network_neighborhoods(network, neighborhood_distance_algorithm, neighborhood_radius):
    all_shortest_paths = {}
    neighborhoods = np.zeros([network.number_of_nodes(), network.number_of_nodes()], dtype=int)
    all_x = list(dict(network.nodes.data("x")).values())

    if neighborhood_distance_algorithm == "euclidean":
        node_radius = neighborhood_radius * (np.max(all_x) - np.min(all_x))
        x = np.matrix(network.nodes.data("x"))[:, 1]
        y = np.matrix(network.nodes.data("y"))[:, 1]
        node_coordinates = np.concatenate([x, y], axis=1)
        node_distances = squareform(pdist(node_coordinates, "euclidean"))
        neighborhoods[node_distances < node_radius] = 1
        return neighborhoods

    if neighborhood_distance_algorithm == "shortpath_weighted_layout":
        network_radius = neighborhood_radius * (np.max(all_x) - np.min(all_x))
        all_shortest_paths = dict(
            nx.all_pairs_dijkstra_path_length(network, weight="length", cutoff=network_radius)
        )
    if neighborhood_distance_algorithm == "shortpath":
        network_radius = neighborhood_radius
        all_shortest_paths = dict(nx.all_pairs_dijkstra_path_length(network, cutoff=network_radius))
    neighbors = [(s, t) for s in all_shortest_paths for t in all_shortest_paths[s].keys()]
    for i in neighbors:
        neighborhoods[i] = 1
    return neighborhoods


def define_domains(
    neighborhood_enrichment_matrix,
    binary_enrichment_matrix_below_alpha,
    annotation_matrix,
    group_distance_metric,
    group_distance_threshold,
):
    m = binary_enrichment_matrix_below_alpha[:, annotation_matrix["top attributes"]].T
    Z = linkage(m, method="average", metric=group_distance_metric)
    max_d = np.max(Z[:, 2] * group_distance_threshold)
    domains = fcluster(Z, max_d, criterion="distance")

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


def trim_domains(annotation_matrix, domains_matrix, min_annotation_size):
    # Remove domains that are the top choice for less than a certain number of neighborhoods
    domain_counts = domains_matrix["primary domain"].value_counts()
    to_remove = domain_counts[domain_counts < min_annotation_size].index

    annotation_matrix["domain"].replace(to_remove, 0, inplace=True)

    domains_matrix.loc[
        domains_matrix["primary domain"].isin(to_remove), ["primary domain", "primary nes"]
    ] = 0

    # Rename the domains (simple renumber)
    renumber_dict = {domain: idx for idx, domain in enumerate(annotation_matrix["domain"].unique())}

    annotation_matrix["domain"] = annotation_matrix["domain"].map(renumber_dict)
    domains_matrix["primary domain"] = domains_matrix["primary domain"].map(renumber_dict)
    domains_matrix.drop(columns=to_remove, inplace=True)

    # Make labels for each domain
    domains_labels = annotation_matrix.groupby("domain")["name"].apply(chop_and_filter)
    trimmed_domains_matrix = pd.DataFrame(
        data={"id": annotation_matrix["domain"].unique(), "label": domains_labels}
    )
    trimmed_domains_matrix.set_index("id", drop=False, inplace=True)

    return annotation_matrix, domains_matrix, trimmed_domains_matrix
