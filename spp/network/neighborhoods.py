from itertools import compress

import networkx as nx
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform


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


def define_top_attributes(
    network,
    ordered_annotation_labels,
    ordered_annotation_enrichment,
    binary_enrichment_matrix_below_alpha,
    min_annotation_size,
    unimodality_type,
):
    annotation_enrichment_matrix = pd.DataFrame(
        {
            "id": range(len(ordered_annotation_labels)),
            "name": ordered_annotation_labels,
            "num enriched neighborhoods": ordered_annotation_enrichment,
        }
    )
    annotation_enrichment_matrix["top attributes"] = False
    # Requirement 1: a minimum number of enriched neighborhoods
    annotation_enrichment_matrix.loc[
        annotation_enrichment_matrix["num enriched neighborhoods"] >= min_annotation_size,
        "top attributes",
    ] = True

    # Requirement 2: 1 connected component in the subnetwork of enriched neighborhoods
    if unimodality_type == "connectivity":
        annotation_enrichment_matrix["num connected components"] = 0
        annotation_enrichment_matrix["size connected components"] = None
        annotation_enrichment_matrix["size connected components"] = annotation_enrichment_matrix[
            "size connected components"
        ].astype(object)
        annotation_enrichment_matrix["num large connected components"] = 0

        for attribute in annotation_enrichment_matrix.index.values[
            annotation_enrichment_matrix["top attributes"]
        ]:
            enriched_neighborhoods = list(
                compress(list(network), binary_enrichment_matrix_below_alpha[:, attribute] > 0)
            )
            enriched_network = nx.subgraph(network, enriched_neighborhoods)

            connected_components = sorted(
                nx.connected_components(enriched_network), key=len, reverse=True
            )
            num_connected_components = len(connected_components)
            size_connected_components = np.array([len(c) for c in connected_components])
            num_large_connected_components = np.sum(
                size_connected_components >= min_annotation_size
            )

            annotation_enrichment_matrix.loc[
                attribute, "num connected components"
            ] = num_connected_components
            annotation_enrichment_matrix.at[
                attribute, "size connected components"
            ] = size_connected_components
            annotation_enrichment_matrix.loc[
                attribute, "num large connected components"
            ] = num_large_connected_components

        # Exclude attributes that have more than 1 connected component
        annotation_enrichment_matrix.loc[
            annotation_enrichment_matrix["num connected components"] > 1, "top attributes"
        ] = False
        return annotation_enrichment_matrix


def define_domains(
    neighborhood_enrichment_matrix,
    binary_enrichment_matrix_below_alpha,
    annotation_matrix,
    group_distance_metric,
    group_distance_threshold,
):
    # # NOTE: FOR DEV
    # annotation_matrix["top attributes"] = np.random.choice(
    #     [True, False], size=len(annotation_matrix)
    # )
    # END
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
    node2domain["primary nes"] = o.values[i.index, i.values]

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


from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK stopwords if not already downloaded
try:
    _ = stopwords.words("english")
except LookupError:
    import nltk

    nltk.download("stopwords")


def chop_and_filter(s, top_words_count=5):
    # Concatenate all strings in the Series into a single string
    single_str = s.str.cat(sep=" ")

    # Tokenize the string into a list of words
    single_list = word_tokenize(single_str)

    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    single_list = [
        word.lower() for word in single_list if word.isalpha() and word.lower() not in stop_words
    ]

    # Count the occurrences of each word and sort them by frequency in descending order
    single_list_count = Counter(single_list)
    sorted_words = sorted(single_list_count, key=single_list_count.get, reverse=True)

    # Join the top N words and return as a comma-separated string
    result_words = sorted_words[:top_words_count]
    return ", ".join(result_words)
