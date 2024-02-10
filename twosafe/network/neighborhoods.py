from itertools import compress

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform


def get_network_neighborhoods(network, neighborhood_distance_algorithm, neighborhood_radius):
    print("Loading network neighborhoods...")
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
    min_annotation_size = 0  ## NOTE: FOR DEV
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
    print(annotation_enrichment_matrix)

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
