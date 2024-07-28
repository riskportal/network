"""
risk/network/annotations
~~~~~~~~~~~~~~~~~~~~~~~
"""

import json
from collections import Counter
from itertools import compress

import networkx as nx
import nltk
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Ensure you have the necessary NLTK data (uncomment if needed)
# nltk.download("punkt")
# nltk.download("stopwords")


def load_annotations(network, annotations_filepath):
    # Convert JSON data to a Python dictionary
    with open(annotations_filepath, "r") as file:
        annotations_input = json.load(file)
    # Flatten the dictionary for easier DataFrame creation
    flattened_annotations = [
        (node, annotations) for annotations, nodes in annotations_input.items() for node in nodes
    ]
    # Create a DataFrame
    annotations = pd.DataFrame(flattened_annotations, columns=["Node", "Annotations"])
    annotations["Is Member"] = 1
    # Pivot the DataFrame to achieve the desired format
    annotations_pivot = annotations.pivot_table(
        index="Node", columns="Annotations", values="Is Member", fill_value=0, dropna=False
    )
    # Get list of node labels as ordered in a graph object
    node_label_order = list(nx.get_node_attributes(network, "label").values())
    # This will reindex the annotations matrix with node labels as found in annotations file - those that are not found,
    # (i.e., rows) will be set to NaN values
    annotations_pivot = annotations_pivot.reindex(index=node_label_order)
    if annotations_pivot.notnull().sum().sum() == 0:
        raise ValueError(
            "No annotations found in the annotations file for the nodes in the network."
        )

    ordered_nodes = tuple(annotations_pivot.index)
    ordered_annotations = tuple(annotations_pivot.columns)
    return {
        "ordered_nodes": ordered_nodes,
        "ordered_annotations": ordered_annotations,
        "matrix": annotations_pivot.to_numpy(),
    }


def define_top_annotations(
    network,
    ordered_annotation_labels,
    neighborhood_enrichment_sums,
    binary_enrichment_matrix_below_alpha,
    min_cluster_size=5,
    max_cluster_size=1000,
):
    annotations_enrichment_matrix = pd.DataFrame(
        {
            "id": range(len(ordered_annotation_labels)),
            "name": ordered_annotation_labels,
            "num enriched neighborhoods": neighborhood_enrichment_sums,
        }
    )
    annotations_enrichment_matrix["top attributes"] = False
    # Requirement 1: a minimum and maximum number of enriched neighborhoods
    # Combining both conditions with a logical AND
    annotations_enrichment_matrix.loc[
        (annotations_enrichment_matrix["num enriched neighborhoods"] >= min_cluster_size)
        & (annotations_enrichment_matrix["num enriched neighborhoods"] <= max_cluster_size),
        "top attributes",
    ] = True

    # Requirement 2: 1 connected component in the subnetwork of enriched neighborhoods:
    annotations_enrichment_matrix["num connected components"] = 0
    annotations_enrichment_matrix["size connected components"] = None
    annotations_enrichment_matrix["size connected components"] = annotations_enrichment_matrix[
        "size connected components"
    ].astype(object)
    annotations_enrichment_matrix["num large connected components"] = 0

    for attribute in annotations_enrichment_matrix.index.values[
        annotations_enrichment_matrix["top attributes"]
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
            np.logical_and(
                size_connected_components >= min_cluster_size,
                size_connected_components <= max_cluster_size,
            )
        )

        annotations_enrichment_matrix.loc[
            attribute, "num connected components"
        ] = num_connected_components
        annotations_enrichment_matrix.at[
            attribute, "size connected components"
        ] = size_connected_components
        annotations_enrichment_matrix.loc[
            attribute, "num large connected components"
        ] = num_large_connected_components

    # Exclude attributes that have more than 1 connected component
    annotations_enrichment_matrix.loc[
        annotations_enrichment_matrix["num connected components"] > 1, "top attributes"
    ] = False

    return annotations_enrichment_matrix


def chop_and_filter(s, top_words_count=6):
    """Process input Series to identify and return the top N frequent, significant words,
    filtering based on stopwords and similarity (Jaccard index)."""
    # Tokenize the concatenated string and filter out stopwords and non-alphabetic words in one step
    stop_words = set(stopwords.words("english"))
    words = [
        word.lower()
        for word in word_tokenize(s.str.cat(sep=" "))
        if word.isalpha() and word.lower() not in stop_words
    ]

    # Simplify the word list to remove similar words based on the Jaccard index
    simplified_words = _simplify_word_list(words, threshold=0.90)

    # Count the occurrences of each word and sort them by frequency in descending order
    word_counts = Counter(simplified_words)
    sorted_words = sorted(word_counts, key=word_counts.get, reverse=True)

    # Select the top N words
    top_words = sorted_words[:top_words_count]
    return ", ".join(top_words)


def _simplify_word_list(words, threshold=0.90):
    """Filter out words that are too similar based on the Jaccard index."""
    filtered_words = []
    for word in words:
        word_set = set(word)
        if all(
            _jaccard_index(word_set, set(other_word)) < threshold for other_word in filtered_words
        ):
            filtered_words.append(word)
    return filtered_words


def _jaccard_index(set1, set2):
    """Calculate the Jaccard Index of two sets."""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union else 0
