"""
risk/network/annotations
~~~~~~~~~~~~~~~~~~~~~~~
"""

from collections import Counter
from itertools import compress, permutations

import networkx as nx
import nltk
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Ensure you have the necessary NLTK data (uncomment if needed)
# nltk.download("punkt")
# nltk.download("stopwords")


def load_annotations(network, annotations_input):
    """
    Convert annotations input to a DataFrame and reindex based on the network's node labels.

    Args:
        annotations_input (dict): A dictionary with annotations.

    Returns:
        dict: A dictionary containing ordered nodes, ordered annotations, and the annotations matrix.
    """
    # Flatten the dictionary to a list of tuples for easier DataFrame creation
    flattened_annotations = [
        (node, annotation) for annotation, nodes in annotations_input.items() for node in nodes
    ]
    # Create a DataFrame from the flattened list
    annotations = pd.DataFrame(flattened_annotations, columns=["Node", "Annotations"])
    annotations["Is Member"] = 1

    # Pivot the DataFrame to create a binary matrix with nodes as rows and annotations as columns
    annotations_pivot = annotations.pivot_table(
        index="Node", columns="Annotations", values="Is Member", fill_value=0, dropna=False
    )
    # Get the list of node labels in the order they appear in the network
    node_label_order = list(nx.get_node_attributes(network, "label").values())
    # Reindex the annotations matrix with the node labels from the network
    annotations_pivot = annotations_pivot.reindex(index=node_label_order)
    # Check if the reindexed annotations matrix has any non-null values
    if annotations_pivot.notnull().sum().sum() == 0:
        raise ValueError(
            "No annotations found in the annotations file for the nodes in the network."
        )

    # Extract ordered nodes and annotations
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
    binary_enrichment_matrix,
    min_cluster_size=5,
    max_cluster_size=1000,
):
    """
    Define top annotations based on neighborhood enrichment sums and binary enrichment matrix.

    Args:
        network (NetworkX graph): The network graph.
        ordered_annotation_labels (list): List of ordered annotation labels.
        neighborhood_enrichment_sums (list): List of neighborhood enrichment sums.
        binary_enrichment_matrix (np.ndarray): Binary enrichment matrix below alpha threshold.
        min_cluster_size (int, optional): Minimum cluster size. Defaults to 5.
        max_cluster_size (int, optional): Maximum cluster size. Defaults to 1000.

    Returns:
        pd.DataFrame: DataFrame with top annotations and their properties.
    """
    annotations_enrichment_matrix = pd.DataFrame(
        {
            "id": range(len(ordered_annotation_labels)),
            "words": ordered_annotation_labels,
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
            compress(list(network), binary_enrichment_matrix[:, attribute] > 0)
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


def get_description(words_column):
    """Process input Series to identify and return the top N frequent, significant words,
    filtering based on stopwords and similarity (Jaccard index)."""
    # Tokenize the concatenated string and filter out stopwords and non-alphabetic words in one step
    stop_words = set(stopwords.words("english"))
    words = [
        word.lower()
        for word in word_tokenize(words_column.str.cat(sep=" "))
        if word.isalpha() and word.lower() not in stop_words
    ]

    # Simplify the word list to remove similar words based on the Jaccard index
    simplified_words = _simplify_word_list(words, threshold=0.90)
    description = _generate_coherent_description(simplified_words)
    return description


def _simplify_word_list(words, threshold=0.80):
    """Filter out words that are too similar based on the Jaccard index, keeping the word with the higher count."""
    word_counts = Counter(words)
    filtered_words = []
    used_words = set()

    for word in word_counts:
        if word in used_words:
            continue
        word_set = set(word)
        similar_words = [
            other_word
            for other_word in word_counts
            if _jaccard_index(word_set, set(other_word)) >= threshold
        ]
        similar_words.sort(key=lambda w: word_counts[w], reverse=True)
        best_word = similar_words[0]
        filtered_words.append(best_word)
        used_words.update(similar_words)

    final_words = [word for word in words if word in filtered_words]

    return final_words


def _jaccard_index(set1, set2):
    """Calculate the Jaccard Index of two sets."""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union else 0


def _generate_coherent_description(words):
    """Generate a coherent description from a list of words."""
    # Count the frequency of each word
    word_counts = Counter(words)
    # Get the most common words
    most_common_words = [word for word, _ in word_counts.most_common()]
    # Filter out common stopwords
    stop_words = set(stopwords.words("english"))
    filtered_words = [word for word in most_common_words if word.lower() not in stop_words]
    # Generate permutations of the filtered words to find a logical order
    perm = permutations(filtered_words)
    # Assume the first permutation as the logical sequence (since they're all equally likely without additional context)
    logical_sequence = next(perm)
    # Join the words to form a coherent description
    description = " ".join(logical_sequence)
    return description
