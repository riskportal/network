from collections import Counter
from itertools import compress

import networkx as nx
import nltk
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Ensure you have the necessary NLTK data (uncomment if needed)
# nltk.download('punkt')
# nltk.download('stopwords')


def define_top_annotations(
    network,
    ordered_annotation_labels,
    ordered_annotation_enrichment,
    binary_enrichment_matrix_below_alpha,
    min_cluster_size,
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
        annotation_enrichment_matrix["num enriched neighborhoods"] >= min_cluster_size,
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
            num_large_connected_components = np.sum(size_connected_components >= min_cluster_size)

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
    simplified_words = simplify_word_list(words, threshold=0.90)

    # Count the occurrences of each word and sort them by frequency in descending order
    word_counts = Counter(simplified_words)
    sorted_words = sorted(word_counts, key=word_counts.get, reverse=True)

    # Select the top N words
    top_words = sorted_words[:top_words_count]
    return ", ".join(top_words)


def simplify_word_list(words, threshold=0.90):
    """Filter out words that are too similar based on the Jaccard index."""
    filtered_words = []
    for word in words:
        word_set = set(word)
        if all(
            jaccard_index(word_set, set(other_word)) < threshold for other_word in filtered_words
        ):
            filtered_words.append(word)
    return filtered_words


def jaccard_index(set1, set2):
    """Calculate the Jaccard Index of two sets."""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union else 0
