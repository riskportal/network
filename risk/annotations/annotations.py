"""
risk/annotations/annotations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from collections import Counter
from itertools import compress
from typing import Any, Dict, List, Set

import networkx as nx
import nltk
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


def _setup_nltk():
    """Ensure necessary NLTK data is downloaded."""
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")


# Ensure you have the necessary NLTK data
_setup_nltk()


def load_annotations(network: nx.Graph, annotations_input: Dict[str, Any]) -> Dict[str, Any]:
    """Convert annotations input to a DataFrame and reindex based on the network's node labels.

    Args:
        annotations_input (dict): A dictionary with annotations.

    Returns:
        dict: A dictionary containing ordered nodes, ordered annotations, and the binary annotations matrix.
    """
    # Flatten the dictionary to a list of tuples for easier DataFrame creation
    flattened_annotations = [
        (node, annotation) for annotation, nodes in annotations_input.items() for node in nodes
    ]
    # Create a DataFrame from the flattened list
    annotations = pd.DataFrame(flattened_annotations, columns=["Node", "Annotations"])
    annotations["Is Member"] = 1
    # Pivot to create a binary matrix with nodes as rows and annotations as columns
    annotations_pivot = annotations.pivot_table(
        index="Node", columns="Annotations", values="Is Member", fill_value=0, dropna=False
    )
    # Reindex the annotations matrix based on the node labels from the network
    node_label_order = list(nx.get_node_attributes(network, "label").values())
    annotations_pivot = annotations_pivot.reindex(index=node_label_order)
    # Raise an error if no valid annotations are found for the nodes in the network
    if annotations_pivot.notnull().sum().sum() == 0:
        raise ValueError(
            "No annotations found in the annotations file for the nodes in the network."
        )

    # Remove columns with all zeros to improve performance
    annotations_pivot = annotations_pivot.loc[:, annotations_pivot.sum(axis=0) != 0]
    # Extract ordered nodes and annotations
    ordered_nodes = tuple(annotations_pivot.index)
    ordered_annotations = tuple(annotations_pivot.columns)
    # Convert the annotations_pivot matrix to a numpy array and ensure it's binary
    annotations_pivot_numpy = (annotations_pivot.fillna(0).to_numpy() > 0).astype(int)

    return {
        "ordered_nodes": ordered_nodes,
        "ordered_annotations": ordered_annotations,
        "matrix": annotations_pivot_numpy,
    }


def define_top_annotations(
    network: nx.Graph,
    ordered_annotation_labels: List[str],
    neighborhood_enrichment_sums: List[int],
    binary_enrichment_matrix: np.ndarray,
    min_cluster_size: int = 5,
    max_cluster_size: int = 1000,
) -> pd.DataFrame:
    """Define top annotations based on neighborhood enrichment sums and binary enrichment matrix.

    Args:
        network (NetworkX graph): The network graph.
        ordered_annotation_labels (list of str): List of ordered annotation labels.
        neighborhood_enrichment_sums (list of int): List of neighborhood enrichment sums.
        binary_enrichment_matrix (np.ndarray): Binary enrichment matrix below alpha threshold.
        min_cluster_size (int, optional): Minimum cluster size. Defaults to 5.
        max_cluster_size (int, optional): Maximum cluster size. Defaults to 1000.

    Returns:
        pd.DataFrame: DataFrame with top annotations and their properties.
    """
    # Create DataFrame to store annotations and their neighborhood enrichment sums
    annotations_enrichment_matrix = pd.DataFrame(
        {
            "id": range(len(ordered_annotation_labels)),
            "words": ordered_annotation_labels,
            "neighborhood enrichment sums": neighborhood_enrichment_sums,
        }
    )
    annotations_enrichment_matrix["top attributes"] = False
    # Apply size constraints to identify potential top attributes
    annotations_enrichment_matrix.loc[
        (annotations_enrichment_matrix["neighborhood enrichment sums"] >= min_cluster_size)
        & (annotations_enrichment_matrix["neighborhood enrichment sums"] <= max_cluster_size),
        "top attributes",
    ] = True
    # Initialize columns for connected components analysis
    annotations_enrichment_matrix["num connected components"] = 0
    annotations_enrichment_matrix["size connected components"] = None
    annotations_enrichment_matrix["size connected components"] = annotations_enrichment_matrix[
        "size connected components"
    ].astype(object)
    annotations_enrichment_matrix["num large connected components"] = 0

    for attribute in annotations_enrichment_matrix.index.values[
        annotations_enrichment_matrix["top attributes"]
    ]:
        # Identify enriched neighborhoods based on the binary enrichment matrix
        enriched_neighborhoods = list(
            compress(list(network), binary_enrichment_matrix[:, attribute])
        )
        enriched_network = nx.subgraph(network, enriched_neighborhoods)
        # Analyze connected components within the enriched subnetwork
        connected_components = sorted(
            nx.connected_components(enriched_network), key=len, reverse=True
        )
        size_connected_components = np.array([len(c) for c in connected_components])

        # Filter the size of connected components by min_cluster_size and max_cluster_size
        filtered_size_connected_components = size_connected_components[
            (size_connected_components >= min_cluster_size)
            & (size_connected_components <= max_cluster_size)
        ]
        # Calculate the number of connected components and large connected components
        num_connected_components = len(connected_components)
        num_large_connected_components = len(filtered_size_connected_components)

        # Assign the number of connected components
        annotations_enrichment_matrix.loc[attribute, "num connected components"] = (
            num_connected_components
        )
        # Filter out attributes with more than one connected component
        annotations_enrichment_matrix.loc[
            annotations_enrichment_matrix["num connected components"] > 1, "top attributes"
        ] = False
        # Assign the number of large connected components
        annotations_enrichment_matrix.loc[attribute, "num large connected components"] = (
            num_large_connected_components
        )
        # Assign the size of connected components, ensuring it is always a list
        annotations_enrichment_matrix.at[attribute, "size connected components"] = (
            filtered_size_connected_components.tolist()
        )

    return annotations_enrichment_matrix


def get_description(words_column: pd.Series) -> str:
    """Process input Series to identify and return the top frequent, significant words,
    filtering based on stopwords and gracefully handling numerical strings.

    Args:
        words_column (pd.Series): A pandas Series containing strings to process.

    Returns:
        str: A coherent description formed from the most frequent and significant words.
    """
    # Concatenate all rows into a single string and tokenize into words
    all_words = words_column.str.cat(sep=" ")
    tokens = word_tokenize(all_words)

    # Separate numeric tokens
    numeric_tokens = [token for token in tokens if token.replace(".", "", 1).isdigit()]
    # If there's only one unique numeric value, return it directly as a string
    unique_numeric_values = set(numeric_tokens)
    if len(unique_numeric_values) == 1:
        return f"{list(unique_numeric_values)[0]}"

    # Ensure that all values in 'words' are strings and include both alphabetic and numeric tokens
    words = [
        str(
            word.lower() if word.istitle() else word
        )  # Convert to string and lowercase all words except proper nouns (e.g., RNA, mRNA)
        for word in tokens
        if word.isalpha()
        or word.replace(".", "", 1).isdigit()  # Keep alphabetic words and numeric strings
    ]
    # Generate a coherent description from the processed words
    description = _generate_coherent_description(words)

    return description


def _simplify_word_list(words: List[str], threshold: float = 0.80) -> List[str]:
    """Filter out words that are too similar based on the Jaccard index, keeping the word with the higher count.

    Args:
        words (list of str): The list of words to be filtered.
        threshold (float, optional): The similarity threshold for the Jaccard index. Defaults to 0.80.

    Returns:
        list of str: A list of filtered words, where similar words are reduced to the most frequent one.
    """
    # Count the occurrences of each word
    word_counts = Counter(words)
    filtered_words = []
    used_words = set()
    # Iterate through the words to find similar words
    for word in word_counts:
        if word in used_words:
            continue

        word_set = set(word)
        # Find similar words based on the Jaccard index
        similar_words = [
            other_word
            for other_word in word_counts
            if _calculate_jaccard_index(word_set, set(other_word)) >= threshold
        ]
        # Sort by frequency and choose the most frequent word
        similar_words.sort(key=lambda w: word_counts[w], reverse=True)
        best_word = similar_words[0]
        filtered_words.append(best_word)
        used_words.update(similar_words)

    final_words = [word for word in words if word in filtered_words]

    return final_words


def _calculate_jaccard_index(set1: Set[Any], set2: Set[Any]) -> float:
    """Calculate the Jaccard Index of two sets.

    Args:
        set1 (set): The first set for comparison.
        set2 (set): The second set for comparison.

    Returns:
        float: The Jaccard Index, which is the ratio of the intersection to the union of the two sets.
               Returns 0 if the union of the sets is empty.
    """
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union else 0


def _generate_coherent_description(words: List[str]) -> str:
    """Generate a coherent description from a list of words or numerical string values.
    If there is only one unique entry, return it directly.

    Args:
        words (list): A list of words or numerical string values.

    Returns:
        str: A coherent description formed by arranging the words in a logical sequence.
    """
    # If there are no words, return a keyword indicating no data is available
    if not words:
        return "N/A"

    # If there's only one unique word, return it directly
    unique_words = set(words)
    if len(unique_words) == 1:
        return list(unique_words)[0]

    # Count the frequency of each word and sort them by frequency
    word_counts = Counter(words)
    most_common_words = [word for word, _ in word_counts.most_common()]
    # Join the most common words to form a coherent description based on frequency
    description = " ".join(most_common_words)

    return description
