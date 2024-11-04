"""
risk/annotations/annotations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import re
from collections import Counter
from itertools import compress
from typing import Any, Dict, List, Set

import networkx as nx
import nltk
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from risk.log import logger


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
# Initialize English stopwords
stop_words = set(stopwords.words("english"))


def load_annotations(
    network: nx.Graph, annotations_input: Dict[str, Any], min_nodes_per_term: int = 2
) -> Dict[str, Any]:
    """Convert annotations input to a DataFrame and reindex based on the network's node labels.

    Args:
        network (nx.Graph): The network graph.
        annotations_input (Dict[str, Any]): A dictionary with annotations.
        min_nodes_per_term (int, optional): The minimum number of network nodes required for each annotation
            term to be included. Defaults to 2.

    Returns:
        Dict[str, Any]: A dictionary containing ordered nodes, ordered annotations, and the binary annotations matrix.

    Raises:
        ValueError: If no annotations are found for the nodes in the network.
        ValueError: If no annotations have at least min_nodes_per_term nodes in the network.
    """
    # Flatten the dictionary to a list of tuples for easier DataFrame creation
    flattened_annotations = [
        (node, annotation) for annotation, nodes in annotations_input.items() for node in nodes
    ]
    # Create a DataFrame from the flattened list
    annotations = pd.DataFrame(flattened_annotations, columns=["node", "annotations"])
    annotations["is_member"] = 1
    # Pivot to create a binary matrix with nodes as rows and annotations as columns
    annotations_pivot = annotations.pivot_table(
        index="node", columns="annotations", values="is_member", fill_value=0, dropna=False
    )
    # Reindex the annotations matrix based on the node labels from the network
    node_label_order = list(nx.get_node_attributes(network, "label").values())
    annotations_pivot = annotations_pivot.reindex(index=node_label_order)
    # Raise an error if no valid annotations are found for the nodes in the network
    if annotations_pivot.notnull().sum().sum() == 0:
        raise ValueError("No terms found in the annotation file for the nodes in the network.")

    # Filter out annotations with fewer than min_nodes_per_term occurrences
    # This assists in reducing noise and focusing on more relevant annotations for statistical analysis
    num_terms_before_filtering = annotations_pivot.shape[1]
    annotations_pivot = annotations_pivot.loc[
        :, (annotations_pivot.sum(axis=0) >= min_nodes_per_term)
    ]
    num_terms_after_filtering = annotations_pivot.shape[1]
    # Log the number of annotations before and after filtering
    logger.info(f"Minimum number of nodes per annotation term: {min_nodes_per_term}")
    logger.info(f"Number of input annotation terms: {num_terms_before_filtering}")
    logger.info(f"Number of remaining annotation terms: {num_terms_after_filtering}")
    if num_terms_after_filtering == 0:
        raise ValueError(
            f"No annotation terms found with at least {min_nodes_per_term} nodes in the network."
        )

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
    neighborhood_significance_sums: List[int],
    significant_significance_matrix: np.ndarray,
    significant_binary_significance_matrix: np.ndarray,
    min_cluster_size: int = 5,
    max_cluster_size: int = 1000,
) -> pd.DataFrame:
    """Define top annotations based on neighborhood significance sums and binary significance matrix.

    Args:
        network (NetworkX graph): The network graph.
        ordered_annotation_labels (list of str): List of ordered annotation labels.
        neighborhood_significance_sums (list of int): List of neighborhood significance sums.
        significant_significance_matrix (np.ndarray): Enrichment matrix below alpha threshold.
        significant_binary_significance_matrix (np.ndarray): Binary significance matrix below alpha threshold.
        min_cluster_size (int, optional): Minimum cluster size. Defaults to 5.
        max_cluster_size (int, optional): Maximum cluster size. Defaults to 1000.

    Returns:
        pd.DataFrame: DataFrame with top annotations and their properties.
    """
    # Sum the columns of the significant significance matrix (positive floating point values)
    significant_significance_scores = significant_significance_matrix.sum(axis=0)
    # Create DataFrame to store annotations, their neighborhood significance sums, and significance scores
    annotations_significance_matrix = pd.DataFrame(
        {
            "id": range(len(ordered_annotation_labels)),
            "full_terms": ordered_annotation_labels,
            "significant_neighborhood_significance_sums": neighborhood_significance_sums,
            "significant_significance_score": significant_significance_scores,
        }
    )
    annotations_significance_matrix["significant_annotations"] = False
    # Apply size constraints to identify potential significant annotations
    annotations_significance_matrix.loc[
        (
            annotations_significance_matrix["significant_neighborhood_significance_sums"]
            >= min_cluster_size
        )
        & (
            annotations_significance_matrix["significant_neighborhood_significance_sums"]
            <= max_cluster_size
        ),
        "significant_annotations",
    ] = True
    # Initialize columns for connected components analysis
    annotations_significance_matrix["num_connected_components"] = 0
    annotations_significance_matrix["size_connected_components"] = None
    annotations_significance_matrix["size_connected_components"] = annotations_significance_matrix[
        "size_connected_components"
    ].astype(object)
    annotations_significance_matrix["num_large_connected_components"] = 0

    for attribute in annotations_significance_matrix.index.values[
        annotations_significance_matrix["significant_annotations"]
    ]:
        # Identify significant neighborhoods based on the binary significance matrix
        significant_neighborhoods = list(
            compress(list(network), significant_binary_significance_matrix[:, attribute])
        )
        significant_network = nx.subgraph(network, significant_neighborhoods)
        # Analyze connected components within the significant subnetwork
        connected_components = sorted(
            nx.connected_components(significant_network), key=len, reverse=True
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
        annotations_significance_matrix.loc[attribute, "num_connected_components"] = (
            num_connected_components
        )
        # Filter out attributes with more than one connected component
        annotations_significance_matrix.loc[
            annotations_significance_matrix["num_connected_components"] > 1,
            "significant_annotations",
        ] = False
        # Assign the number of large connected components
        annotations_significance_matrix.loc[attribute, "num_large_connected_components"] = (
            num_large_connected_components
        )
        # Assign the size of connected components, ensuring it is always a list
        annotations_significance_matrix.at[attribute, "size_connected_components"] = (
            filtered_size_connected_components.tolist()
        )

    return annotations_significance_matrix


def get_weighted_description(words_column: pd.Series, scores_column: pd.Series) -> str:
    """Generate a weighted description from words and their corresponding scores,
    with support for stopwords filtering and improved weighting logic.

    Args:
        words_column (pd.Series): A pandas Series containing strings to process.
        scores_column (pd.Series): A pandas Series containing significance scores to weigh the terms.

    Returns:
        str: A coherent description formed from the most frequent and significant words, weighed by significance scores.
    """
    # Handle case where all scores are the same
    if scores_column.max() == scores_column.min():
        normalized_scores = pd.Series([1] * len(scores_column))
    else:
        # Normalize the significance scores to be between 0 and 1
        normalized_scores = (scores_column - scores_column.min()) / (
            scores_column.max() - scores_column.min()
        )

    # Combine words and normalized scores to create weighted words
    weighted_words = []
    for word, score in zip(words_column, normalized_scores):
        word = str(word)
        if word not in stop_words:  # Skip stopwords
            weight = max(1, int((0 if pd.isna(score) else score) * 10))
            weighted_words.extend([word] * weight)

    # Tokenize the weighted words, but preserve number-word patterns like '4-alpha'
    tokens = word_tokenize(" ".join(weighted_words))
    # Ensure we treat "4-alpha" or other "number-word" patterns as single tokens
    combined_tokens = []
    for token in tokens:
        # Match patterns like '4-alpha' or '5-hydroxy' and keep them together
        if re.match(r"^\d+-\w+", token):
            combined_tokens.append(token)
        elif token.replace(".", "", 1).isdigit():  # Handle pure numeric tokens
            # Ignore pure numbers as descriptions unless necessary
            continue
        else:
            combined_tokens.append(token)

    # Prevent descriptions like just '4' from being selected
    if len(combined_tokens) == 1 and combined_tokens[0].isdigit():
        return "N/A"  # Return "N/A" for cases where it's just a number

    # Simplify the word list and generate the description
    simplified_words = _simplify_word_list(combined_tokens)
    description = _generate_coherent_description(simplified_words)

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
        words (List): A list of words or numerical string values.

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
