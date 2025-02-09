"""
risk/annotations/annotations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import os
import re
import zipfile
from collections import Counter
from itertools import compress
from typing import Any, Dict, List, Set

import networkx as nx
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from risk.log import logger
from scipy.sparse import coo_matrix


def ensure_nltk_resource(resource: str) -> None:
    """Ensure the specified NLTK resource is available."""
    # Define the path to the resource within the NLTK data directory
    resource_path = f"corpora/{resource}"
    # Check if the resource is already available.
    try:
        nltk.data.find(resource_path)
        return
    except LookupError:
        print(f"Resource '{resource}' not found. Attempting to download...")

    # Download the resource.
    nltk.download(resource)
    # Check again after downloading.
    try:
        nltk.data.find(resource_path)
        return
    except LookupError:
        print(f"Resource '{resource}' still not found after download. Checking for a ZIP file...")

    # Look for a ZIP file in all known NLTK data directories.
    for data_path in nltk.data.path:
        zip_path = os.path.join(data_path, "corpora", f"{resource}.zip")
        if os.path.isfile(zip_path):
            print(f"Found ZIP file for '{resource}' at: {zip_path}")
            target_dir = os.path.join(data_path, "corpora")
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(path=target_dir)
            print(f"Unzipped '{resource}' successfully.")
            break  # Stop after unzipping the first found ZIP.

    # Final check: Try to load the resource one last time.
    try:
        nltk.data.find(resource_path)
        print(f"Resource '{resource}' is now available.")
    except LookupError:
        raise LookupError(f"Resource '{resource}' could not be found, downloaded, or unzipped.")


# Ensure the NLTK stopwords and WordNet resources are available
ensure_nltk_resource("stopwords")
ensure_nltk_resource("wordnet")
# Use NLTK's stopwords - load all languages
STOP_WORDS = set(word for lang in stopwords.fileids() for word in stopwords.words(lang))
# Initialize the WordNet lemmatizer, which is used for normalizing words
LEMMATIZER = WordNetLemmatizer()


def load_annotations(
    network: nx.Graph, annotations_input: Dict[str, Any], min_nodes_per_term: int = 2
) -> Dict[str, Any]:
    """Convert annotations input to a sparse matrix and reindex based on the network's node labels.

    Args:
        network (nx.Graph): The network graph.
        annotations_input (Dict[str, Any]): A dictionary with annotations.
        min_nodes_per_term (int, optional): The minimum number of network nodes required for each annotation
            term to be included. Defaults to 2.

    Returns:
        Dict[str, Any]: A dictionary containing ordered nodes, ordered annotations, and the sparse binary annotations
            matrix.

    Raises:
        ValueError: If no annotations are found for the nodes in the network.
        ValueError: If no annotations have at least min_nodes_per_term nodes in the network.
    """
    # Step 1: Map nodes and annotations to indices
    node_label_order = [attr["label"] for _, attr in network.nodes(data=True) if "label" in attr]
    node_to_idx = {node: i for i, node in enumerate(node_label_order)}
    annotation_to_idx = {annotation: i for i, annotation in enumerate(annotations_input)}
    # Step 2: Construct a sparse binary matrix directly
    row = []
    col = []
    data = []
    for annotation, nodes in annotations_input.items():
        for node in nodes:
            if node in node_to_idx and annotation in annotation_to_idx:
                row.append(node_to_idx[node])
                col.append(annotation_to_idx[annotation])
                data.append(1)

    # Create a sparse binary matrix
    num_nodes = len(node_to_idx)
    num_annotations = len(annotation_to_idx)
    annotations_pivot = coo_matrix((data, (row, col)), shape=(num_nodes, num_annotations)).tocsr()
    # Step 3: Filter out annotations with fewer than min_nodes_per_term occurrences
    valid_annotations = annotations_pivot.sum(axis=0).A1 >= min_nodes_per_term
    annotations_pivot = annotations_pivot[:, valid_annotations]
    # Step 4: Raise errors for empty matrices
    if annotations_pivot.nnz == 0:
        raise ValueError("No terms found in the annotation file for the nodes in the network.")

    num_remaining_annotations = annotations_pivot.shape[1]
    if num_remaining_annotations == 0:
        raise ValueError(
            f"No annotation terms found with at least {min_nodes_per_term} nodes in the network."
        )

    # Step 5: Extract ordered nodes and annotations
    ordered_nodes = tuple(node_label_order)
    ordered_annotations = tuple(
        annotation for annotation, is_valid in zip(annotation_to_idx, valid_annotations) if is_valid
    )

    # Log the filtering details
    logger.info(f"Minimum number of nodes per annotation term: {min_nodes_per_term}")
    logger.info(f"Number of input annotation terms: {num_annotations}")
    logger.info(f"Number of remaining annotation terms: {num_remaining_annotations}")

    return {
        "ordered_nodes": ordered_nodes,
        "ordered_annotations": ordered_annotations,
        "matrix": annotations_pivot,
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
    using improved weighting logic with normalization, lemmatization, and aggregation.

    Args:
        words_column (pd.Series): A pandas Series containing strings (phrases) to process.
        scores_column (pd.Series): A pandas Series containing significance scores to weigh the terms.

    Returns:
        str: A coherent description formed from the most frequent and significant words.
    """
    # Normalize significance scores to [0,1]. If all scores are identical, use 1.
    if scores_column.max() == scores_column.min():
        normalized_scores = pd.Series([1] * len(scores_column), index=scores_column.index)
    else:
        normalized_scores = (scores_column - scores_column.min()) / (
            scores_column.max() - scores_column.min()
        )

    # Accumulate weighted counts for each token (after cleaning and lemmatization)
    weighted_counts = {}
    for phrase, score in zip(words_column, normalized_scores):
        # Tokenize the phrase
        tokens = word_tokenize(str(phrase))
        # Determine the weight (scale factor; here multiplying normalized score by 10)
        weight = max(1, int((0 if pd.isna(score) else score) * 10))
        for token in tokens:
            # Clean token: lowercase and remove extraneous punctuation (but preserve intra-word hyphens)
            token_clean = re.sub(r"[^\w\-]", "", token).strip()
            if not token_clean:
                continue
            # Skip tokens that are pure numbers
            if token_clean.isdigit():
                continue
            # Skip stopwords
            if token_clean in STOP_WORDS:
                continue
            # Lemmatize the token to merge similar forms
            token_norm = LEMMATIZER.lemmatize(token_clean)
            weighted_counts[token_norm] = weighted_counts.get(token_norm, 0) + weight

    # Reconstruct a weighted token list by repeating each token by its aggregated count.
    weighted_words = []
    for token, count in weighted_counts.items():
        weighted_words.extend([token] * count)

    # Combine tokens that match number-word patterns (e.g. "4-alpha") and remove pure numeric tokens.
    combined_tokens = []
    for token in weighted_words:
        if re.match(r"^\d+-\w+", token):
            combined_tokens.append(token)
        elif token.replace(".", "", 1).isdigit():
            continue
        else:
            combined_tokens.append(token)

    # If the only token is numeric, return a default value.
    if len(combined_tokens) == 1 and combined_tokens[0].isdigit():
        return "N/A"

    # Simplify the token list to remove near-duplicates based on the Jaccard index.
    simplified_words = _simplify_word_list(combined_tokens)
    # Generate a coherent description from the simplified words.
    description = _generate_coherent_description(simplified_words)

    return description


def _simplify_word_list(words: List[str], threshold: float = 0.80) -> List[str]:
    """Filter out words that are too similar based on the Jaccard index,
    keeping the word with the higher aggregated count.

    Args:
        words (List[str]): The list of tokens to be filtered.
        threshold (float, optional): The similarity threshold for the Jaccard index. Defaults to 0.80.

    Returns:
        List[str]: A list of filtered words, where similar words are reduced to the most frequent one.
    """
    # Count the occurrences (which reflect the weighted importance)
    word_counts = Counter(words)
    filtered_words = []
    used_words = set()

    # Iterate through words sorted by descending weighted frequency
    for word in sorted(word_counts, key=lambda w: word_counts[w], reverse=True):
        if word in used_words:
            continue

        word_set = set(word)
        # Find similar words (including the current word) based on the Jaccard index
        similar_words = [
            other_word
            for other_word in word_counts
            if _calculate_jaccard_index(word_set, set(other_word)) >= threshold
        ]
        # Choose the word with the highest weighted count among the similar group
        similar_words.sort(key=lambda w: word_counts[w], reverse=True)
        best_word = similar_words[0]
        filtered_words.append(best_word)
        used_words.update(similar_words)

    # Preserve the original order (by frequency) from the filtered set
    final_words = [word for word in words if word in filtered_words]

    return final_words


def _calculate_jaccard_index(set1: Set[Any], set2: Set[Any]) -> float:
    """Calculate the Jaccard index between two sets.

    Args:
        set1 (Set[Any]): The first set.
        set2 (Set[Any]): The second set.

    Returns:
        float: The Jaccard index (intersection over union). Returns 0 if the union is empty.
    """
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union else 0


def _generate_coherent_description(words: List[str]) -> str:
    """Generate a coherent description from a list of words.

    If there is only one unique entry, return it directly.
    Otherwise, order the words by frequency and join them into a single string.

    Args:
        words (List[str]): A list of tokens.

    Returns:
        str: A coherent, space-separated description.
    """
    if not words:
        return "N/A"

    # If there is only one unique word, return it directly
    unique_words = set(words)
    if len(unique_words) == 1:
        return list(unique_words)[0]

    # Count weighted occurrences and sort in descending order.
    word_counts = Counter(words)
    most_common_words = [word for word, _ in word_counts.most_common()]
    description = " ".join(most_common_words)

    return description
