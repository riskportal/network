"""
risk/stats/zscore
~~~~~~~~~~~~~~~~~~
"""

from typing import Any, Dict

import numpy as np
from scipy.stats import norm


def compute_zscore_test(
    neighborhoods: np.ndarray, annotations: np.ndarray, null_distribution: str = "network"
) -> Dict[str, Any]:
    """Compute Z-score test for enrichment and depletion in neighborhoods with selectable null distribution.

    Args:
        neighborhoods (np.ndarray): Binary matrix representing neighborhoods.
        annotations (np.ndarray): Binary matrix representing annotations.
        null_distribution (str, optional): Type of null distribution ('network' or 'annotations'). Defaults to "network".

    Returns:
        Dict[str, Any]: Dictionary containing depletion and enrichment p-values.
    """
    # Get the total number of nodes in the network
    total_node_count = neighborhoods.shape[0]

    if null_distribution == "network":
        # Case 1: Use all nodes as the background
        background_population = total_node_count
        neighborhood_sums = np.sum(neighborhoods, axis=0, keepdims=True).T
        annotation_sums = np.sum(annotations, axis=0, keepdims=True)
    elif null_distribution == "annotations":
        # Case 2: Only consider nodes with at least one annotation
        annotated_nodes = np.sum(annotations, axis=1) > 0
        background_population = np.sum(annotated_nodes)
        neighborhood_sums = np.sum(neighborhoods[annotated_nodes], axis=0, keepdims=True).T
        annotation_sums = np.sum(annotations[annotated_nodes], axis=0, keepdims=True)
    else:
        raise ValueError(
            "Invalid null_distribution value. Choose either 'network' or 'annotations'."
        )

    # Matrix multiplication for annotated nodes in each neighborhood
    observed = neighborhoods.T @ annotations
    # Compute expected values under the null distribution
    expected = (neighborhood_sums @ annotation_sums) / background_population
    # Compute standard deviation under the null distribution
    std_dev = np.sqrt(
        expected
        * (1 - annotation_sums / background_population)
        * (1 - neighborhood_sums / background_population)
    )
    # Avoid division by zero
    std_dev[std_dev == 0] = np.nan  # Mark invalid computations
    # Compute Z-scores
    z_scores = (observed - expected) / std_dev
    # Convert Z-scores to depletion and enrichment p-values
    enrichment_pvals = norm.sf(z_scores)  # Upper tail
    depletion_pvals = norm.cdf(z_scores)  # Lower tail

    return {"depletion_pvals": depletion_pvals, "enrichment_pvals": enrichment_pvals}
