"""
risk/stats/hypergeom
~~~~~~~~~~~~~~~~~~~~
"""

from typing import Any, Dict

import numpy as np
from scipy.stats import hypergeom


def compute_hypergeom_test(
    neighborhoods: np.ndarray, annotations: np.ndarray, null_distribution: str = "network"
) -> Dict[str, Any]:
    """Compute hypergeometric test for enrichment and depletion in neighborhoods with selectable null distribution.

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
    annotated_in_neighborhood = neighborhoods.T @ annotations
    # Calculate depletion and enrichment p-values using the hypergeometric distribution
    depletion_pvals = hypergeom.cdf(
        annotated_in_neighborhood, background_population, annotation_sums, neighborhood_sums
    )
    enrichment_pvals = hypergeom.sf(
        annotated_in_neighborhood - 1, background_population, annotation_sums, neighborhood_sums
    )

    return {"depletion_pvals": depletion_pvals, "enrichment_pvals": enrichment_pvals}
