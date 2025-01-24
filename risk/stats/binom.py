"""
risk/stats/binomial
~~~~~~~~~~~~~~~~~~~
"""

from typing import Any, Dict

import numpy as np
from scipy.stats import binom


def compute_binom_test(
    neighborhoods: np.ndarray, annotations: np.ndarray, null_distribution: str = "network"
) -> Dict[str, Any]:
    """Compute Binomial test for enrichment and depletion in neighborhoods with selectable null distribution.

    Args:
        neighborhoods (np.ndarray): Binary matrix representing neighborhoods (rows as nodes, columns as neighbors).
        annotations (np.ndarray): Binary matrix representing annotations (rows as nodes, columns as annotations).
        null_distribution (str, optional): Type of null distribution ('network' or 'annotations'). Defaults to "network".

    Returns:
        Dict[str, Any]: Dictionary containing depletion and enrichment p-values.
    """
    # Calculate the total counts of annotated nodes and neighborhood sizes
    annotated_counts = neighborhoods @ annotations
    neighborhood_sizes = neighborhoods.sum(axis=1, keepdims=True)
    annotation_totals = annotations.sum(axis=0, keepdims=True)
    total_nodes = neighborhoods.shape[1]  # Total number of nodes in the network

    # Compute p for the Binomial distribution based on the chosen null distribution
    if null_distribution == "network":
        p_values = (
            annotation_totals / total_nodes
        )  # Probability of annotation per node across the network
    elif null_distribution == "annotations":
        p_values = annotation_totals / annotations.sum()  # Probability weighted by annotations
    else:
        raise ValueError(
            "Invalid null_distribution value. Choose either 'network' or 'annotations'."
        )

    # Compute enrichment and depletion p-values
    enrichment_pvals = 1 - binom.cdf(annotated_counts - 1, neighborhood_sizes, p_values)
    depletion_pvals = binom.cdf(annotated_counts, neighborhood_sizes, p_values)

    return {"enrichment_pvals": enrichment_pvals, "depletion_pvals": depletion_pvals}
