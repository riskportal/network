"""
risk/stats/chi2
~~~~~~~~~~~~~~~
"""

from typing import Any, Dict

import numpy as np
from scipy.stats import chi2


def compute_chi2_test(
    neighborhoods: np.ndarray, annotations: np.ndarray, null_distribution: str = "network"
) -> Dict[str, Any]:
    """Compute chi-squared test for enrichment and depletion in neighborhoods with selectable null distribution.

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
        neighborhood_sums = np.sum(
            neighborhoods, axis=0, keepdims=True
        ).T  # Column sums of neighborhoods
        annotation_sums = np.sum(annotations, axis=0, keepdims=True)  # Column sums of annotations
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

    # Observed values: number of annotated nodes in each neighborhood
    observed = neighborhoods.T @ annotations  # Shape: (neighborhoods, annotations)
    # Expected values under the null
    expected = (neighborhood_sums @ annotation_sums) / background_population
    # Chi-squared statistic: sum((observed - expected)^2 / expected)
    with np.errstate(divide="ignore", invalid="ignore"):  # Handle divide-by-zero
        chi2_stat = np.where(expected > 0, (observed - expected) ** 2 / expected, 0)

    # Compute p-values for enrichment (upper tail) and depletion (lower tail)
    enrichment_pvals = chi2.sf(chi2_stat, df=1)  # Survival function for upper tail
    depletion_pvals = chi2.cdf(chi2_stat, df=1)  # Cumulative distribution for lower tail

    return {"depletion_pvals": depletion_pvals, "enrichment_pvals": enrichment_pvals}
