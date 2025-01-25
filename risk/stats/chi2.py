"""
risk/stats/chi2
~~~~~~~~~~~~~~~
"""

from typing import Any, Dict

import numpy as np
from scipy.sparse import csr_matrix
from scipy.stats import chi2


def compute_chi2_test(
    neighborhoods: csr_matrix,
    annotations: csr_matrix,
    null_distribution: str = "network",
) -> Dict[str, Any]:
    """Compute chi-squared test for enrichment and depletion in neighborhoods with selectable null distribution.

    Args:
        neighborhoods (csr_matrix): Sparse binary matrix representing neighborhoods.
        annotations (csr_matrix): Sparse binary matrix representing annotations.
        null_distribution (str, optional): Type of null distribution ('network' or 'annotations'). Defaults to "network".

    Returns:
        Dict[str, Any]: Dictionary containing depletion and enrichment p-values.
    """
    # Total number of nodes in the network
    total_node_count = neighborhoods.shape[0]

    if null_distribution == "network":
        # Case 1: Use all nodes as the background
        background_population = total_node_count
        neighborhood_sums = neighborhoods.sum(axis=0)  # Column sums of neighborhoods
        annotation_sums = annotations.sum(axis=0)  # Column sums of annotations
    elif null_distribution == "annotations":
        # Case 2: Only consider nodes with at least one annotation
        annotated_nodes = (
            np.ravel(annotations.sum(axis=1)) > 0
        )  # Row-wise sum to filter nodes with annotations
        background_population = annotated_nodes.sum()  # Total number of annotated nodes
        neighborhood_sums = neighborhoods[annotated_nodes].sum(
            axis=0
        )  # Neighborhood sums for annotated nodes
        annotation_sums = annotations[annotated_nodes].sum(
            axis=0
        )  # Annotation sums for annotated nodes
    else:
        raise ValueError(
            "Invalid null_distribution value. Choose either 'network' or 'annotations'."
        )

    # Convert to dense arrays for downstream computations
    neighborhood_sums = np.asarray(neighborhood_sums).reshape(-1, 1)  # Ensure column vector shape
    annotation_sums = np.asarray(annotation_sums).reshape(1, -1)  # Ensure row vector shape

    # Observed values: number of annotated nodes in each neighborhood
    observed = neighborhoods.T @ annotations  # Shape: (neighborhoods, annotations)
    # Expected values under the null
    expected = (neighborhood_sums @ annotation_sums) / background_population
    # Chi-squared statistic: sum((observed - expected)^2 / expected)
    with np.errstate(divide="ignore", invalid="ignore"):  # Handle divide-by-zero
        chi2_stat = np.where(expected > 0, np.power(observed - expected, 2) / expected, 0)

    # Compute p-values for enrichment (upper tail) and depletion (lower tail)
    enrichment_pvals = chi2.sf(chi2_stat, df=1)  # Survival function for upper tail
    depletion_pvals = chi2.cdf(chi2_stat, df=1)  # Cumulative distribution for lower tail

    return {"depletion_pvals": depletion_pvals, "enrichment_pvals": enrichment_pvals}
