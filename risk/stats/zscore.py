"""
risk/stats/zscore
~~~~~~~~~~~~~~~~~~
"""

from typing import Any, Dict

import numpy as np
from scipy.sparse import csr_matrix
from scipy.stats import norm


def compute_zscore_test(
    neighborhoods: csr_matrix,
    annotations: csr_matrix,
    null_distribution: str = "network",
) -> Dict[str, Any]:
    """
    Compute Z-score test for enrichment and depletion in neighborhoods with selectable null distribution.

    Args:
        neighborhoods (csr_matrix): Sparse binary matrix representing neighborhoods.
        annotations (csr_matrix): Sparse binary matrix representing annotations.
        null_distribution (str, optional): Type of null distribution ('network' or 'annotations'). Defaults to "network".

    Returns:
        Dict[str, Any]: Dictionary containing depletion and enrichment p-values.
    """
    # Total number of nodes in the network
    total_node_count = neighborhoods.shape[1]

    # Compute sums
    if null_distribution == "network":
        background_population = total_node_count
        neighborhood_sums = neighborhoods.sum(axis=0).A.flatten()  # Dense column sums
        annotation_sums = annotations.sum(axis=0).A.flatten()  # Dense row sums
    elif null_distribution == "annotations":
        annotated_nodes = annotations.sum(axis=1).A.flatten() > 0  # Dense boolean mask
        background_population = annotated_nodes.sum()
        neighborhood_sums = neighborhoods[annotated_nodes].sum(axis=0).A.flatten()
        annotation_sums = annotations[annotated_nodes].sum(axis=0).A.flatten()
    else:
        raise ValueError(
            "Invalid null_distribution value. Choose either 'network' or 'annotations'."
        )

    # Observed values
    observed = (neighborhoods.T @ annotations).toarray()  # Convert sparse result to dense
    # Expected values under the null
    neighborhood_sums = neighborhood_sums.reshape(-1, 1)  # Ensure correct shape
    annotation_sums = annotation_sums.reshape(1, -1)  # Ensure correct shape
    expected = (neighborhood_sums @ annotation_sums) / background_population

    # Standard deviation under the null
    std_dev = np.sqrt(
        expected
        * (1 - annotation_sums / background_population)
        * (1 - neighborhood_sums / background_population)
    )
    std_dev[std_dev == 0] = np.nan  # Avoid division by zero
    # Compute Z-scores
    z_scores = (observed - expected) / std_dev

    # Convert Z-scores to depletion and enrichment p-values
    enrichment_pvals = norm.sf(z_scores)  # Upper tail
    depletion_pvals = norm.cdf(z_scores)  # Lower tail

    return {"depletion_pvals": depletion_pvals, "enrichment_pvals": enrichment_pvals}
