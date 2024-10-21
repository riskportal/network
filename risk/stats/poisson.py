"""
risk/stats/poisson
~~~~~~~~~~~~~~~~~~
"""

from typing import Any, Dict

import numpy as np
from scipy.stats import poisson


def compute_poisson_test(
    neighborhoods: np.ndarray, annotations: np.ndarray, null_distribution: str = "network"
) -> Dict[str, Any]:
    """Compute Poisson test for enrichment and depletion in neighborhoods with selectable null distribution.

    Args:
        neighborhoods (np.ndarray): Binary matrix representing neighborhoods.
        annotations (np.ndarray): Binary matrix representing annotations.
        null_distribution (str, optional): Type of null distribution ('network' or 'annotations'). Defaults to "network".

    Returns:
        Dict[str, Any]: Dictionary containing depletion and enrichment p-values.
    """
    # Matrix multiplication to get the number of annotated nodes in each neighborhood
    annotated_in_neighborhood = neighborhoods @ annotations

    # Compute lambda_expected based on the chosen null distribution
    if null_distribution == "network":
        # Use the mean across neighborhoods (axis=1)
        lambda_expected = np.mean(annotated_in_neighborhood, axis=1, keepdims=True)
    elif null_distribution == "annotations":
        # Use the mean across annotations (axis=0)
        lambda_expected = np.mean(annotated_in_neighborhood, axis=0, keepdims=True)
    else:
        raise ValueError(
            "Invalid null_distribution value. Choose either 'network' or 'annotations'."
        )

    # Compute p-values for enrichment and depletion using Poisson distribution
    enrichment_pvals = 1 - poisson.cdf(annotated_in_neighborhood - 1, lambda_expected)
    depletion_pvals = poisson.cdf(annotated_in_neighborhood, lambda_expected)

    return {"enrichment_pvals": enrichment_pvals, "depletion_pvals": depletion_pvals}
