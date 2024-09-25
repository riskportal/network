"""
risk/stats/poisson
~~~~~~~~~~~~~~~~~~
"""

from typing import Dict, Any

import numpy as np
from scipy.stats import poisson


def compute_poisson_test(neighborhoods: np.ndarray, annotations: np.ndarray) -> Dict[str, Any]:
    """Compute Poisson test for enrichment and depletion in neighborhoods.

    Args:
        neighborhoods (np.ndarray): Binary matrix representing neighborhoods, where rows are nodes
            and columns are neighborhoods. Entries indicate the presence (1) or absence (0) of a node
            in a neighborhood.
        annotations (np.ndarray): Binary matrix representing annotations, where rows are nodes
            and columns are annotations. Entries indicate the presence (1) or absence (0) of a node
            being annotated.

    Returns:
        Dict[str, Any]: A dictionary with two keys:
            - "enrichment_pvals" (np.ndarray): P-values for enrichment, indicating the probability
              of observing more annotations in a neighborhood than expected under the Poisson distribution.
            - "depletion_pvals" (np.ndarray): P-values for depletion, indicating the probability of
              observing fewer annotations in a neighborhood than expected under the Poisson distribution.
    """
    neighborhoods = (neighborhoods > 0).astype(int)
    annotations = (annotations > 0).astype(int)
    annotated_in_neighborhood = np.dot(neighborhoods, annotations)
    lambda_expected = np.mean(annotated_in_neighborhood, axis=0)
    # Enrichment (observing more than expected)
    enrichment_pvals = 1 - poisson.cdf(annotated_in_neighborhood - 1, lambda_expected)

    # Depletion (observing fewer than expected)
    depletion_pvals = poisson.cdf(annotated_in_neighborhood, lambda_expected)

    return {"enrichment_pvals": enrichment_pvals, "depletion_pvals": depletion_pvals}
