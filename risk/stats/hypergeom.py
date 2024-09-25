"""
risk/stats/hypergeom
~~~~~~~~~~~~~~~~~~~~
"""

from typing import Any, Dict

import numpy as np
from scipy.stats import hypergeom


def compute_hypergeom_test(
    neighborhoods: np.ndarray,
    annotations: np.ndarray,
) -> Dict[str, Any]:
    """Compute hypergeometric test for enrichment and depletion in neighborhoods.

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
              of observing more annotations in a neighborhood than expected under the hypergeometric test.
            - "depletion_pvals" (np.ndarray): P-values for depletion, indicating the probability
              of observing fewer annotations in a neighborhood than expected under the hypergeometric test.
    """
    # Ensure both matrices are binary (presence/absence)
    neighborhoods = (neighborhoods > 0).astype(int)
    annotations = (annotations > 0).astype(int)
    total_node_count = annotations.shape[0]
    # Sum of values in each neighborhood
    neighborhood_sums = np.sum(neighborhoods, axis=0)[:, np.newaxis]
    # Repeating neighborhood sums for each annotation
    neighborhood_size_matrix = np.tile(neighborhood_sums, (1, annotations.shape[1]))
    # Total number of nodes annotated to each attribute
    annotated_node_counts = np.tile(np.sum(annotations, axis=0), (neighborhoods.shape[1], 1))
    # Nodes in each neighborhood annotated to each attribute
    annotated_in_neighborhood = np.dot(neighborhoods, annotations)
    # Calculate p-values using the hypergeometric distribution
    depletion_pvals = hypergeom.cdf(
        annotated_in_neighborhood, total_node_count, annotated_node_counts, neighborhood_size_matrix
    )
    enrichment_pvals = hypergeom.sf(
        annotated_in_neighborhood - 1,
        total_node_count,
        annotated_node_counts,
        neighborhood_size_matrix,
    )
    return {"depletion_pvals": depletion_pvals, "enrichment_pvals": enrichment_pvals}
