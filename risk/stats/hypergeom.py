"""
risk/stats/hypergeom
~~~~~~~~~~~~~~~~~~~~
"""

from typing import Any, Dict

import numpy as np
from scipy.sparse import csr_matrix
from scipy.stats import hypergeom


def compute_hypergeom_test(
    neighborhoods: csr_matrix,
    annotations: csr_matrix,
    null_distribution: str = "network",
) -> Dict[str, Any]:
    """
    Compute hypergeometric test for enrichment and depletion in neighborhoods with selectable null distribution.

    Args:
        neighborhoods (csr_matrix): Sparse binary matrix representing neighborhoods.
        annotations (csr_matrix): Sparse binary matrix representing annotations.
        null_distribution (str, optional): Type of null distribution ('network' or 'annotations'). Defaults to "network".

    Returns:
        Dict[str, Any]: Dictionary containing depletion and enrichment p-values.
    """
    # Get the total number of nodes in the network
    total_nodes = neighborhoods.shape[1]

    # Compute sums
    neighborhood_sums = neighborhoods.sum(axis=0).A.flatten()  # Convert to dense array
    annotation_sums = annotations.sum(axis=0).A.flatten()  # Convert to dense array

    if null_distribution == "network":
        background_population = total_nodes
    elif null_distribution == "annotations":
        annotated_nodes = annotations.sum(axis=1).A.flatten() > 0  # Boolean mask
        background_population = annotated_nodes.sum()
        neighborhood_sums = neighborhoods[annotated_nodes].sum(axis=0).A.flatten()
        annotation_sums = annotations[annotated_nodes].sum(axis=0).A.flatten()
    else:
        raise ValueError(
            "Invalid null_distribution value. Choose either 'network' or 'annotations'."
        )

    # Observed counts
    annotated_in_neighborhood = neighborhoods.T @ annotations  # Sparse result
    annotated_in_neighborhood = annotated_in_neighborhood.toarray()  # Convert to dense
    # Align shapes for broadcasting
    neighborhood_sums = neighborhood_sums.reshape(-1, 1)
    annotation_sums = annotation_sums.reshape(1, -1)
    background_population = np.array(background_population).reshape(1, 1)

    # Compute hypergeometric p-values
    depletion_pvals = hypergeom.cdf(
        annotated_in_neighborhood, background_population, annotation_sums, neighborhood_sums
    )
    enrichment_pvals = hypergeom.sf(
        annotated_in_neighborhood - 1, background_population, annotation_sums, neighborhood_sums
    )

    return {"depletion_pvals": depletion_pvals, "enrichment_pvals": enrichment_pvals}
