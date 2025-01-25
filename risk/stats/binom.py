"""
risk/stats/binomial
~~~~~~~~~~~~~~~~~~~
"""

from typing import Any, Dict

from scipy.sparse import csr_matrix
from scipy.stats import binom


def compute_binom_test(
    neighborhoods: csr_matrix,
    annotations: csr_matrix,
    null_distribution: str = "network",
) -> Dict[str, Any]:
    """Compute Binomial test for enrichment and depletion in neighborhoods with selectable null distribution.

    Args:
        neighborhoods (csr_matrix): Sparse binary matrix representing neighborhoods.
        annotations (csr_matrix): Sparse binary matrix representing annotations.
        null_distribution (str, optional): Type of null distribution ('network' or 'annotations'). Defaults to "network".

    Returns:
        Dict[str, Any]: Dictionary containing depletion and enrichment p-values.
    """
    # Get the total number of nodes in the network
    total_nodes = neighborhoods.shape[1]

    # Compute sums (remain sparse here)
    neighborhood_sizes = neighborhoods.sum(axis=1)  # Row sums
    annotation_totals = annotations.sum(axis=0)  # Column sums
    # Compute probabilities (convert to dense)
    if null_distribution == "network":
        p_values = (annotation_totals / total_nodes).A.flatten()  # Dense 1D array
    elif null_distribution == "annotations":
        p_values = (annotation_totals / annotations.sum()).A.flatten()  # Dense 1D array
    else:
        raise ValueError(
            "Invalid null_distribution value. Choose either 'network' or 'annotations'."
        )

    # Observed counts (sparse matrix multiplication)
    annotated_counts = neighborhoods @ annotations  # Sparse result
    annotated_counts_dense = annotated_counts.toarray()  # Convert for dense operations

    # Compute enrichment and depletion p-values
    enrichment_pvals = 1 - binom.cdf(annotated_counts_dense - 1, neighborhood_sizes.A, p_values)
    depletion_pvals = binom.cdf(annotated_counts_dense, neighborhood_sizes.A, p_values)

    return {"enrichment_pvals": enrichment_pvals, "depletion_pvals": depletion_pvals}
