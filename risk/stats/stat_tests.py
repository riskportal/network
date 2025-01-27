"""
risk/stats/stat_tests
~~~~~~~~~~~~~~~~~~~~~
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict

import numpy as np
from scipy.sparse import csr_matrix
from scipy.stats import binom
from scipy.stats import chi2
from scipy.stats import hypergeom
from scipy.stats import norm
from scipy.stats import poisson


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


def compute_hypergeom_test(
    neighborhoods: csr_matrix,
    annotations: csr_matrix,
    null_distribution: str = "network",
) -> Dict[str, Any]:
    """Compute hypergeometric test for enrichment and depletion in neighborhoods with selectable null distribution.

    Args:
        neighborhoods (csr_matrix): Sparse binary matrix representing neighborhoods.
        annotations (csr_matrix): Sparse binary matrix representing annotations.
        null_distribution (str, optional): Type of null distribution ('network' or 'annotations').
            Defaults to "network".

    Returns:
        Dict[str, Any]: Dictionary containing depletion and enrichment p-values.
    """
    # Get the total number of nodes in the network
    total_nodes = neighborhoods.shape[1]
    # Calculate neighborhood and annotation sums
    neighborhood_sums = neighborhoods.sum(axis=0).A1  # A1 returns a 1D array
    annotation_sums = annotations.sum(axis=0).A1

    if null_distribution == "annotations":
        annotated_nodes = annotations.getnnz(axis=1) > 0  # Nodes with any annotation
        background_population = annotated_nodes.sum()
        neighborhoods = neighborhoods[annotated_nodes]
        annotations = annotations[annotated_nodes]
        neighborhood_sums = neighborhoods.sum(axis=0).A1
        annotation_sums = annotations.sum(axis=0).A1
    elif null_distribution == "network":
        background_population = total_nodes
    else:
        raise ValueError("Invalid null_distribution value. Choose 'network' or 'annotations'.")

    # Sparse matrix multiplication for observed counts
    annotated_in_neighborhood = neighborhoods.T @ annotations  # Result is sparse

    def compute_pvals_for_column(col_idx: int):
        """Compute depletion and enrichment p-values for a single annotation column."""
        observed_counts = annotated_in_neighborhood[:, col_idx].toarray().flatten()
        ann_total = annotation_sums[col_idx]
        # Compute depletion and enrichment p-values
        depletion_pvals = hypergeom.cdf(
            observed_counts, background_population, ann_total, neighborhood_sums
        )
        enrichment_pvals = hypergeom.sf(
            observed_counts - 1, background_population, ann_total, neighborhood_sums
        )
        return col_idx, depletion_pvals, enrichment_pvals

    # Use ThreadPoolExecutor to process columns in parallel
    num_columns = annotations.shape[1]
    depletion_pvals = np.empty((annotated_in_neighborhood.shape[0], num_columns), dtype=np.float64)
    enrichment_pvals = np.empty((annotated_in_neighborhood.shape[0], num_columns), dtype=np.float64)
    with ThreadPoolExecutor() as executor:
        results = executor.map(compute_pvals_for_column, range(num_columns))

    # Collect results
    for col_idx, dep_pval, enr_pval in results:
        depletion_pvals[:, col_idx] = dep_pval
        enrichment_pvals[:, col_idx] = enr_pval

    return {"depletion_pvals": depletion_pvals, "enrichment_pvals": enrichment_pvals}


def compute_poisson_test(
    neighborhoods: csr_matrix,
    annotations: csr_matrix,
    null_distribution: str = "network",
) -> Dict[str, Any]:
    """
    Compute Poisson test for enrichment and depletion in neighborhoods with selectable null distribution.

    Args:
        neighborhoods (csr_matrix): Sparse binary matrix representing neighborhoods.
        annotations (csr_matrix): Sparse binary matrix representing annotations.
        null_distribution (str, optional): Type of null distribution ('network' or 'annotations'). Defaults to "network".

    Returns:
        Dict[str, Any]: Dictionary containing depletion and enrichment p-values.
    """
    # Matrix multiplication to get the number of annotated nodes in each neighborhood
    annotated_in_neighborhood = neighborhoods @ annotations  # Sparse result
    # Convert annotated counts to dense for downstream calculations
    annotated_in_neighborhood_dense = annotated_in_neighborhood.toarray()

    # Compute lambda_expected based on the chosen null distribution
    if null_distribution == "network":
        # Use the mean across neighborhoods (axis=1)
        lambda_expected = np.mean(annotated_in_neighborhood_dense, axis=1, keepdims=True)
    elif null_distribution == "annotations":
        # Use the mean across annotations (axis=0)
        lambda_expected = np.mean(annotated_in_neighborhood_dense, axis=0, keepdims=True)
    else:
        raise ValueError(
            "Invalid null_distribution value. Choose either 'network' or 'annotations'."
        )

    # Compute p-values for enrichment and depletion using Poisson distribution
    enrichment_pvals = 1 - poisson.cdf(annotated_in_neighborhood_dense - 1, lambda_expected)
    depletion_pvals = poisson.cdf(annotated_in_neighborhood_dense, lambda_expected)

    return {"enrichment_pvals": enrichment_pvals, "depletion_pvals": depletion_pvals}


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
