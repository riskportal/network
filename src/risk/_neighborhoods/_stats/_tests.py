"""
risk/_neighborhoods/_stats/_tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from typing import Any, Dict

import numpy as np
from scipy.sparse import csr_matrix
from scipy.stats import binom, chi2, hypergeom, norm, poisson


def compute_binom_test(
    neighborhoods: csr_matrix,
    annotation: csr_matrix,
    null_distribution: str = "network",
) -> Dict[str, Any]:
    """
    Compute Binomial test for enrichment and depletion in neighborhoods with selectable null distribution.

    Args:
        neighborhoods (csr_matrix): Sparse binary matrix representing neighborhoods.
        annotation (csr_matrix): Sparse binary matrix representing annotation.
        null_distribution (str, optional): Type of null distribution ('network' or 'annotation'). Defaults to "network".

    Returns:
        Dict[str, Any]: Dictionary containing depletion and enrichment p-values.

    Raises:
        ValueError: If an invalid null_distribution value is provided.
    """
    # Get the total number of nodes in the network
    total_nodes = neighborhoods.shape[1]

    # Compute sums (remain sparse here)
    neighborhood_sizes = neighborhoods.sum(axis=1)  # Row sums
    annotation_totals = annotation.sum(axis=0)  # Column sums
    # Compute probabilities (convert to dense)
    if null_distribution == "network":
        p_values = (annotation_totals / total_nodes).A.flatten()  # Dense 1D array
    elif null_distribution == "annotation":
        p_values = (annotation_totals / annotation.sum()).A.flatten()  # Dense 1D array
    else:
        raise ValueError(
            "Invalid null_distribution value. Choose either 'network' or 'annotation'."
        )

    # Observed counts (sparse matrix multiplication)
    annotated_counts = neighborhoods @ annotation  # Sparse result
    annotated_counts_dense = annotated_counts.toarray()  # Convert for dense operations

    # Compute enrichment and depletion p-values
    enrichment_pvals = 1 - binom.cdf(annotated_counts_dense - 1, neighborhood_sizes.A, p_values)
    depletion_pvals = binom.cdf(annotated_counts_dense, neighborhood_sizes.A, p_values)

    return {"enrichment_pvals": enrichment_pvals, "depletion_pvals": depletion_pvals}


def compute_chi2_test(
    neighborhoods: csr_matrix,
    annotation: csr_matrix,
    null_distribution: str = "network",
) -> Dict[str, Any]:
    """
    Compute chi-squared test for enrichment and depletion in neighborhoods with selectable null distribution.

    Args:
        neighborhoods (csr_matrix): Sparse binary matrix representing neighborhoods.
        annotation (csr_matrix): Sparse binary matrix representing annotation.
        null_distribution (str, optional): Type of null distribution ('network' or 'annotation'). Defaults to "network".

    Returns:
        Dict[str, Any]: Dictionary containing depletion and enrichment p-values.

    Raises:
        ValueError: If an invalid null_distribution value is provided.
    """
    # Total number of nodes in the network
    total_node_count = neighborhoods.shape[0]

    if null_distribution == "network":
        # Case 1: Use all nodes as the background
        background_population = total_node_count
        neighborhood_sums = neighborhoods.sum(axis=0)  # Column sums of neighborhoods
        annotation_sums = annotation.sum(axis=0)  # Column sums of annotations
    elif null_distribution == "annotation":
        # Case 2: Only consider nodes with at least one annotation
        annotated_nodes = (
            np.ravel(annotation.sum(axis=1)) > 0
        )  # Row-wise sum to filter nodes with annotations
        background_population = annotated_nodes.sum()  # Total number of annotated nodes
        neighborhood_sums = neighborhoods[annotated_nodes].sum(
            axis=0
        )  # Neighborhood sums for annotated nodes
        annotation_sums = annotation[annotated_nodes].sum(
            axis=0
        )  # Annotation sums for annotated nodes
    else:
        raise ValueError(
            "Invalid null_distribution value. Choose either 'network' or 'annotation'."
        )

    # Convert to dense arrays for downstream computations
    neighborhood_sums = np.asarray(neighborhood_sums).reshape(-1, 1)  # Ensure column vector shape
    annotation_sums = np.asarray(annotation_sums).reshape(1, -1)  # Ensure row vector shape

    # Observed values: number of annotated nodes in each neighborhood
    observed = neighborhoods.T @ annotation  # Shape: (neighborhoods, annotation)
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
    annotation: csr_matrix,
    null_distribution: str = "network",
) -> Dict[str, Any]:
    """
    Compute hypergeometric test for enrichment and depletion in neighborhoods with selectable null distribution.

    Args:
        neighborhoods (csr_matrix): Sparse binary matrix representing neighborhoods.
        annotation (csr_matrix): Sparse binary matrix representing annotation.
        null_distribution (str, optional): Type of null distribution ('network' or 'annotation'). Defaults to "network".

    Returns:
        Dict[str, Any]: Dictionary containing depletion and enrichment p-values.

    Raises:
        ValueError: If an invalid null_distribution value is provided.
    """
    # Get the total number of nodes in the network
    total_nodes = neighborhoods.shape[1]

    # Compute sums
    neighborhood_sums = neighborhoods.sum(axis=0).A.flatten()  # Convert to dense array
    annotation_sums = annotation.sum(axis=0).A.flatten()  # Convert to dense array

    if null_distribution == "network":
        background_population = total_nodes
    elif null_distribution == "annotation":
        annotated_nodes = annotation.sum(axis=1).A.flatten() > 0  # Boolean mask
        background_population = annotated_nodes.sum()
        neighborhood_sums = neighborhoods[annotated_nodes].sum(axis=0).A.flatten()
        annotation_sums = annotation[annotated_nodes].sum(axis=0).A.flatten()
    else:
        raise ValueError(
            "Invalid null_distribution value. Choose either 'network' or 'annotation'."
        )

    # Observed counts
    annotated_in_neighborhood = neighborhoods.T @ annotation  # Sparse result
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


def compute_poisson_test(
    neighborhoods: csr_matrix,
    annotation: csr_matrix,
    null_distribution: str = "network",
) -> Dict[str, Any]:
    """
    Compute Poisson test for enrichment and depletion in neighborhoods with selectable null distribution.

    Args:
        neighborhoods (csr_matrix): Sparse binary matrix representing neighborhoods.
        annotation (csr_matrix): Sparse binary matrix representing annotation.
        null_distribution (str, optional): Type of null distribution ('network' or 'annotation'). Defaults to "network".

    Returns:
        Dict[str, Any]: Dictionary containing depletion and enrichment p-values.

    Raises:
        ValueError: If an invalid null_distribution value is provided.
    """
    # Matrix multiplication to get the number of annotated nodes in each neighborhood
    annotated_in_neighborhood = neighborhoods @ annotation  # Sparse result
    # Convert annotated counts to dense for downstream calculations
    annotated_in_neighborhood_dense = annotated_in_neighborhood.toarray()

    # Compute lambda_expected based on the chosen null distribution
    if null_distribution == "network":
        # Use the mean across neighborhoods (axis=1)
        lambda_expected = np.mean(annotated_in_neighborhood_dense, axis=1, keepdims=True)
    elif null_distribution == "annotation":
        # Use the mean across annotations (axis=0)
        lambda_expected = np.mean(annotated_in_neighborhood_dense, axis=0, keepdims=True)
    else:
        raise ValueError(
            "Invalid null_distribution value. Choose either 'network' or 'annotation'."
        )

    # Compute p-values for enrichment and depletion using Poisson distribution
    enrichment_pvals = 1 - poisson.cdf(annotated_in_neighborhood_dense - 1, lambda_expected)
    depletion_pvals = poisson.cdf(annotated_in_neighborhood_dense, lambda_expected)

    return {"enrichment_pvals": enrichment_pvals, "depletion_pvals": depletion_pvals}


def compute_zscore_test(
    neighborhoods: csr_matrix,
    annotation: csr_matrix,
    null_distribution: str = "network",
) -> Dict[str, Any]:
    """
    Compute z-score test for enrichment and depletion in neighborhoods with selectable null distribution.

    Args:
        neighborhoods (csr_matrix): Sparse binary matrix representing neighborhoods.
        annotation (csr_matrix): Sparse binary matrix representing annotation.
        null_distribution (str, optional): Type of null distribution ('network' or 'annotation'). Defaults to "network".

    Returns:
        Dict[str, Any]: Dictionary containing depletion and enrichment p-values.

    Raises:
        ValueError: If an invalid null_distribution value is provided.
    """
    # Total number of nodes in the network
    total_node_count = neighborhoods.shape[1]

    # Compute sums
    if null_distribution == "network":
        background_population = total_node_count
        neighborhood_sums = neighborhoods.sum(axis=0).A.flatten()  # Dense column sums
        annotation_sums = annotation.sum(axis=0).A.flatten()  # Dense row sums
    elif null_distribution == "annotation":
        annotated_nodes = annotation.sum(axis=1).A.flatten() > 0  # Dense boolean mask
        background_population = annotated_nodes.sum()
        neighborhood_sums = neighborhoods[annotated_nodes].sum(axis=0).A.flatten()
        annotation_sums = annotation[annotated_nodes].sum(axis=0).A.flatten()
    else:
        raise ValueError(
            "Invalid null_distribution value. Choose either 'network' or 'annotation'."
        )

    # Observed values
    observed = (neighborhoods.T @ annotation).toarray()  # Convert sparse result to dense
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
    # Compute z-scores
    z_scores = (observed - expected) / std_dev

    # Convert z-scores to depletion and enrichment p-values
    enrichment_pvals = norm.sf(z_scores)  # Upper tail
    depletion_pvals = norm.cdf(z_scores)  # Lower tail

    return {"depletion_pvals": depletion_pvals, "enrichment_pvals": enrichment_pvals}
