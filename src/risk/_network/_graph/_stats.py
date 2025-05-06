"""
risk/_network/_graph/_stats
~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from typing import Any, Dict, Union

import numpy as np
from statsmodels.stats.multitest import fdrcorrection


def calculate_significance_matrices(
    depletion_pvals: np.ndarray,
    enrichment_pvals: np.ndarray,
    tail: str = "right",
    pval_cutoff: float = 0.05,
    fdr_cutoff: float = 0.05,
) -> Dict[str, Any]:
    """
    Calculate significance matrices based on p-values and specified tail.

    Args:
        depletion_pvals (np.ndarray): Matrix of depletion p-values.
        enrichment_pvals (np.ndarray): Matrix of enrichment p-values.
        tail (str, optional): The tail type for significance selection ('left', 'right', 'both'). Defaults to 'right'.
        pval_cutoff (float, optional): Cutoff for p-value significance. Defaults to 0.05.
        fdr_cutoff (float, optional): Cutoff for FDR significance if applied. Defaults to 0.05.

    Returns:
        Dict[str, Any]: Dictionary containing the enrichment matrix, binary significance matrix,
            and the matrix of significant enrichment values.
    """
    if fdr_cutoff < 1.0:
        # Apply FDR correction to depletion p-values
        depletion_qvals = np.apply_along_axis(fdrcorrection, 1, depletion_pvals)[:, 1, :]
        depletion_alpha_threshold_matrix = _compute_threshold_matrix(
            depletion_pvals, depletion_qvals, pval_cutoff=pval_cutoff, fdr_cutoff=fdr_cutoff
        )
        # Compute the depletion matrix using both q-values and p-values
        depletion_matrix = (depletion_qvals**2) * (depletion_pvals**0.5)

        # Apply FDR correction to enrichment p-values
        enrichment_qvals = np.apply_along_axis(fdrcorrection, 1, enrichment_pvals)[:, 1, :]
        enrichment_alpha_threshold_matrix = _compute_threshold_matrix(
            enrichment_pvals, enrichment_qvals, pval_cutoff=pval_cutoff, fdr_cutoff=fdr_cutoff
        )
        # Compute the enrichment matrix using both q-values and p-values
        enrichment_matrix = (enrichment_pvals**0.5) * (enrichment_qvals**2)
    else:
        # Compute threshold matrices based on p-value cutoffs only
        depletion_alpha_threshold_matrix = _compute_threshold_matrix(
            depletion_pvals, pval_cutoff=pval_cutoff
        )
        depletion_matrix = depletion_pvals

        enrichment_alpha_threshold_matrix = _compute_threshold_matrix(
            enrichment_pvals, pval_cutoff=pval_cutoff
        )
        enrichment_matrix = enrichment_pvals

    # Apply a negative log10 transformation for visualization purposes
    log_depletion_matrix = -np.log10(depletion_matrix)
    log_enrichment_matrix = -np.log10(enrichment_matrix)

    # Select the appropriate significance matrices based on the specified tail
    significance_matrix, significant_binary_significance_matrix = _select_significance_matrices(
        tail,
        log_depletion_matrix,
        depletion_alpha_threshold_matrix,
        log_enrichment_matrix,
        enrichment_alpha_threshold_matrix,
    )

    # Filter the enrichment matrix using the binary significance matrix
    significant_significance_matrix = np.where(
        significant_binary_significance_matrix == 1, significance_matrix, 0
    )

    return {
        "significance_matrix": significance_matrix,
        "significant_significance_matrix": significant_significance_matrix,
        "significant_binary_significance_matrix": significant_binary_significance_matrix,
    }


def _select_significance_matrices(
    tail: str,
    log_depletion_matrix: np.ndarray,
    depletion_alpha_threshold_matrix: np.ndarray,
    log_enrichment_matrix: np.ndarray,
    enrichment_alpha_threshold_matrix: np.ndarray,
) -> tuple:
    """
    Select significance matrices based on the specified tail type.

    Args:
        tail (str): The tail type for significance selection. Options are 'left', 'right', or 'both'.
        log_depletion_matrix (np.ndarray): Matrix of log-transformed depletion values.
        depletion_alpha_threshold_matrix (np.ndarray): Alpha threshold matrix for depletion significance.
        log_enrichment_matrix (np.ndarray): Matrix of log-transformed enrichment values.
        enrichment_alpha_threshold_matrix (np.ndarray): Alpha threshold matrix for enrichment significance.

    Returns:
        tuple: A tuple containing the selected enrichment matrix and binary significance matrix.

    Raises:
        ValueError: If the provided tail type is not 'left', 'right', or 'both'.
    """
    if tail not in {"left", "right", "both"}:
        raise ValueError("Invalid value for 'tail'. Must be 'left', 'right', or 'both'.")

    if tail == "left":
        # Select depletion matrix and corresponding alpha threshold for left-tail analysis
        significance_matrix = -log_depletion_matrix
        alpha_threshold_matrix = depletion_alpha_threshold_matrix
    elif tail == "right":
        # Select enrichment matrix and corresponding alpha threshold for right-tail analysis
        significance_matrix = log_enrichment_matrix
        alpha_threshold_matrix = enrichment_alpha_threshold_matrix
    elif tail == "both":
        # Select the matrix with the highest absolute values while preserving the sign
        significance_matrix = np.where(
            np.abs(log_depletion_matrix) >= np.abs(log_enrichment_matrix),
            -log_depletion_matrix,
            log_enrichment_matrix,
        )
        # Combine alpha thresholds using a logical OR operation
        alpha_threshold_matrix = np.logical_or(
            depletion_alpha_threshold_matrix, enrichment_alpha_threshold_matrix
        )
    else:
        raise ValueError("Invalid value for 'tail'. Must be 'left', 'right', or 'both'.")

    # Create a binary significance matrix where valid indices meet the alpha threshold
    valid_idxs = ~np.isnan(alpha_threshold_matrix)
    significant_binary_significance_matrix = np.zeros(alpha_threshold_matrix.shape)
    significant_binary_significance_matrix[valid_idxs] = alpha_threshold_matrix[valid_idxs]

    return significance_matrix, significant_binary_significance_matrix


def _compute_threshold_matrix(
    pvals: np.ndarray,
    fdr_pvals: Union[np.ndarray, None] = None,
    pval_cutoff: float = 0.05,
    fdr_cutoff: float = 0.05,
) -> np.ndarray:
    """
    Compute a threshold matrix indicating significance based on p-value and FDR cutoffs.

    Args:
        pvals (np.ndarray): Array of p-values for statistical tests.
        fdr_pvals (np.ndarray, optional): Array of FDR-corrected p-values corresponding to the p-values. Defaults to None.
        pval_cutoff (float, optional): Cutoff for p-value significance. Defaults to 0.05.
        fdr_cutoff (float, optional): Cutoff for FDR significance. Defaults to 0.05.

    Returns:
        np.ndarray: A threshold matrix where 1 indicates significance based on the provided cutoffs, 0 otherwise.
    """
    if fdr_pvals is not None:
        # Compute the threshold matrix based on both p-value and FDR cutoffs
        pval_below_cutoff = pvals <= pval_cutoff
        fdr_below_cutoff = fdr_pvals <= fdr_cutoff
        threshold_matrix = np.logical_and(pval_below_cutoff, fdr_below_cutoff).astype(int)
    else:
        # Compute the threshold matrix based only on p-value cutoff
        threshold_matrix = (pvals <= pval_cutoff).astype(int)

    return threshold_matrix
