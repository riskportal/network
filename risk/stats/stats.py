"""
risk/stats/stats
~~~~~~~~~~~~~~~~
"""

from tqdm import tqdm

import numpy as np
from statsmodels.stats.multitest import fdrcorrection

from risk.stats.permutation import (
    compute_neighborhood_score_by_sum_cython,
    compute_neighborhood_score_by_stdev_cython,
    compute_neighborhood_score_by_z_score_cython,
)

DISPATCH_PERMUTATION_TABLE = {
    "sum": compute_neighborhood_score_by_sum_cython,
    "stdev": compute_neighborhood_score_by_stdev_cython,
    "z_score": compute_neighborhood_score_by_z_score_cython,
}


def compute_permutation(
    neighborhoods,
    annotations,
    score_metric="sum",
    null_distribution="network",
    num_permutations=1000,
    random_seed=888,
):
    # NOTE: Both `neighborhoods` and `annotations` are binary matrices and must NOT have any NaN values
    neighborhoods = neighborhoods.astype(np.float32)
    annotations = annotations.astype(np.float32)
    neighborhood_score_func = DISPATCH_PERMUTATION_TABLE[score_metric]
    counts_depletion, counts_enrichment = _run_permutation_test(
        neighborhoods,
        annotations,
        neighborhood_score_func,
        null_distribution,
        num_permutations,
        random_seed,
    )
    # Compute P-values
    # Necessary conservative adjustment: when p-value = 0, set it to 1/num_permutations
    depletion_pvals = np.maximum(counts_depletion, 1) / num_permutations
    enrichment_pvals = np.maximum(counts_enrichment, 1) / num_permutations

    return {
        "depletion_pvals": depletion_pvals,
        "enrichment_pvals": enrichment_pvals,
    }


def _run_permutation_test(
    neighborhoods,
    annotations,
    neighborhood_score_func,
    null_distribution="network",
    num_permutations=1000,
    random_seed=888,
):
    np.random.seed(random_seed)
    # NOTE: `annotations` is a Numpy 2D matrix of type float.64 WITH NaN values!
    # NOTE: Prior to introducing `annotations` to ANY permutation test, all NaN values must be set to 0
    # First capture which distribution we want to assess enrichment: 1) Network - checks if neighborhood of
    # nodes enriched for an annotation compared to all nodes in a network or 2) Annotations file - checks if
    # neighborhood of nodes enriched for an annotation compared to all nodes found in the annotations
    if null_distribution == "network":
        idxs = range(annotations.shape[0])
    else:
        idxs = np.nonzero(np.sum(~np.isnan(annotations), axis=1))[0]
    # Setting all NaN values to 0 AFTER capturing appropriate permutation indices
    annotations[np.isnan(annotations)] = 0
    # The observed test statistic indices for `annotations`
    annotation_matrix_obsv = annotations[idxs]
    # `neighborhoods_matrix_obsv` must match in column number to `annotation_matrix_obsv` row number
    neighborhoods_matrix_obsv = neighborhoods.T[idxs].T
    # This is the observed test statistic
    with np.errstate(invalid="ignore", divide="ignore"):
        observed_neighborhood_scores = neighborhood_score_func(
            neighborhoods_matrix_obsv, annotation_matrix_obsv
        )
    # Make two empty matrices to track which permuted test statistics fell below or exceeded the observed test statistic
    counts_depletion = np.zeros(observed_neighborhood_scores.shape)
    counts_enrichment = np.zeros(observed_neighborhood_scores.shape)
    # Running permutations
    for i in tqdm(
        range(num_permutations),
        desc=f"Running {num_permutations} permutations",
        total=num_permutations,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    ):
        # `annotation_matrix_permut` must match in row number to `neighborhoods_matrix_obsv` column number
        annotation_matrix_permut = annotations[np.random.permutation(idxs)]
        # Below is NOT the bottleneck...
        with np.errstate(invalid="ignore", divide="ignore"):
            permuted_neighborhood_scores = neighborhood_score_func(
                neighborhoods_matrix_obsv, annotation_matrix_permut
            )
        counts_depletion = np.add(
            counts_depletion, permuted_neighborhood_scores <= observed_neighborhood_scores
        )
        counts_enrichment = np.add(
            counts_enrichment, permuted_neighborhood_scores >= observed_neighborhood_scores
        )

    return counts_depletion, counts_enrichment


def calculate_significance_matrices(
    depletion_pvals,
    enrichment_pvals,
    tail="right",
    pval_cutoff=0.05,
    apply_fdr=False,
    fdr_cutoff=0.05,
):
    """
    Calculate significance matrices based on the specified tail.

    Args:
        tail (str): The tail type ('left', 'right', 'both').
        apply_fdr (bool): Whether to apply FDR correction.
        depletion_pvals (np.ndarray): Depletion p-values matrix.
        enrichment_pvals (np.ndarray): Enrichment p-values matrix.
        pval_cutoff (float): p-value cutoff for significance.
        fdr_cutoff (float): FDR cutoff for significance.

    Returns:
        dict: Dictionary containing the significance matrix and binary significance matrix.
    """
    if apply_fdr:
        depletion_qvals = np.apply_along_axis(fdrcorrection, 1, depletion_pvals)[:, 1, :]
        depletion_alpha_threshold_matrix = _compute_threshold_matrix(
            depletion_pvals, depletion_qvals, pval_cutoff=pval_cutoff, fdr_cutoff=fdr_cutoff
        )
        depletion_matrix = (depletion_qvals**2) * (depletion_pvals**0.5)
        enrichment_qvals = np.apply_along_axis(fdrcorrection, 1, enrichment_pvals)[:, 1, :]
        enrichment_alpha_threshold_matrix = _compute_threshold_matrix(
            enrichment_pvals, enrichment_qvals, pval_cutoff=pval_cutoff, fdr_cutoff=fdr_cutoff
        )
        enrichment_matrix = (enrichment_qvals**2) * (enrichment_pvals**0.5)
    else:
        depletion_alpha_threshold_matrix = _compute_threshold_matrix(
            depletion_pvals, pval_cutoff=pval_cutoff
        )
        depletion_matrix = depletion_pvals
        enrichment_alpha_threshold_matrix = _compute_threshold_matrix(
            enrichment_pvals, pval_cutoff=pval_cutoff
        )
        enrichment_matrix = enrichment_pvals

    # Negative log10 transformation for visualization
    log_depletion_matrix = -np.log10(depletion_matrix)
    log_enrichment_matrix = -np.log10(enrichment_matrix)

    significance_matrix, binary_significance_matrix = _select_significance_matrices(
        tail,
        log_depletion_matrix,
        depletion_alpha_threshold_matrix,
        log_enrichment_matrix,
        enrichment_alpha_threshold_matrix,
    )
    return {
        "significance_matrix": significance_matrix,
        "binary_significance_matrix": binary_significance_matrix,
    }


def _select_significance_matrices(
    tail,
    log_depletion_matrix,
    depletion_alpha_threshold_matrix,
    log_enrichment_matrix,
    enrichment_alpha_threshold_matrix,
):
    """
    Select significance matrices based on the specified tail.

    Args:
        tail (str): The tail type ('left', 'right', 'both').
        log_depletion_matrix (np.ndarray): Log depletion matrix.
        depletion_alpha_threshold_matrix (np.ndarray): Alpha threshold matrix for depletion.
        log_enrichment_matrix (np.ndarray): Log enrichment matrix.
        enrichment_alpha_threshold_matrix (np.ndarray): Alpha threshold matrix for enrichment.

    Returns:
        dict: Dictionary containing the significance matrix and binary significance matrix.
    """
    if tail not in {"left", "right", "both"}:
        raise ValueError("Invalid value for 'tail'. Must be 'left', 'right', or 'both'.")

    if tail == "left":
        significance_matrix = -log_depletion_matrix
        alpha_threshold_matrix = depletion_alpha_threshold_matrix
    elif tail == "right":
        significance_matrix = log_enrichment_matrix
        alpha_threshold_matrix = enrichment_alpha_threshold_matrix
    elif tail == "both":
        # Determine the highest absolute enrichment while preserving signs
        significance_matrix = np.where(
            np.abs(log_depletion_matrix) >= np.abs(log_enrichment_matrix),
            -log_depletion_matrix,
            log_enrichment_matrix,
        )
        alpha_threshold_matrix = np.logical_or(
            depletion_alpha_threshold_matrix, enrichment_alpha_threshold_matrix
        )

    valid_idxs = ~np.isnan(alpha_threshold_matrix)
    binary_significance_matrix = np.zeros(alpha_threshold_matrix.shape)
    # Filter for alpha cutoff here
    binary_significance_matrix[valid_idxs] = alpha_threshold_matrix[valid_idxs]

    return significance_matrix, binary_significance_matrix


def _compute_threshold_matrix(pvals, fdr_pvals=None, pval_cutoff=0.05, fdr_cutoff=0.05):
    """
    Computes a threshold matrix based on whether the values meet both p-value and FDR cutoffs.

    Parameters:
    - pvals: A numpy array of p-values.
    - fdr_pvals: Optional; a numpy array of FDR-corrected p-values corresponding to `pvals`.
    - pval_cutoff: The p-value cutoff for significance.
    - fdr_cutoff: The FDR cutoff for significance.

    Returns:
    - A threshold matrix where 1 indicates both p-value and FDR meet the cutoffs, 0 otherwise.
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
