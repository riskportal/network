"""
risk/stats/stats
~~~~~~~~~~~~~~~~
"""

import numpy as np
from statsmodels.stats.multitest import fdrcorrection
from rich.progress import Progress

from risk.stats.permutation import (
    compute_neighborhood_score_by_sum_cython,
    compute_neighborhood_score_by_variance_cython,
    compute_neighborhood_score_by_zscore_cython,
)

DISPATCH_PERMUTATION_TABLE = {
    "sum": compute_neighborhood_score_by_sum_cython,
    "variance": compute_neighborhood_score_by_variance_cython,
    "zscore": compute_neighborhood_score_by_zscore_cython,
}


def compute_pvalues_by_randomization(
    neighborhoods_matrix,
    annotation_matrix,
    neighborhood_score_metric,
    network_enrichment_direction,
    pval_cutoff=1.00,
    apply_fdr=False,
    fdr_cutoff=1.00,
    null_distribution="network",
    num_permutations=1000,
    random_seed=888,
):
    # NOTE: Both `neighborhoods_matrix` and `annotation_matrix` are binary matrices and must NOT have any NaN values
    neighborhoods_matrix = neighborhoods_matrix.astype(np.float32)
    annotation_matrix = annotation_matrix.astype(np.float32)
    neighborhood_score_func = DISPATCH_PERMUTATION_TABLE[neighborhood_score_metric]
    counts_neg, counts_pos = run_permutation_test(
        neighborhoods_matrix,
        annotation_matrix,
        neighborhood_score_func,
        null_distribution,
        num_permutations,
        random_seed,
    )
    # Compute P-values
    # Necessary conservative adjustment: when p-value = 0, set it to 1/num_permutations
    neg_pvals = np.maximum(counts_neg, 1) / num_permutations
    pos_pvals = np.maximum(counts_pos, 1) / num_permutations
    # Correct for multiple testing
    if apply_fdr:
        neg_qvals = np.apply_along_axis(fdrcorrection, 1, neg_pvals)[:, 1, :]
        neg_alpha_threshold_matrix = compute_threshold_matrix(
            neg_pvals, neg_qvals, pval_cutoff=pval_cutoff, fdr_cutoff=fdr_cutoff
        )
        neg_enrichment_score = (neg_qvals**2) * (neg_pvals**0.5)
        pos_qvals = np.apply_along_axis(fdrcorrection, 1, pos_pvals)[:, 1, :]
        pos_alpha_threshold_matrix = compute_threshold_matrix(
            pos_pvals, pos_qvals, pval_cutoff=pval_cutoff, fdr_cutoff=fdr_cutoff
        )
        pos_enrichment_score = (pos_qvals**2) * (pos_pvals**0.5)
    else:
        neg_alpha_threshold_matrix = compute_threshold_matrix(neg_pvals, pval_cutoff=pval_cutoff)
        neg_enrichment_score = neg_pvals
        pos_alpha_threshold_matrix = compute_threshold_matrix(pos_pvals, pval_cutoff=pval_cutoff)
        pos_enrichment_score = pos_pvals

    # Log-transform into neighborhood enrichment scores (NES)
    nes_neg = -np.log10(neg_enrichment_score)
    nes_pos = -np.log10(pos_enrichment_score)

    if network_enrichment_direction == "lowest":
        # These two matrices should have the same shape
        nes = nes_neg
        alpha_threshold_matrix = neg_alpha_threshold_matrix
    else:
        # These two matrices should have the same shape
        nes = nes_pos
        alpha_threshold_matrix = pos_alpha_threshold_matrix

    valid_idxs = ~np.isnan(alpha_threshold_matrix)
    nes_binary = np.zeros(alpha_threshold_matrix.shape)
    # Filter for alpha cutoff here
    nes_binary[valid_idxs] = alpha_threshold_matrix[valid_idxs]
    sum_enriched_neighborhoods = np.sum(nes_binary, axis=0)
    return {
        "neighborhood_enrichment_matrix": nes,
        "neighborhood_binary_enrichment_matrix_below_alpha": nes_binary,
        "neighborhood_enrichment_sums": sum_enriched_neighborhoods,
    }


def run_permutation_test(
    neighborhoods_matrix,
    annotation_matrix,
    neighborhood_score_func,
    null_distribution="network",
    num_permutations=1000,
    random_seed=888,
):
    np.random.seed(random_seed)
    # NOTE: `annotation_matrix` is a Numpy 2D matrix of type float.64 WITH NaN values!
    # NOTE: Prior to introducing `annotation_matrix` to ANY permuation test, all NaN values must be set to 0
    # First capture which distribution we want to assess enrichment: 1) Network - checks if neighborhood of
    # nodes enriched for an annotation compared to all nodes in a network or 2) Annotation file - checks if
    # neighborhood of nodes enriched for an annotation compared to all nodes found in the annotation
    if null_distribution == "network":
        idxs = range(annotation_matrix.shape[0])
    else:
        idxs = np.nonzero(np.sum(~np.isnan(annotation_matrix), axis=1))[0]
    # Setting all NaN values to 0 AFTER capturing appropriate permutation indices
    annotation_matrix[np.isnan(annotation_matrix)] = 0
    # The observed test statistic indices for `annotation_matrix`
    annotation_matrix_obsv = annotation_matrix[idxs]
    # `neighborhoods_matrix_obsv` must match in column number to `annotation_matrix_obsv` row number
    neighborhoods_matrix_obsv = neighborhoods_matrix.T[idxs].T
    # This is the observed test statistic
    with np.errstate(invalid="ignore", divide="ignore"):
        N_in_neighborhood_in_group_obsv = neighborhood_score_func(
            neighborhoods_matrix_obsv, annotation_matrix_obsv
        )
    # Make two empty matrices to track which permuted test statistics fell below or exceeded the observed test statistic
    counts_neg = np.zeros(N_in_neighborhood_in_group_obsv.shape)
    counts_pos = np.zeros(N_in_neighborhood_in_group_obsv.shape)
    with Progress() as progress:
        task = progress.add_task(
            f"[cyan]Running[/cyan] [yellow]{num_permutations} permutation 0/{num_permutations}[/yellow]",
            total=num_permutations,
        )
        # We are computing the permuted test statistics
        for i in range(num_permutations):
            # `annotation_matrix_permut` must match in row number to `neighborhoods_matrix_obsv` column number
            annotation_matrix_permut = annotation_matrix[np.random.permutation(idxs)]
            # Below is NOT the bottleneck...
            with np.errstate(invalid="ignore", divide="ignore"):
                N_in_neighborhood_in_group_perm = neighborhood_score_func(
                    neighborhoods_matrix_obsv, annotation_matrix_permut
                )
            counts_neg = np.add(
                counts_neg, N_in_neighborhood_in_group_perm <= N_in_neighborhood_in_group_obsv
            )
            counts_pos = np.add(
                counts_pos, N_in_neighborhood_in_group_perm >= N_in_neighborhood_in_group_obsv
            )
            progress.update(
                task,
                advance=1,
                description=f"[cyan]Running[/cyan] [yellow]permutation {i+1}/{num_permutations}",
            )

    return counts_neg, counts_pos


def compute_threshold_matrix(pvals, fdr_pvals=None, pval_cutoff=0.05, fdr_cutoff=0.05):
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
