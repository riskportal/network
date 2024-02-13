import numpy as np
from statsmodels.stats.multitest import fdrcorrection
from rich.progress import Progress

from spp.stats.permutation import (
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
    alpha_cutoff,
    null_distribution="network",
    num_permutations=1000,
    random_seed=888,
    multiple_testing=False,
):
    # NOTE: Both `neighborhoods_matrix` and `annotation_matrix` are binary matrices and must NOT have any NaN values
    neighborhoods_matrix = neighborhoods_matrix.astype(np.int8)
    annotation_matrix = annotation_matrix.astype(np.float64)
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
    neg_pvals = counts_neg / num_permutations
    pos_pvals = counts_pos / num_permutations
    # Correct for multiple testing
    if multiple_testing:
        out = np.apply_along_axis(fdrcorrection, 1, neg_pvals)
        neg_pvals = (out[:, 1, :] ** 0.5) * (neg_pvals**2)
        out = np.apply_along_axis(fdrcorrection, 1, pos_pvals)
        pos_pvals = (out[:, 1, :] ** 0.5) * (pos_pvals**2)

    # Log-transform into neighborhood enrichment scores (NES)
    # Necessary conservative adjustment: when p-value = 0, set it to 1/num_permutations
    nes_pos = -np.log10(np.where(pos_pvals == 0, 1 / num_permutations, pos_pvals))
    nes_neg = -np.log10(np.where(neg_pvals == 0, 1 / num_permutations, neg_pvals))

    if network_enrichment_direction == "highest":
        nes = nes_pos
    if network_enrichment_direction == "lowest":
        nes = nes_neg
    else:
        # Only other option is 'both'
        nes = nes_pos - nes_neg

    valid_idxs = ~np.isnan(nes)
    nes_binary = np.zeros(nes.shape)
    nes_binary[valid_idxs] = np.abs(nes[valid_idxs]) > -np.log10(alpha_cutoff)
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
            f"[yellow]Running {num_permutations} permutations", total=num_permutations
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
                task, advance=1, description=f"[yellow]Running permutation {i+1}/{num_permutations}"
            )

    return counts_neg, counts_pos
