"""
risk/stats/stats
~~~~~~~~~~~~~~~~
"""

import networkx as nx
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
    network,
    neighborhoods,
    annotations,
    score_metric="sum",
    null_distribution="network",
    tail="right",
    num_permutations=1000,
    impute_neighbors=False,
    imputation_depth=1,
    pval_cutoff=1.00,
    apply_fdr=False,
    fdr_cutoff=1.00,
    random_seed=888,
):
    # NOTE: Both `neighborhoods` and `annotations` are binary matrices and must NOT have any NaN values
    neighborhoods = neighborhoods.astype(np.float32)
    annotations = annotations.astype(np.float32)
    neighborhood_score_func = DISPATCH_PERMUTATION_TABLE[score_metric]
    counts_neg, counts_pos = _run_permutation_test(
        neighborhoods,
        annotations,
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
        neg_alpha_threshold_matrix = _compute_threshold_matrix(
            neg_pvals, neg_qvals, pval_cutoff=pval_cutoff, fdr_cutoff=fdr_cutoff
        )
        neg_enrichment_score = (neg_qvals**2) * (neg_pvals**0.5)
        pos_qvals = np.apply_along_axis(fdrcorrection, 1, pos_pvals)[:, 1, :]
        pos_alpha_threshold_matrix = _compute_threshold_matrix(
            pos_pvals, pos_qvals, pval_cutoff=pval_cutoff, fdr_cutoff=fdr_cutoff
        )
        pos_enrichment_score = (pos_qvals**2) * (pos_pvals**0.5)
    else:
        neg_alpha_threshold_matrix = _compute_threshold_matrix(neg_pvals, pval_cutoff=pval_cutoff)
        neg_enrichment_score = neg_pvals
        pos_alpha_threshold_matrix = _compute_threshold_matrix(pos_pvals, pval_cutoff=pval_cutoff)
        pos_enrichment_score = pos_pvals

    # Negative log10 transformation for visualization
    log_neg_enrichment_score = -np.log10(neg_enrichment_score)
    log_pos_enrichment_score = -np.log10(pos_enrichment_score)

    if tail not in {"left", "right", "both"}:
        raise ValueError("Invalid value for 'tail'. Must be 'left', 'right', or 'both'.")

    if tail == "left":
        enrichment_score = -log_neg_enrichment_score
        alpha_threshold_matrix = neg_alpha_threshold_matrix
    elif tail == "right":
        enrichment_score = log_pos_enrichment_score
        alpha_threshold_matrix = pos_alpha_threshold_matrix
    elif tail == "both":
        # Determine the highest absolute enrichment while preserving signs
        combined_enrichment_score = np.where(
            np.abs(log_neg_enrichment_score) >= np.abs(log_pos_enrichment_score),
            -log_neg_enrichment_score,
            log_pos_enrichment_score,
        )
        enrichment_score = combined_enrichment_score
        alpha_threshold_matrix = np.logical_or(
            neg_alpha_threshold_matrix, pos_alpha_threshold_matrix
        )

    if impute_neighbors:
        # Impute zero rows with nearest enriched neighbors
        enrichment_score, alpha_threshold_matrix = _impute_zero_rows(
            network, enrichment_score, alpha_threshold_matrix, max_depth=imputation_depth
        )

    valid_idxs = ~np.isnan(alpha_threshold_matrix)
    enrichment_score_binary = np.zeros(alpha_threshold_matrix.shape)
    # Filter for alpha cutoff here
    enrichment_score_binary[valid_idxs] = alpha_threshold_matrix[valid_idxs]
    sum_enriched_neighborhoods = np.sum(enrichment_score_binary, axis=0)
    return {
        "enrichment_sums": sum_enriched_neighborhoods,
        "enrichment_matrix": enrichment_score,
        "binary_enrichment_matrix_below_alpha": enrichment_score_binary,
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
    # NOTE: Prior to introducing `annotations` to ANY permuation test, all NaN values must be set to 0
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
            annotation_matrix_permut = annotations[np.random.permutation(idxs)]
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


def _impute_zero_rows(network, enrichment_matrix, alpha_threshold_matrix, max_depth=3):
    """
    Impute rows with sums of zero in the enrichment matrix based on the closest non-zero neighbors in the network graph.

    Args:
        network (NetworkX graph): The network graph with nodes having IDs matching the matrix indices.
        enrichment_matrix (np.ndarray): The enrichment matrix with rows to be imputed.
        alpha_threshold_matrix (np.ndarray): The alpha threshold matrix to be imputed similarly.
        max_depth (int): Maximum depth of nodes to traverse for imputing values.

    Returns:
        tuple: The imputed enrichment matrix and the imputed alpha threshold matrix.
    """
    zero_row_indices = np.where(alpha_threshold_matrix.sum(axis=1) == 0)[0]

    def get_euclidean_distance(node1, node2):
        pos1 = np.array(
            [
                network.nodes[node1].get(coord, 0)
                for coord in ["x", "y", "z"]
                if coord in network.nodes[node1]
            ]
        )
        pos2 = np.array(
            [
                network.nodes[node2].get(coord, 0)
                for coord in ["x", "y", "z"]
                if coord in network.nodes[node2]
            ]
        )
        return np.linalg.norm(pos1 - pos2)

    def impute_recursive(zero_row_indices, depth):
        if depth > max_depth:
            return

        rows_to_impute = []

        for row_index in zero_row_indices:
            neighbors = nx.single_source_shortest_path_length(network, row_index, cutoff=depth)
            valid_neighbors = [
                n
                for n in neighbors
                if n != row_index
                and alpha_threshold_matrix[n].sum() != 0
                and enrichment_matrix[n].sum() != 0
            ]

            if valid_neighbors:
                closest_neighbor = min(
                    valid_neighbors, key=lambda n: get_euclidean_distance(row_index, n)
                )
                enrichment_matrix[row_index] = enrichment_matrix[closest_neighbor] / np.sqrt(
                    depth + 1
                )
                alpha_threshold_matrix[row_index] = alpha_threshold_matrix[closest_neighbor]
            else:
                rows_to_impute.append(row_index)

        if rows_to_impute:
            impute_recursive(rows_to_impute, depth + 1)

    impute_recursive(zero_row_indices, 1)

    return enrichment_matrix, alpha_threshold_matrix
