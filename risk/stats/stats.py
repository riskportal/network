"""
risk/stats/stats
~~~~~~~~~~~~~~~~
"""

from multiprocessing import Pool, Lock
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
    max_workers=1,
):
    # NOTE: Both `neighborhoods` and `annotations` are binary matrices and must NOT have any NaN values
    neighborhoods = neighborhoods.astype(np.float32)
    annotations = annotations.astype(np.float32)
    neighborhood_score_func = DISPATCH_PERMUTATION_TABLE[score_metric]
    counts_depletion, counts_enrichment = _run_permutation_test(
        neighborhoods=neighborhoods,
        annotations=annotations,
        neighborhood_score_func=neighborhood_score_func,
        null_distribution=null_distribution,
        num_permutations=num_permutations,
        random_seed=random_seed,
        max_workers=max_workers,
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
    max_workers=4,
):
    np.random.seed(random_seed)
    if null_distribution == "network":
        idxs = range(annotations.shape[0])
    else:
        idxs = np.nonzero(np.sum(~np.isnan(annotations), axis=1))[0]

    annotations[np.isnan(annotations)] = 0
    annotation_matrix_obsv = annotations[idxs]
    neighborhoods_matrix_obsv = neighborhoods.T[idxs].T

    with np.errstate(invalid="ignore", divide="ignore"):
        observed_neighborhood_scores = neighborhood_score_func(
            neighborhoods_matrix_obsv, annotation_matrix_obsv
        )

    counts_depletion = np.zeros(observed_neighborhood_scores.shape)
    counts_enrichment = np.zeros(observed_neighborhood_scores.shape)

    subset_size = num_permutations // max_workers
    remainder = num_permutations % max_workers

    if max_workers == 1:
        local_counts_depletion, local_counts_enrichment = _permutation_process_subset(
            annotations,
            idxs,
            neighborhoods_matrix_obsv,
            observed_neighborhood_scores,
            neighborhood_score_func,
            num_permutations,
            0,
            False,
        )
        counts_depletion = np.add(counts_depletion, local_counts_depletion)
        counts_enrichment = np.add(counts_enrichment, local_counts_enrichment)
    else:
        params_list = [
            (
                annotations,
                idxs,
                neighborhoods_matrix_obsv,
                observed_neighborhood_scores,
                neighborhood_score_func,
                subset_size + (1 if i < remainder else 0),
                i,
                True,
            )
            for i in range(max_workers)
        ]

        lock = Lock()
        with Pool(max_workers, initializer=_init, initargs=(lock,)) as pool:
            results = pool.starmap(_permutation_process_subset, params_list)

            for local_counts_depletion, local_counts_enrichment in results:
                counts_depletion = np.add(counts_depletion, local_counts_depletion)
                counts_enrichment = np.add(counts_enrichment, local_counts_enrichment)

    return counts_depletion, counts_enrichment


def _permutation_process_subset(
    annotation_matrix,
    idxs,
    neighborhoods_matrix_obsv,
    observed_neighborhood_scores,
    neighborhood_score_func,
    subset_size,
    worker_id,
    use_lock,
):
    local_counts_depletion = np.zeros(observed_neighborhood_scores.shape)
    local_counts_enrichment = np.zeros(observed_neighborhood_scores.shape)

    text = f"Worker {worker_id + 1} Progress"
    if use_lock:
        with lock:
            progress = tqdm(total=subset_size, desc=text, position=worker_id, leave=False)
    else:
        progress = tqdm(total=subset_size, desc=text, position=worker_id, leave=False)

    for _ in range(subset_size):
        annotation_matrix_permut = annotation_matrix[np.random.permutation(idxs)]
        with np.errstate(invalid="ignore", divide="ignore"):
            permuted_neighborhood_scores = neighborhood_score_func(
                neighborhoods_matrix_obsv, annotation_matrix_permut
            )
        local_counts_depletion = np.add(
            local_counts_depletion, permuted_neighborhood_scores <= observed_neighborhood_scores
        )
        local_counts_enrichment = np.add(
            local_counts_enrichment, permuted_neighborhood_scores >= observed_neighborhood_scores
        )
        if use_lock:
            with lock:
                progress.update(1)
        else:
            progress.update(1)

    if use_lock:
        with lock:
            progress.close()
    else:
        progress.close()

    return local_counts_depletion, local_counts_enrichment


def _init(lock_):
    global lock
    lock = lock_


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
