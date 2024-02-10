import os
import time

import numpy as np
import pebble
from statsmodels.stats.multitest import fdrcorrection
from tqdm import tqdm


def compute_pvalues_by_randomization(
    neighborhoods,
    annotation_matrix,
    neighborhood_score_metric,
    network_enrichment_direction,
    alpha_cutoff,
    num_permutations=1000,
    max_workers=1,
    random_seed=888,
    multiple_testing=False,
):
    print("Computing P-values by randomization...")
    # Pause for 1 sec to prevent the progress bar from showing up too early
    time.sleep(1)
    num_permutations_per_process = (
        np.ceil(num_permutations / max_workers).astype(int) if max_workers > 1 else num_permutations
    )
    arg_tuple = (
        neighborhoods,
        annotation_matrix,
        neighborhood_score_metric,
        num_permutations_per_process,
        random_seed,
    )
    counts_neg, counts_pos = map_permutation_processes(arg_tuple, max_workers=max_workers)
    N_in_neighborhood_in_group = compute_neighborhood_score(
        neighborhoods, annotation_matrix, neighborhood_score_metric
    )
    idx = np.isnan(N_in_neighborhood_in_group)
    counts_neg[idx] = np.nan
    counts_pos[idx] = np.nan
    # Compute P-values
    neg_pvals = counts_neg / num_permutations
    pos_pvals = counts_pos / num_permutations
    # Correct for multiple testing
    if multiple_testing:
        out = np.apply_along_axis(fdrcorrection, 1, neg_pvals)
        neg_pvals = out[:, 1, :]
        out = np.apply_along_axis(fdrcorrection, 1, pos_pvals)
        pos_pvals = out[:, 1, :]

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


def map_permutation_processes(iterable, max_workers=os.cpu_count(), timeout=None):
    if max_workers == 1:
        counts_neg, counts_pos = distributed_permutation_test(iterable)
    else:
        ...
        # with pebble.ProcessPool(max_workers=max_workers) as pool:
        #     future = pool.map(distributed_permutation_test, iterable, timeout=timeout)
        # results = future.result()
        # counts_neg_list, counts_pos_list = map(list, zip(*results))
        # counts_neg = np.sum(np.stack(counts_neg_list, axis=2), axis=2)
        # counts_pos = np.sum(np.stack(counts_pos_list, axis=2), axis=2)
    return counts_neg, counts_pos


def distributed_permutation_test(arg_tuple):
    (
        neighborhoods,
        annotation_matrix,
        neighborhood_score_metric,
        num_permutations,
        random_seed,
    ) = arg_tuple
    np.random.seed(random_seed)
    N_in_neighborhood_in_group = compute_neighborhood_score(
        neighborhoods, annotation_matrix, neighborhood_score_metric
    )
    idxs = np.nonzero(np.sum(~np.isnan(annotation_matrix), axis=1))[0]
    counts_neg = np.zeros(N_in_neighborhood_in_group.shape)
    counts_pos = np.zeros(N_in_neighborhood_in_group.shape)
    for _ in tqdm(np.arange(num_permutations)):
        # Permute only the rows that have values
        annotation_matrix[idxs, :] = annotation_matrix[np.random.permutation(idxs), :]
        N_in_neighborhood_in_group_perm = compute_neighborhood_score(
            neighborhoods, annotation_matrix, neighborhood_score_metric
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            counts_neg = np.add(
                counts_neg, N_in_neighborhood_in_group_perm <= N_in_neighborhood_in_group
            )
            counts_pos = np.add(
                counts_pos, N_in_neighborhood_in_group_perm >= N_in_neighborhood_in_group
            )

    return counts_neg, counts_pos


def compute_neighborhood_score(neighborhoods, annotation_matrix, neighborhood_score_metric):
    with np.errstate(invalid="ignore", divide="ignore"):
        A = neighborhoods
        B = np.where(~np.isnan(annotation_matrix), annotation_matrix, 0)
        NA = A
        NB = np.where(~np.isnan(annotation_matrix), 1, 0)
        AB = np.dot(A, B)  # sum of attribute values in a neighborhood
        neighborhood_score = AB
        if neighborhood_score_metric == "z-score":
            N = np.dot(NA, NB)  # number of not-NaNs values in a neighborhood
            M = np.divide(AB, N)  # average attribute value in a neighborhood
            EXX = np.divide(np.dot(A, np.power(B, 2)), N)
            EEX = np.power(M, 2)
            std = np.sqrt(EXX - EEX)  # standard deviation of attribute values in a neighborhood
            neighborhood_score = np.divide(M, std)
            neighborhood_score[std == 0] = np.nan
            neighborhood_score[N < 3] = np.nan

    return neighborhood_score
