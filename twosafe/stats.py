import os

import numpy as np
import pebble
from tqdm import tqdm


def map_permutation_processes(iterable, max_workers=os.cpu_count(), timeout=None):
    if max_workers == 1:
        return _distributed_permutation_test(iterable)

    else:
        with pebble.ProcessPool(max_workers=max_workers, initializer=_limit_cpu) as pool:
            future = pool.map(_distributed_permutation_test, iterable, timeout=timeout)
        results = future.result()
        counts_neg_list, counts_pos_list = map(list, zip(*results))
        counts_neg = np.sum(np.stack(counts_neg_list, axis=2), axis=2)
        counts_pos = np.sum(np.stack(counts_pos_list, axis=2), axis=2)
        return counts_neg, counts_pos


def compute_neighborhood_score(neighborhood_to_node, node_to_attribute, neighborhood_score_type):
    with np.errstate(invalid="ignore", divide="ignore"):
        A = neighborhood_to_node
        B = np.where(~np.isnan(node_to_attribute), node_to_attribute, 0)
        NA = A
        NB = np.where(~np.isnan(node_to_attribute), 1, 0)
        AB = np.dot(A, B)  # sum of attribute values in a neighborhood
        neighborhood_score = AB
        if neighborhood_score_type == "z-score":
            N = np.dot(NA, NB)  # number of not-NaNs values in a neighborhood
            M = np.divide(AB, N)  # average attribute value in a neighborhood
            EXX = np.divide(np.dot(A, np.power(B, 2)), N)
            EEX = np.power(M, 2)
            std = np.sqrt(EXX - EEX)  # standard deviation of attribute values in a neighborhood
            neighborhood_score = np.divide(M, std)
            neighborhood_score[std == 0] = np.nan
            neighborhood_score[N < 3] = np.nan

    return neighborhood_score


def _distributed_permutation_test(arg_tuple):
    (
        neighborhood_to_node,
        node_to_attribute,
        neighborhood_score_type,
        num_permutations,
        random_seed,
    ) = arg_tuple
    np.random.seed(random_seed)
    node_to_attribute = np.copy(node_to_attribute)

    N_in_neighborhood_in_group = compute_neighborhood_score(
        neighborhood_to_node, node_to_attribute, neighborhood_score_type
    )
    idxs = np.nonzero(np.sum(~np.isnan(node_to_attribute), axis=1))[0]
    counts_neg = np.zeros(N_in_neighborhood_in_group.shape)
    counts_pos = np.zeros(N_in_neighborhood_in_group.shape)
    for _ in tqdm(np.arange(num_permutations)):
        # Permute only the rows that have values
        node_to_attribute[idxs, :] = node_to_attribute[np.random.permutation(idxs), :]
        N_in_neighborhood_in_group_perm = compute_neighborhood_score(
            neighborhood_to_node, node_to_attribute, neighborhood_score_type
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            counts_neg = np.add(
                counts_neg, N_in_neighborhood_in_group_perm <= N_in_neighborhood_in_group
            )
            counts_pos = np.add(
                counts_pos, N_in_neighborhood_in_group_perm >= N_in_neighborhood_in_group
            )

    return counts_neg, counts_pos
