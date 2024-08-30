"""
risk/stats/permutation/permutation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from multiprocessing import get_context, Manager
from tqdm import tqdm
from typing import Any, Callable, Dict

import numpy as np
from threadpoolctl import threadpool_limits

from risk.stats.permutation.test_functions import DISPATCH_TEST_FUNCTIONS


def compute_permutation_test(
    neighborhoods: np.ndarray,
    annotations: np.ndarray,
    score_metric: str = "sum",
    null_distribution: str = "network",
    num_permutations: int = 1000,
    random_seed: int = 888,
    max_workers: int = 1,
) -> Dict[str, Any]:
    """Compute permutation test for enrichment and depletion in neighborhoods.

    Args:
        neighborhoods (np.ndarray): Binary matrix representing neighborhoods.
        annotations (np.ndarray): Binary matrix representing annotations.
        score_metric (str, optional): Metric to use for scoring ('sum', 'mean', etc.). Defaults to "sum".
        null_distribution (str, optional): Type of null distribution ('network' or other). Defaults to "network".
        num_permutations (int, optional): Number of permutations to run. Defaults to 1000.
        random_seed (int, optional): Seed for random number generation. Defaults to 888.
        max_workers (int, optional): Number of workers for multiprocessing. Defaults to 1.

    Returns:
        dict: Dictionary containing depletion and enrichment p-values.
    """
    # Ensure that the matrices are in the correct format and free of NaN values
    neighborhoods = neighborhoods.astype(np.float32)
    annotations = annotations.astype(np.float32)
    # Retrieve the appropriate neighborhood score function based on the metric
    neighborhood_score_func = DISPATCH_TEST_FUNCTIONS[score_metric]

    # Run the permutation test to calculate depletion and enrichment counts
    counts_depletion, counts_enrichment = _run_permutation_test(
        neighborhoods=neighborhoods,
        annotations=annotations,
        neighborhood_score_func=neighborhood_score_func,
        null_distribution=null_distribution,
        num_permutations=num_permutations,
        random_seed=random_seed,
        max_workers=max_workers,
    )
    # Compute p-values for depletion and enrichment
    # If counts are 0, set p-value to 1/num_permutations to avoid zero p-values
    depletion_pvals = np.maximum(counts_depletion, 1) / num_permutations
    enrichment_pvals = np.maximum(counts_enrichment, 1) / num_permutations

    return {
        "depletion_pvals": depletion_pvals,
        "enrichment_pvals": enrichment_pvals,
    }


def _run_permutation_test(
    neighborhoods: np.ndarray,
    annotations: np.ndarray,
    neighborhood_score_func: Callable,
    null_distribution: str = "network",
    num_permutations: int = 1000,
    random_seed: int = 888,
    max_workers: int = 4,
) -> tuple:
    """Run a permutation test to calculate enrichment and depletion counts.

    Args:
        neighborhoods (np.ndarray): The neighborhood matrix.
        annotations (np.ndarray): The annotation matrix.
        neighborhood_score_func (Callable): Function to calculate neighborhood scores.
        null_distribution (str, optional): Type of null distribution. Defaults to "network".
        num_permutations (int, optional): Number of permutations. Defaults to 1000.
        random_seed (int, optional): Seed for random number generation. Defaults to 888.
        max_workers (int, optional): Number of workers for multiprocessing. Defaults to 4.

    Returns:
        tuple: Depletion and enrichment counts.
    """
    # Initialize the RNG for reproducibility
    rng = np.random.default_rng(seed=random_seed)
    # Determine the indices to use based on the null distribution type
    if null_distribution == "network":
        idxs = range(annotations.shape[0])
    else:
        idxs = np.nonzero(np.sum(~np.isnan(annotations), axis=1))[0]

    # Replace NaNs with zeros in the annotations matrix
    annotations[np.isnan(annotations)] = 0
    annotation_matrix_obsv = annotations[idxs]
    neighborhoods_matrix_obsv = neighborhoods.T[idxs].T
    # Calculate observed neighborhood scores
    with np.errstate(invalid="ignore", divide="ignore"):
        observed_neighborhood_scores = neighborhood_score_func(
            neighborhoods_matrix_obsv, annotation_matrix_obsv
        )

    # Initialize count matrices for depletion and enrichment
    counts_depletion = np.zeros(observed_neighborhood_scores.shape)
    counts_enrichment = np.zeros(observed_neighborhood_scores.shape)

    # Determine the number of permutations to run in each worker process
    subset_size = num_permutations // max_workers
    remainder = num_permutations % max_workers

    # Use the spawn context for creating a new multiprocessing pool
    ctx = get_context("spawn")
    manager = Manager()
    progress_counter = manager.Value("i", 0)
    total_progress = num_permutations

    # Execute the permutation test using multiprocessing
    with ctx.Pool(max_workers) as pool:
        with tqdm(total=total_progress, desc="Total progress", position=0) as progress:
            # Prepare parameters for multiprocessing
            params_list = [
                (
                    annotations,
                    np.array(idxs),
                    neighborhoods_matrix_obsv,
                    observed_neighborhood_scores,
                    neighborhood_score_func,
                    subset_size + (1 if i < remainder else 0),
                    progress_counter,
                    rng,  # Pass the RNG to each process
                )
                for i in range(max_workers)
            ]

            # Start the permutation process in parallel
            results = pool.starmap_async(_permutation_process_subset, params_list, chunksize=1)

            # Update progress bar based on progress_counter
            # NOTE: Waiting for results to be ready while updating progress bar gives a big improvement
            # in performance, especially for large number of permutations and workers
            while not results.ready():
                progress.update(progress_counter.value - progress.n)
                results.wait(0.05)  # Wait for 50ms
            # Ensure progress bar reaches 100%
            progress.update(total_progress - progress.n)

            # Accumulate results from each worker
            for local_counts_depletion, local_counts_enrichment in results.get():
                counts_depletion = np.add(counts_depletion, local_counts_depletion)
                counts_enrichment = np.add(counts_enrichment, local_counts_enrichment)

    return counts_depletion, counts_enrichment


def _permutation_process_subset(
    annotation_matrix: np.ndarray,
    idxs: np.ndarray,
    neighborhoods_matrix_obsv: np.ndarray,
    observed_neighborhood_scores: np.ndarray,
    neighborhood_score_func: Callable,
    subset_size: int,
    progress_counter,
    rng: np.random.Generator,
) -> tuple:
    """Process a subset of permutations for the permutation test.

    Args:
        annotation_matrix (np.ndarray): The annotation matrix.
        idxs (np.ndarray): Indices of valid rows in the matrix.
        neighborhoods_matrix_obsv (np.ndarray): Observed neighborhoods matrix.
        observed_neighborhood_scores (np.ndarray): Observed neighborhood scores.
        neighborhood_score_func (Callable): Function to calculate neighborhood scores.
        subset_size (int): Number of permutations to run in this subset.
        progress_counter: Shared counter for tracking progress.
        rng (np.random.Generator): Random number generator object.

    Returns:
        tuple: Local counts of depletion and enrichment.
    """
    # Initialize local count matrices for this worker
    local_counts_depletion = np.zeros(observed_neighborhood_scores.shape)
    local_counts_enrichment = np.zeros(observed_neighborhood_scores.shape)
    # NOTE: Limit the number of threads used by NumPy's BLAS implementation to 1.
    # This can help prevent oversubscription of CPU resources during multiprocessing,
    # ensuring that each process doesn't use more than one CPU core.
    with threadpool_limits(limits=1, user_api="blas"):
        for _ in range(subset_size):
            # Permute the annotation matrix using the RNG
            annotation_matrix_permut = annotation_matrix[rng.permutation(idxs)]
            # Calculate permuted neighborhood scores
            with np.errstate(invalid="ignore", divide="ignore"):
                permuted_neighborhood_scores = neighborhood_score_func(
                    neighborhoods_matrix_obsv, annotation_matrix_permut
                )

            # Update local depletion and enrichment counts based on permuted scores
            local_counts_depletion = np.add(
                local_counts_depletion, permuted_neighborhood_scores <= observed_neighborhood_scores
            )
            local_counts_enrichment = np.add(
                local_counts_enrichment,
                permuted_neighborhood_scores >= observed_neighborhood_scores,
            )

            # Update the shared progress counter
            progress_counter.value += 1

    return local_counts_depletion, local_counts_enrichment
