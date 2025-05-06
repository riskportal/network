"""
risk/_neighborhoods/_stats/_permutation/_permutation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from multiprocessing import Manager, get_context
from multiprocessing.managers import ValueProxy
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
from scipy.sparse import csr_matrix
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from ._test_functions import DISPATCH_TEST_FUNCTIONS


def compute_permutation_test(
    neighborhoods: csr_matrix,
    annotation: csr_matrix,
    score_metric: str = "sum",
    null_distribution: str = "network",
    num_permutations: int = 1000,
    random_seed: int = 888,
    max_workers: int = 1,
) -> Dict[str, Any]:
    """
    Compute permutation test for enrichment and depletion in neighborhoods.

    Args:
        neighborhoods (csr_matrix): Sparse binary matrix representing neighborhoods.
        annotation (csr_matrix): Sparse binary matrix representing annotation.
        score_metric (str, optional): Metric to use for scoring ('sum' or 'stdev'). Defaults to "sum".
        null_distribution (str, optional): Type of null distribution ('network' or 'annotation'). Defaults to "network".
        num_permutations (int, optional): Number of permutations to run. Defaults to 1000.
        random_seed (int, optional): Seed for random number generation. Defaults to 888.
        max_workers (int, optional): Number of workers for multiprocessing. Defaults to 1.

    Returns:
        Dict[str, Any]: Dictionary containing depletion and enrichment p-values.
    """
    # Ensure that the matrices are in the correct format and free of NaN values
    # NOTE: Keep the data type as float32 to avoid locking issues with dot product operations
    neighborhoods = neighborhoods.astype(np.float32)
    annotation = annotation.astype(np.float32)
    # Retrieve the appropriate neighborhood score function based on the metric
    neighborhood_score_func = DISPATCH_TEST_FUNCTIONS[score_metric]

    # Run the permutation test to calculate depletion and enrichment counts
    counts_depletion, counts_enrichment = _run_permutation_test(
        neighborhoods=neighborhoods,
        annotation=annotation,
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
    neighborhoods: csr_matrix,
    annotation: csr_matrix,
    neighborhood_score_func: Callable,
    null_distribution: str = "network",
    num_permutations: int = 1000,
    random_seed: int = 888,
    max_workers: int = 4,
) -> tuple:
    """
    Run the permutation test to calculate depletion and enrichment counts.

    Args:
        neighborhoods (csr_matrix): Sparse binary matrix representing neighborhoods.
        annotation (csr_matrix): Sparse binary matrix representing annotation.
        neighborhood_score_func (Callable): Function to calculate neighborhood scores.
        null_distribution (str, optional): Type of null distribution ('network' or 'annotation'). Defaults to "network".
        num_permutations (int, optional): Number of permutations. Defaults to 1000.
        random_seed (int, optional): Seed for random number generation. Defaults to 888.
        max_workers (int, optional): Number of workers for multiprocessing. Defaults to 4.

    Returns:
        tuple: Depletion and enrichment counts.

    Raises:
        ValueError: If an invalid null_distribution value is provided.
    """
    # Initialize the RNG for reproducibility
    rng = np.random.default_rng(seed=random_seed)
    # Determine the indices to use based on the null distribution type
    if null_distribution == "network":
        idxs = range(annotation.shape[0])
    elif null_distribution == "annotation":
        idxs = np.nonzero(annotation.getnnz(axis=1) > 0)[0]
    else:
        raise ValueError(
            "Invalid null_distribution value. Choose either 'network' or 'annotation'."
        )

    # Replace NaNs with zeros in the sparse annotation matrix
    annotation.data[np.isnan(annotation.data)] = 0
    annotation_matrix_obsv = annotation[idxs]
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

    # Generate precomputed permutations
    permutations = [rng.permutation(idxs) for _ in range(num_permutations)]
    # Divide permutations into batches for workers
    batch_size = subset_size + (1 if remainder > 0 else 0)
    permutation_batches = [
        permutations[i * batch_size : (i + 1) * batch_size] for i in range(max_workers)
    ]

    # Execute the permutation test using multiprocessing
    with ctx.Pool(max_workers) as pool:
        with tqdm(total=total_progress, desc="Total progress", position=0) as progress:
            # Prepare parameters for multiprocessing
            params_list = [
                (
                    permutation_batches[i],  # Pass the batch of precomputed permutations
                    annotation,
                    neighborhoods_matrix_obsv,
                    observed_neighborhood_scores,
                    neighborhood_score_func,
                    num_permutations,
                    progress_counter,
                    max_workers,
                )
                for i in range(max_workers)
            ]

            # Start the permutation process in parallel
            results = pool.starmap_async(_permutation_process_batch, params_list, chunksize=1)

            # Update progress bar based on progress_counter
            while not results.ready():
                progress.update(progress_counter.value - progress.n)
                results.wait(0.1)  # Wait for 100ms
            # Ensure progress bar reaches 100%
            progress.update(total_progress - progress.n)

    # Accumulate results from each worker
    for local_counts_depletion, local_counts_enrichment in results.get():
        counts_depletion = np.add(counts_depletion, local_counts_depletion)
        counts_enrichment = np.add(counts_enrichment, local_counts_enrichment)

    return counts_depletion, counts_enrichment


def _permutation_process_batch(
    permutations: Union[List, Tuple, np.ndarray],
    annotation_matrix: csr_matrix,
    neighborhoods_matrix_obsv: csr_matrix,
    observed_neighborhood_scores: np.ndarray,
    neighborhood_score_func: Callable,
    num_permutations: int,
    progress_counter: ValueProxy,
    max_workers: int,
) -> tuple:
    """
    Process a batch of permutations in a worker process.

    Args:
        permutations (Union[List, Tuple, np.ndarray]): Permutation batch to process.
        annotation_matrix (csr_matrix): Sparse binary matrix representing annotation.
        neighborhoods_matrix_obsv (csr_matrix): Sparse binary matrix representing observed neighborhoods.
        observed_neighborhood_scores (np.ndarray): Observed neighborhood scores.
        neighborhood_score_func (Callable): Function to calculate neighborhood scores.
        num_permutations (int): Number of total permutations across all subsets.
        progress_counter (multiprocessing.managers.ValueProxy): Shared counter for tracking progress.
        max_workers (int): Number of workers for multiprocessing.

    Returns:
        tuple: Local counts of depletion and enrichment.
    """
    # Initialize local count matrices for this worker
    local_counts_depletion = np.zeros(observed_neighborhood_scores.shape)
    local_counts_enrichment = np.zeros(observed_neighborhood_scores.shape)

    # Limit the number of threads used by NumPy's BLAS implementation to 1 when more than one worker is used
    # NOTE: This does not work for Mac M chips due to a bug in the threadpoolctl package
    # This is currently a known issue and is being addressed by the maintainers [https://github.com/joblib/threadpoolctl/issues/135]
    limits = None if max_workers == 1 else 1
    with threadpool_limits(limits=limits, user_api="blas"):
        # Initialize a local counter for batched progress updates
        local_progress = 0
        # Calculate the modulo value based on total permutations for 1/100th frequency updates
        modulo_value = max(1, num_permutations // 100)

        for permuted_idxs in permutations:
            # Apply precomputed permutation
            annotation_matrix_permut = annotation_matrix[permuted_idxs]
            # Calculate permuted neighborhood scores
            with np.errstate(invalid="ignore", divide="ignore"):
                permuted_neighborhood_scores = neighborhood_score_func(
                    neighborhoods_matrix_obsv, annotation_matrix_permut
                )

            # Update local depletion and enrichment counts
            local_counts_depletion = np.add(
                local_counts_depletion, permuted_neighborhood_scores <= observed_neighborhood_scores
            )
            local_counts_enrichment = np.add(
                local_counts_enrichment,
                permuted_neighborhood_scores >= observed_neighborhood_scores,
            )

            # Update progress
            local_progress += 1
            if local_progress % modulo_value == 0:
                progress_counter.value += modulo_value

        # Final progress update for any remaining iterations
        if local_progress % modulo_value != 0:
            progress_counter.value += modulo_value

    return local_counts_depletion, local_counts_enrichment
