"""
risk/stats/stats
~~~~~~~~~~~~~~~~
"""

from multiprocessing import Pool, Lock
from typing import Any, Callable, Union

import numpy as np
from statsmodels.stats.multitest import fdrcorrection


def _is_notebook() -> bool:
    """Determine the type of interactive environment and return it as a dictionary.

    Returns:
        bool: True if the environment is a Jupyter notebook, False otherwise.
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other types of shell
    except NameError:
        return False  # Standard Python interpreter


if _is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


from risk.stats.permutation import (
    compute_neighborhood_score_by_sum_cython,
    compute_neighborhood_score_by_stdev_cython,
    compute_neighborhood_score_by_z_score_cython,
    compute_neighborhood_score_by_sum_python,
    compute_neighborhood_score_by_stdev_python,
    compute_neighborhood_score_by_z_score_python,
)

CYTHON_DISPATCH_PERMUTATION_TABLE = {
    "sum": compute_neighborhood_score_by_sum_cython,
    "stdev": compute_neighborhood_score_by_stdev_cython,
    "z_score": compute_neighborhood_score_by_z_score_cython,
}
PYTHON_DISPATCH_PERMUTATION_TABLE = {
    "sum": compute_neighborhood_score_by_sum_python,
    "stdev": compute_neighborhood_score_by_stdev_python,
    "z_score": compute_neighborhood_score_by_z_score_python,
}


def compute_permutation(
    neighborhoods: np.ndarray,
    annotations: np.ndarray,
    score_metric: str = "sum",
    null_distribution: str = "network",
    num_permutations: int = 1000,
    use_cython: bool = True,
    random_seed: int = 888,
    max_workers: int = 1,
) -> dict:
    """Compute permutation test for enrichment and depletion in neighborhoods.

    Args:
        neighborhoods (np.ndarray): Binary matrix representing neighborhoods.
        annotations (np.ndarray): Binary matrix representing annotations.
        score_metric (str, optional): Metric to use for scoring ('sum', 'mean', etc.). Defaults to "sum".
        null_distribution (str, optional): Type of null distribution ('network' or other). Defaults to "network".
        num_permutations (int, optional): Number of permutations to run. Defaults to 1000.
        random_seed (int, optional): Seed for random number generation. Defaults to 888.
        use_cython (bool, optional): Whether to use Cython for computation. Defaults to True.
        max_workers (int, optional): Number of workers for multiprocessing. Defaults to 1.

    Returns:
        dict: Dictionary containing depletion and enrichment p-values.
    """
    # Ensure that the matrices are in the correct format and free of NaN values
    neighborhoods = neighborhoods.astype(np.float32)
    annotations = annotations.astype(np.float32)
    # Retrieve the appropriate scoring function based on the metric and Cython usage
    if use_cython:
        neighborhood_score_func = CYTHON_DISPATCH_PERMUTATION_TABLE[score_metric]
    else:
        neighborhood_score_func = PYTHON_DISPATCH_PERMUTATION_TABLE[score_metric]
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
    # Set the random seed for reproducibility
    np.random.seed(random_seed)
    # Determine indices based on null distribution type
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
    # Determine subset size for each worker
    subset_size = num_permutations // max_workers
    remainder = num_permutations % max_workers

    if max_workers == 1:
        # If single-threaded, run the permutation process directly
        local_counts_depletion, local_counts_enrichment = _permutation_process_subset(
            annotations,
            np.array(idxs),
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
        # Prepare parameters for multiprocessing
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

        # Initialize a multiprocessing pool with a lock
        lock = Lock()
        with Pool(max_workers, initializer=_init, initargs=(lock,)) as pool:
            results = pool.starmap(_permutation_process_subset, params_list)
            # Accumulate results from each worker
            for local_counts_depletion, local_counts_enrichment in results:
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
    worker_id: int,
    use_lock: bool,
) -> tuple:
    """Process a subset of permutations for the permutation test.

    Args:
        annotation_matrix (np.ndarray): The annotation matrix.
        idxs (np.ndarray): Indices of valid rows in the matrix.
        neighborhoods_matrix_obsv (np.ndarray): Observed neighborhoods matrix.
        observed_neighborhood_scores (np.ndarray): Observed neighborhood scores.
        neighborhood_score_func (Callable): Function to calculate neighborhood scores.
        subset_size (int): Number of permutations to run in this subset.
        worker_id (int): ID of the worker process.
        use_lock (bool): Whether to use a lock for multiprocessing synchronization.

    Returns:
        tuple: Local counts of depletion and enrichment.
    """
    # Initialize local count matrices for this worker
    local_counts_depletion = np.zeros(observed_neighborhood_scores.shape)
    local_counts_enrichment = np.zeros(observed_neighborhood_scores.shape)

    if _is_notebook():
        # Hack to ensure progress bar displays correctly in Jupyter notebooks
        print(" ", end="", flush=True)

    # Initialize progress bar for tracking permutation progress
    text = f"Worker {worker_id + 1} Progress"
    if use_lock:
        with lock:
            # Set mininterval to 0.1 to prevent rapid updates and improve performance
            progress = tqdm(
                total=subset_size, desc=text, position=worker_id, leave=False, mininterval=0.1
            )
    else:
        progress = tqdm(
            total=subset_size, desc=text, position=worker_id, leave=False, mininterval=0.1
        )

    for _ in range(subset_size):
        # Permute the annotation matrix
        annotation_matrix_permut = annotation_matrix[np.random.permutation(idxs)]
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
            local_counts_enrichment, permuted_neighborhood_scores >= observed_neighborhood_scores
        )
        # Update progress bar
        if use_lock:
            with lock:
                progress.update(1)
        else:
            progress.update(1)

    # Close the progress bar once processing is complete
    if use_lock:
        with lock:
            progress.close()
    else:
        progress.close()

    return local_counts_depletion, local_counts_enrichment


def _init(lock_: Any) -> None:
    """Initialize a global lock for multiprocessing.

    Args:
        lock_ (Any): A lock object to be used in multiprocessing.
    """
    global lock
    lock = lock_  # Assign the provided lock to a global variable


def calculate_significance_matrices(
    depletion_pvals: np.ndarray,
    enrichment_pvals: np.ndarray,
    tail: str = "right",
    pval_cutoff: float = 0.05,
    apply_fdr: bool = False,
    fdr_cutoff: float = 0.05,
) -> dict:
    """Calculate significance matrices based on p-values and specified tail.

    Args:
        depletion_pvals (np.ndarray): Matrix of depletion p-values.
        enrichment_pvals (np.ndarray): Matrix of enrichment p-values.
        tail (str, optional): The tail type for significance selection ('left', 'right', 'both'). Defaults to 'right'.
        pval_cutoff (float, optional): Cutoff for p-value significance. Defaults to 0.05.
        apply_fdr (bool, optional): Whether to apply FDR correction. Defaults to False.
        fdr_cutoff (float, optional): Cutoff for FDR significance if applied. Defaults to 0.05.

    Returns:
        dict: Dictionary containing the enrichment matrix, binary significance matrix,
              and the matrix of significant enrichment values.
    """
    if apply_fdr:
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
        enrichment_matrix = (enrichment_qvals**2) * (enrichment_pvals**0.5)
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
    enrichment_matrix, binary_enrichment_matrix = _select_significance_matrices(
        tail,
        log_depletion_matrix,
        depletion_alpha_threshold_matrix,
        log_enrichment_matrix,
        enrichment_alpha_threshold_matrix,
    )

    # Filter the enrichment matrix using the binary significance matrix
    significant_enrichment_matrix = np.where(binary_enrichment_matrix == 1, enrichment_matrix, 0)

    return {
        "enrichment_matrix": enrichment_matrix,
        "binary_enrichment_matrix": binary_enrichment_matrix,
        "significant_enrichment_matrix": significant_enrichment_matrix,
    }


def _select_significance_matrices(
    tail: str,
    log_depletion_matrix: np.ndarray,
    depletion_alpha_threshold_matrix: np.ndarray,
    log_enrichment_matrix: np.ndarray,
    enrichment_alpha_threshold_matrix: np.ndarray,
) -> tuple:
    """Select significance matrices based on the specified tail type.

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
        enrichment_matrix = -log_depletion_matrix
        alpha_threshold_matrix = depletion_alpha_threshold_matrix
    elif tail == "right":
        # Select enrichment matrix and corresponding alpha threshold for right-tail analysis
        enrichment_matrix = log_enrichment_matrix
        alpha_threshold_matrix = enrichment_alpha_threshold_matrix
    elif tail == "both":
        # Select the matrix with the highest absolute values while preserving the sign
        enrichment_matrix = np.where(
            np.abs(log_depletion_matrix) >= np.abs(log_enrichment_matrix),
            -log_depletion_matrix,
            log_enrichment_matrix,
        )
        # Combine alpha thresholds using a logical OR operation
        alpha_threshold_matrix = np.logical_or(
            depletion_alpha_threshold_matrix, enrichment_alpha_threshold_matrix
        )

    # Create a binary significance matrix where valid indices meet the alpha threshold
    valid_idxs = ~np.isnan(alpha_threshold_matrix)
    binary_enrichment_matrix = np.zeros(alpha_threshold_matrix.shape)
    binary_enrichment_matrix[valid_idxs] = alpha_threshold_matrix[valid_idxs]

    return enrichment_matrix, binary_enrichment_matrix


def _compute_threshold_matrix(
    pvals: np.ndarray,
    fdr_pvals: Union[np.ndarray, None] = None,
    pval_cutoff: float = 0.05,
    fdr_cutoff: float = 0.05,
) -> np.ndarray:
    """Compute a threshold matrix indicating significance based on p-value and FDR cutoffs.

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
