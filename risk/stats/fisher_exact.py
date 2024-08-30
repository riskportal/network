"""
risk/stats/fisher_exact
~~~~~~~~~~~~~~~~~~~~~~~
"""

from multiprocessing import get_context, Manager
from tqdm import tqdm
from typing import Any, Dict

import numpy as np
from scipy.stats import fisher_exact


def compute_fisher_exact_test(
    neighborhoods: np.ndarray,
    annotations: np.ndarray,
    max_workers: int = 4,
) -> Dict[str, Any]:
    """Compute Fisher's exact test for enrichment and depletion in neighborhoods.

    Args:
        neighborhoods (np.ndarray): Binary matrix representing neighborhoods.
        annotations (np.ndarray): Binary matrix representing annotations.
        max_workers (int, optional): Number of workers for multiprocessing. Defaults to 4.

    Returns:
        dict: Dictionary containing depletion and enrichment p-values.
    """
    # Ensure that the matrices are binary (boolean) and free of NaN values
    neighborhoods = neighborhoods.astype(bool)  # Convert to boolean
    annotations = annotations.astype(bool)  # Convert to boolean

    # Initialize the process of calculating p-values using multiprocessing
    ctx = get_context("spawn")
    manager = Manager()
    progress_counter = manager.Value("i", 0)
    total_tasks = neighborhoods.shape[1] * annotations.shape[1]

    # Calculate the workload per worker
    chunk_size = total_tasks // max_workers
    remainder = total_tasks % max_workers

    # Execute the Fisher's exact test using multiprocessing
    with ctx.Pool(max_workers) as pool:
        with tqdm(total=total_tasks, desc="Total progress", position=0) as progress:
            params_list = []
            start_idx = 0
            for i in range(max_workers):
                end_idx = start_idx + chunk_size + (1 if i < remainder else 0)
                params_list.append(
                    (neighborhoods, annotations, start_idx, end_idx, progress_counter)
                )
                start_idx = end_idx

            # Start the Fisher's exact test process in parallel
            results = pool.starmap_async(_fisher_exact_process_subset, params_list, chunksize=1)

            # Update progress bar based on progress_counter
            while not results.ready():
                progress.update(progress_counter.value - progress.n)
                results.wait(0.05)  # Wait for 50ms
            # Ensure progress bar reaches 100%
            progress.update(total_tasks - progress.n)

            # Accumulate results from each worker
            depletion_pvals, enrichment_pvals = [], []
            for dp, ep in results.get():
                depletion_pvals.extend(dp)
                enrichment_pvals.extend(ep)

    # Reshape the results back into arrays with the appropriate dimensions
    depletion_pvals = np.array(depletion_pvals).reshape(
        neighborhoods.shape[1], annotations.shape[1]
    )
    enrichment_pvals = np.array(enrichment_pvals).reshape(
        neighborhoods.shape[1], annotations.shape[1]
    )

    return {
        "depletion_pvals": depletion_pvals,
        "enrichment_pvals": enrichment_pvals,
    }


def _fisher_exact_process_subset(
    neighborhoods: np.ndarray,
    annotations: np.ndarray,
    start_idx: int,
    end_idx: int,
    progress_counter,
) -> tuple:
    """Process a subset of neighborhoods using Fisher's exact test.

    Args:
        neighborhoods (np.ndarray): The full neighborhood matrix.
        annotations (np.ndarray): The annotation matrix.
        start_idx (int): Starting index of the neighborhood-annotation pairs to process.
        end_idx (int): Ending index of the neighborhood-annotation pairs to process.
        progress_counter: Shared counter for tracking progress.

    Returns:
        tuple: Local p-values for depletion and enrichment.
    """
    # Initialize lists to store p-values for depletion and enrichment
    depletion_pvals = []
    enrichment_pvals = []
    # Process the subset of tasks assigned to this worker
    for idx in range(start_idx, end_idx):
        i = idx // annotations.shape[1]  # Neighborhood index
        j = idx % annotations.shape[1]  # Annotation index

        neighborhood = neighborhoods[:, i]
        annotation = annotations[:, j]

        # Calculate the contingency table values
        TP = np.sum(neighborhood & annotation)
        FP = np.sum(neighborhood & ~annotation)
        FN = np.sum(~neighborhood & annotation)
        TN = np.sum(~neighborhood & ~annotation)
        table = np.array([[TP, FP], [FN, TN]])

        # Perform Fisher's exact test for depletion (alternative='less')
        _, p_value_depletion = fisher_exact(table, alternative="less")
        depletion_pvals.append(p_value_depletion)
        # Perform Fisher's exact test for enrichment (alternative='greater')
        _, p_value_enrichment = fisher_exact(table, alternative="greater")
        enrichment_pvals.append(p_value_enrichment)

        # Update the shared progress counter
        progress_counter.value += 1

    return depletion_pvals, enrichment_pvals
