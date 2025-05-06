"""
risk/_neighborhoods/_stats/_permutation/_test_functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import numpy as np
from scipy.sparse import csr_matrix

# NOTE: Cython optimizations provided minimal performance benefits.
# The final version with Cython is archived in the `cython_permutation` branch.

# DISPATCH_TEST_FUNCTIONS can be found at the end of the file.


def compute_neighborhood_score_by_sum(
    neighborhoods_matrix: csr_matrix, annotation_matrix: csr_matrix
) -> np.ndarray:
    """
    Compute the sum of attribute values for each neighborhood using sparse matrices.

    Args:
        neighborhoods_matrix (csr_matrix): Sparse binary matrix representing neighborhoods.
        annotation_matrix (csr_matrix): Sparse matrix representing annotation values.

    Returns:
        np.ndarray: Dense array of summed attribute values for each neighborhood.
    """
    # Calculate the neighborhood score as the dot product of neighborhoods and annotation
    neighborhood_score = neighborhoods_matrix @ annotation_matrix  # Sparse matrix multiplication
    # Convert the result to a dense array for downstream calculations
    neighborhood_score_dense = neighborhood_score.toarray()
    return neighborhood_score_dense


def compute_neighborhood_score_by_stdev(
    neighborhoods_matrix: csr_matrix, annotation_matrix: csr_matrix
) -> np.ndarray:
    """
    Compute the standard deviation of neighborhood scores for sparse matrices.

    Args:
        neighborhoods_matrix (csr_matrix): Sparse binary matrix representing neighborhoods.
        annotation_matrix (csr_matrix): Sparse matrix representing annotation values.

    Returns:
        np.ndarray: Standard deviation of the neighborhood scores.
    """
    # Calculate the neighborhood score as the dot product of neighborhoods and annotation
    neighborhood_score = neighborhoods_matrix @ annotation_matrix  # Sparse matrix multiplication
    # Calculate the number of elements in each neighborhood (sum of rows)
    N = neighborhoods_matrix.sum(axis=1).A.flatten()  # Convert to 1D array
    # Avoid division by zero by replacing zeros in N with np.nan temporarily
    N[N == 0] = np.nan
    # Compute the mean of the neighborhood scores
    M = neighborhood_score.multiply(1 / N[:, None]).toarray()  # Sparse element-wise division
    # Compute the mean of squares (EXX) directly using squared annotation matrix
    annotation_squared = annotation_matrix.multiply(annotation_matrix)  # Element-wise squaring
    EXX = (neighborhoods_matrix @ annotation_squared).multiply(1 / N[:, None]).toarray()
    # Calculate variance as EXX - M^2
    variance = EXX - np.power(M, 2)
    # Compute the standard deviation as the square root of the variance
    neighborhood_stdev = np.sqrt(variance)
    # Replace np.nan back with zeros in case N was 0 (no elements in the neighborhood)
    neighborhood_stdev[np.isnan(neighborhood_stdev)] = 0
    return neighborhood_stdev


# Dictionary to dispatch statistical test functions based on the score metric
DISPATCH_TEST_FUNCTIONS = {
    "sum": compute_neighborhood_score_by_sum,
    "stdev": compute_neighborhood_score_by_stdev,
}
