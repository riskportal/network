"""
risk/stats/permutation/test_functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import numpy as np

# Note: Cython optimizations provided minimal performance benefits.
# The final version with Cython is archived in the `cython_permutation` branch.
# DISPATCH_TEST_FUNCTIONS can be found at the end of the file.


def compute_neighborhood_score_by_sum(
    neighborhoods_matrix: np.ndarray, annotation_matrix: np.ndarray
) -> np.ndarray:
    """Compute the sum of attribute values for each neighborhood.

    Args:
        neighborhoods_matrix (np.ndarray): Binary matrix representing neighborhoods.
        annotation_matrix (np.ndarray): Matrix representing annotation values.

    Returns:
        np.ndarray: Sum of attribute values for each neighborhood.
    """
    # Calculate the neighborhood score as the dot product of neighborhoods and annotations
    neighborhood_sum = np.dot(neighborhoods_matrix, annotation_matrix)
    return neighborhood_sum


def compute_neighborhood_score_by_stdev(
    neighborhoods_matrix: np.ndarray, annotation_matrix: np.ndarray
) -> np.ndarray:
    """Compute the standard deviation of neighborhood scores.

    Args:
        neighborhoods_matrix (np.ndarray): Binary matrix representing neighborhoods.
        annotation_matrix (np.ndarray): Matrix representing annotation values.

    Returns:
        np.ndarray: Standard deviation of the neighborhood scores.
    """
    # Calculate the neighborhood score as the dot product of neighborhoods and annotations
    neighborhood_score = np.dot(neighborhoods_matrix, annotation_matrix)
    # Calculate the number of elements in each neighborhood
    N = np.sum(neighborhoods_matrix, axis=1)
    # Compute the mean of the neighborhood scores
    M = neighborhood_score / N[:, None]
    # Compute the mean of squares (EXX) directly using squared annotation matrix
    EXX = np.dot(neighborhoods_matrix, annotation_matrix**2) / N[:, None]
    # Calculate variance as EXX - M^2
    variance = EXX - M**2
    # Compute the standard deviation as the square root of the variance
    neighborhood_stdev = np.sqrt(variance)
    return neighborhood_stdev


# Dictionary to dispatch statistical test functions based on the score metric
DISPATCH_TEST_FUNCTIONS = {
    "sum": compute_neighborhood_score_by_sum,
    "stdev": compute_neighborhood_score_by_stdev,
}
