"""
risk/stats/permutation/_python/permutation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import numpy as np
from threadpoolctl import threadpool_limits


def compute_neighborhood_score_by_sum_python(
    neighborhoods_matrix: np.ndarray, annotation_matrix: np.ndarray
) -> np.ndarray:
    """Compute the sum of attribute values for each neighborhood.

    Args:
        neighborhoods_matrix (np.ndarray): Binary matrix representing neighborhoods.
        annotation_matrix (np.ndarray): Matrix representing annotation values.

    Returns:
        np.ndarray: Sum of attribute values for each neighborhood.
    """
    # Limit the number of threads used by np.dot
    with threadpool_limits(limits=1, user_api="blas"):
        # Directly compute the dot product to get the sum of attribute values in each neighborhood
        neighborhood_score = np.dot(neighborhoods_matrix, annotation_matrix)

    return neighborhood_score


def compute_neighborhood_score_by_stdev_python(
    neighborhoods_matrix: np.ndarray, annotation_matrix: np.ndarray
) -> np.ndarray:
    """Compute the standard deviation of neighborhood scores.

    Args:
        neighborhoods_matrix (np.ndarray): Binary matrix representing neighborhoods.
        annotation_matrix (np.ndarray): Matrix representing annotation values.

    Returns:
        np.ndarray: Standard deviation of the neighborhood scores.
    """
    # Perform dot product directly using the inputs with limited threads
    with threadpool_limits(limits=1, user_api="blas"):
        neighborhood_score = np.dot(neighborhoods_matrix, annotation_matrix)

    # Calculate the number of elements in each neighborhood and reshape for broadcasting
    N = np.sum(neighborhoods_matrix, axis=1)
    N_reshaped = N[:, None]
    # Compute the mean of the neighborhood scores
    M = neighborhood_score / N_reshaped
    with threadpool_limits(limits=1, user_api="blas"):
        # Compute the mean of squares (EXX) for annotation values
        EXX = np.dot(neighborhoods_matrix, np.power(annotation_matrix, 2)) / N_reshaped

    # Calculate variance as EXX - M^2
    variance = EXX - np.power(M, 2)
    # Compute the standard deviation as the square root of the variance
    stdev = np.sqrt(variance)
    return stdev


def compute_neighborhood_score_by_z_score_python(
    neighborhoods_matrix: np.ndarray, annotation_matrix: np.ndarray
) -> np.ndarray:
    """Compute Z-scores for neighborhood scores.

    Args:
        neighborhoods_matrix (np.ndarray): Binary matrix representing neighborhoods.
        annotation_matrix (np.ndarray): Matrix representing annotation values.

    Returns:
        np.ndarray: Z-scores for each neighborhood.
    """
    # Perform dot product directly using the inputs with limited threads
    with threadpool_limits(limits=1, user_api="blas"):
        # Calculate the neighborhood score as the dot product of neighborhoods and annotations
        neighborhood_score = np.dot(neighborhoods_matrix, annotation_matrix)

    # Calculate the number of elements in each neighborhood
    N = np.dot(neighborhoods_matrix, np.ones(annotation_matrix.shape))
    # Compute the mean of the neighborhood scores
    M = neighborhood_score / N
    with threadpool_limits(limits=1, user_api="blas"):
        # Compute the mean of squares (EXX) and the squared mean (EEX)
        EXX = np.dot(neighborhoods_matrix, np.power(annotation_matrix, 2)) / N

    EEX = np.power(M, 2)
    # Calculate the standard deviation for each neighborhood
    std = np.sqrt(EXX - EEX)
    # Calculate Z-scores, handling cases where std is 0 or N is less than 3
    with np.errstate(divide="ignore", invalid="ignore"):
        z_scores = np.divide(M, std)
        z_scores[std == 0] = np.nan  # Handle division by zero
        z_scores[N < 3] = np.nan  # Apply threshold for minimum number of elements

    return z_scores
