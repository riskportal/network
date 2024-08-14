"""
risk/stats/permutation
~~~~~~~~~~~~~~~~~~~~~~
"""

import numpy as np

# Note: Cython optimizations provided minimal performance benefits.
# The final version with Cython is archived in the `cython_permutation` branch.


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
    neighborhood_score = np.dot(neighborhoods_matrix, annotation_matrix)
    return neighborhood_score


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
    stdev = np.sqrt(variance)
    return stdev


def compute_neighborhood_score_by_z_score(
    neighborhoods_matrix: np.ndarray, annotation_matrix: np.ndarray
) -> np.ndarray:
    """Compute Z-scores for neighborhood scores.

    Args:
        neighborhoods_matrix (np.ndarray): Binary matrix representing neighborhoods.
        annotation_matrix (np.ndarray): Matrix representing annotation values.

    Returns:
        np.ndarray: Z-scores for each neighborhood.
    """
    # Calculate the neighborhood score as the dot product of neighborhoods and annotations
    neighborhood_score = np.dot(neighborhoods_matrix, annotation_matrix)
    # Calculate the number of elements in each neighborhood
    N = np.dot(
        neighborhoods_matrix, np.ones(annotation_matrix.shape[1], dtype=annotation_matrix.dtype)
    )
    # Compute the mean of the neighborhood scores
    M = neighborhood_score / N
    # Compute the mean of squares (EXX)
    EXX = np.dot(neighborhoods_matrix, annotation_matrix**2) / N
    # Calculate the standard deviation for each neighborhood
    variance = EXX - M**2
    std = np.sqrt(variance)
    # Calculate Z-scores, handling cases where std is 0 or N is less than 3
    with np.errstate(divide="ignore", invalid="ignore"):
        z_scores = M / std
        z_scores[(std == 0) | (N < 3)] = (
            np.nan
        )  # Handle division by zero and apply minimum threshold

    return z_scores
