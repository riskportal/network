"""
risk/stats/permutation/_python/permutation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import numpy as np


def compute_neighborhood_score_by_sum_python(neighborhoods_matrix, annotation_matrix):
    # Directly compute the dot product as we know there are no NaN values
    neighborhood_score = np.dot(
        neighborhoods_matrix, annotation_matrix
    )  # sum of attribute values in a neighborhood

    return neighborhood_score


def compute_neighborhood_score_by_variance_python(neighborhoods_matrix, annotation_matrix):
    # Convert to float64 for arithmetic operations, if not already
    A = neighborhoods_matrix.astype(np.float64)
    B = annotation_matrix
    # Calculate neighborhood score (dot product of neighborhoods and annotations matrix)
    neighborhood_score = np.dot(A, B)
    # Sum across rows for A to get N, reshape for broadcasting
    N = np.sum(A, axis=1)
    N_reshaped = N[:, None]
    # Calculate mean of the dot product
    M = neighborhood_score / N_reshaped
    # Compute the mean of squares (EXX)
    EXX = np.dot(A, np.power(B, 2)) / N_reshaped
    # Variance computation
    # Variance = EXX - (Mean)^2
    # Note: This directly computes the variance without standardizing by the standard deviation
    variance = EXX - np.power(M, 2)

    return variance


def compute_neighborhood_score_by_z_score_python(neighborhoods_matrix, annotation_matrix):
    # Since there are no NaN values, we can directly use the annotation_matrix
    A = neighborhoods_matrix
    B = annotation_matrix
    neighborhood_score = np.dot(A, B)  # sum of attribute values in a neighborhood
    # Compute the number of elements in each neighborhood (assuming binary matrix for neighborhoods)
    N = np.dot(A, np.ones(B.shape))  # No need to check for NaN, assuming all ones
    # Calculate the mean values for each neighborhood
    M = neighborhood_score / N
    # Use broadcasting to compute squared mean and mean of squares
    EXX = np.dot(A, B**2) / N
    EEX = M**2
    # Calculate standard deviation for each neighborhood
    std = np.sqrt(EXX - EEX)
    # Calculate Z-scores, handling division by zero by setting to NaN where std is 0
    with np.errstate(divide="ignore", invalid="ignore"):
        neighborhood_score = np.divide(M, std)
        neighborhood_score[std == 0] = np.nan  # Handle division by zero
        neighborhood_score[N < 3] = np.nan  # Apply threshold for minimum number of elements

    return neighborhood_score
