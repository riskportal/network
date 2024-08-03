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


def compute_neighborhood_score_by_stdev_python(neighborhoods_matrix, annotation_matrix):
    # Calculate neighborhood score (dot product of neighborhoods and annotations matrix)
    neighborhood_score = np.dot(neighborhoods_matrix, annotation_matrix)
    # Sum across rows for neighborhoods_matrix to get N, reshape for broadcasting
    N = np.sum(neighborhoods_matrix, axis=1)
    N_reshaped = N[:, None]
    # Calculate mean of the dot product
    M = neighborhood_score / N_reshaped
    # Compute the mean of squares (EXX)
    EXX = np.dot(neighborhoods_matrix, np.power(annotation_matrix, 2)) / N_reshaped
    # Variance computation
    variance = EXX - np.power(M, 2)
    # Standard deviation computation
    stdev = np.sqrt(variance)
    return stdev


def compute_neighborhood_score_by_z_score_python(neighborhoods_matrix, annotation_matrix):
    # Sum of attribute values in a neighborhood (dot product)
    neighborhood_score = np.dot(neighborhoods_matrix, annotation_matrix)
    # Compute the number of elements in each neighborhood (assuming binary matrix for neighborhoods)
    N = np.dot(neighborhoods_matrix, np.ones(annotation_matrix.shape))
    # Calculate the mean values for each neighborhood
    M = neighborhood_score / N
    # Compute squared mean and mean of squares
    EXX = np.dot(neighborhoods_matrix, np.power(annotation_matrix, 2)) / N
    EEX = np.power(M, 2)
    # Calculate standard deviation for each neighborhood
    std = np.sqrt(EXX - EEX)
    # Calculate Z-scores, handling division by zero by setting to NaN where std is 0
    with np.errstate(divide="ignore", invalid="ignore"):
        z_scores = np.divide(M, std)
        z_scores[std == 0] = np.nan  # Handle division by zero
        z_scores[N < 3] = np.nan  # Apply threshold for minimum number of elements

    return z_scores
