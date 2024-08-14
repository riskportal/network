# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False
cimport cython
import numpy as np
cimport numpy as np


def compute_neighborhood_score_by_sum_cython(
    np.ndarray[np.float32_t, ndim=2] neighborhoods,
    np.ndarray[np.float32_t, ndim=2] annotation_matrix,
):
    cdef np.float32_t[:, :] neighborhood_score
    # Calculate the dot product of neighborhoods and annotation matrix
    neighborhood_score = np.dot(neighborhoods, annotation_matrix)
    return np.asarray(neighborhood_score)


def compute_neighborhood_score_by_stdev_cython(
    np.ndarray[np.float32_t, ndim=2] neighborhoods,
    np.ndarray[np.float32_t, ndim=2] annotation_matrix,
):
    cdef np.ndarray[np.float32_t, ndim=2] neighborhood_score
    cdef np.ndarray[np.float32_t, ndim=2] EXX
    cdef np.ndarray[np.float32_t, ndim=1] N
    cdef np.ndarray[np.float32_t, ndim=2] M
    cdef np.ndarray[np.float32_t, ndim=2] variance

    # Calculate the dot product of neighborhoods and annotation matrix
    neighborhood_score = np.dot(neighborhoods, annotation_matrix)    
    # Sum across rows for neighborhoods to get N, reshape for broadcasting
    N = np.sum(neighborhoods, axis=1)
    # Mean of the dot product
    M = neighborhood_score / N[:, None]
    # Compute the mean of squares (EXX) directly using squared annotation matrix
    EXX = np.dot(neighborhoods, annotation_matrix * annotation_matrix) / N[:, None]
    # Variance computation in place to reduce temporary arrays
    variance = EXX - M * M
    # Directly return the standard deviation
    return np.sqrt(variance)


def compute_neighborhood_score_by_z_score_cython(
    np.ndarray[np.float32_t, ndim=2] neighborhoods,
    np.ndarray[np.float32_t, ndim=2] annotation_matrix,
):
    cdef np.ndarray[np.float32_t, ndim=2] neighborhood_score
    cdef np.ndarray[np.float32_t, ndim=2] EXX
    cdef np.ndarray[np.float32_t, ndim=1] N
    cdef np.ndarray[np.float32_t, ndim=2] M
    cdef np.ndarray[np.float32_t, ndim=2] variance
    cdef np.ndarray[np.float32_t, ndim=2] stdev

    # Calculate the dot product of neighborhoods and annotation matrix
    neighborhood_score = np.dot(neighborhoods, annotation_matrix)
    # Sum across rows for neighborhoods to get N
    N = np.sum(neighborhoods, axis=1)
    # Mean of the dot product
    M = neighborhood_score / N[:, None]
    # Compute the mean of squares (EXX) directly using squared annotation matrix
    EXX = np.dot(neighborhoods, annotation_matrix * annotation_matrix) / N[:, None]
    # Variance computation in place to reduce temporary arrays
    variance = EXX - M * M
    # Standard deviation computation
    stdev = np.sqrt(variance)
    # Z-score computation with error handling
    with np.errstate(divide='ignore', invalid='ignore'):
        neighborhood_score = M / stdev
        # Handle divisions by zero or stdev == 0
        neighborhood_score[np.isnan(neighborhood_score)] = 0

    return neighborhood_score
