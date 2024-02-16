# cython: language_level=3
import numpy as np
cimport numpy as np
cimport cython


@cython.boundscheck(False)  # Disable bounds checking for entire function
@cython.wraparound(False)   # Disable negative index wrapping for entire function
def compute_neighborhood_score_by_sum_cython(
    np.ndarray[np.float32_t, ndim=2] neighborhoods,
    np.ndarray[np.float32_t, ndim=2] annotation_matrix,
    ):
    # NOTE: `np.dot` is already highly optimized. The smaller the dtype the faster the algorithm!
    cdef np.float32_t[:, :] neighborhood_score = np.dot(neighborhoods, annotation_matrix)
    return np.asarray(neighborhood_score)


@cython.boundscheck(False)
@cython.wraparound(False)
def compute_neighborhood_score_by_variance_cython(
    np.ndarray[np.float32_t, ndim=2] neighborhoods,
    np.ndarray[np.float32_t, ndim=2] annotation_matrix,
    ):
    # Convert to float32 for arithmetic operations, if not already
    cdef np.ndarray[np.float32_t, ndim=2] A = neighborhoods.astype(np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] B = annotation_matrix
    cdef np.ndarray[np.float32_t, ndim=2] neighborhood_score = np.dot(A, B)    
    # Sum across rows for A to get N, reshape for broadcasting
    cdef np.ndarray[np.float32_t, ndim=1] N = np.sum(A, axis=1)
    cdef np.ndarray[np.float32_t, ndim=2] N_reshaped = N[:, None]
    # Mean of the dot product
    cdef np.ndarray[np.float32_t, ndim=2] M = neighborhood_score / N_reshaped
    # Compute the mean of squares (EXX)
    cdef np.ndarray[np.float32_t, ndim=2] EXX = np.dot(A, np.power(B, 2)) / N_reshaped
    # Variance computation
    # Variance = EXX - (Mean)^2
    # Note: This directly computes the variance without standardizing by the standard deviation
    cdef np.ndarray[np.float32_t, ndim=2] variance = EXX - M**2

    return variance


@cython.boundscheck(False)
@cython.wraparound(False)
def compute_neighborhood_score_by_zscore_cython(
    np.ndarray[np.float32_t, ndim=2] neighborhoods,
    np.ndarray[np.float32_t, ndim=2] annotation_matrix,
    ):
    # Convert to float32 for arithmetic operations, if not already
    cdef np.ndarray[np.float32_t, ndim=2] A = neighborhoods.astype(np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] B = annotation_matrix
    cdef np.ndarray[np.float32_t, ndim=2] neighborhood_score = np.dot(A, B)
    # Sum across rows for A to get N, reshape for broadcasting
    cdef np.ndarray[np.float32_t, ndim=1] N = np.sum(A, axis=1)
    cdef np.ndarray[np.float32_t, ndim=2] N_reshaped = N[:, None]
    # Mean of the dot product
    cdef np.ndarray[np.float32_t, ndim=2] M = neighborhood_score / N_reshaped
    # Compute squared mean and mean of squares
    cdef np.ndarray[np.float32_t, ndim=2] EXX = np.dot(A, np.power(B, 2)) / N_reshaped
    cdef np.ndarray[np.float32_t, ndim=2] EEX = M**2
    # Standard deviation
    cdef np.ndarray[np.float32_t, ndim=2] std = np.sqrt(EXX - EEX)
    
    # Z-score computation with error handling
    with np.errstate(divide='ignore', invalid='ignore'):
        neighborhood_score = np.divide(M, std)
        # Handle divisions by zero or std == 0
        neighborhood_score[np.isnan(neighborhood_score)] = 0  # Assuming requirement to reset NaN results to 0

    return neighborhood_score
