# cython: language_level=3
import numpy as np
cimport numpy as np
cimport cython
from threadpoolctl import threadpool_limits


@cython.boundscheck(False)  # Disable bounds checking for entire function
@cython.wraparound(False)   # Disable negative index wrapping for entire function
def compute_neighborhood_score_by_sum_cython(
    np.ndarray[np.float32_t, ndim=2] neighborhoods,
    np.ndarray[np.float32_t, ndim=2] annotation_matrix,
    ):
    cdef np.float32_t[:, :] neighborhood_score
    # Limit the number of threads used by np.dot
    with threadpool_limits(limits=1, user_api='blas'):
        neighborhood_score = np.dot(neighborhoods, annotation_matrix)
    
    return np.asarray(neighborhood_score)


@cython.boundscheck(False)
@cython.wraparound(False)
def compute_neighborhood_score_by_stdev_cython(
    np.ndarray[np.float32_t, ndim=2] neighborhoods,
    np.ndarray[np.float32_t, ndim=2] annotation_matrix,
    ):
    cdef np.ndarray[np.float32_t, ndim=2] neighborhood_score
    cdef np.ndarray[np.float32_t, ndim=2] EXX
    # Perform dot product directly using the inputs with limited threads
    with threadpool_limits(limits=1, user_api='blas'):
        neighborhood_score = np.dot(neighborhoods, annotation_matrix)
    
    # Sum across rows for neighborhoods to get N, reshape for broadcasting
    cdef np.ndarray[np.float32_t, ndim=1] N = np.sum(neighborhoods, axis=1)
    cdef np.ndarray[np.float32_t, ndim=2] N_reshaped = N[:, None]
    # Mean of the dot product
    cdef np.ndarray[np.float32_t, ndim=2] M = neighborhood_score / N_reshaped
    # Compute the mean of squares (EXX) with limited threads
    with threadpool_limits(limits=1, user_api='blas'):
        EXX = np.dot(neighborhoods, np.power(annotation_matrix, 2)) / N_reshaped
    
    # Variance computation
    cdef np.ndarray[np.float32_t, ndim=2] variance = EXX - M**2
    # Standard deviation computation
    cdef np.ndarray[np.float32_t, ndim=2] stdev = np.sqrt(variance)
    
    return stdev


@cython.boundscheck(False)
@cython.wraparound(False)
def compute_neighborhood_score_by_z_score_cython(
    np.ndarray[np.float32_t, ndim=2] neighborhoods,
    np.ndarray[np.float32_t, ndim=2] annotation_matrix,
    ):
    cdef np.ndarray[np.float32_t, ndim=2] neighborhood_score
    cdef np.ndarray[np.float32_t, ndim=2] EXX
    # Perform dot product directly using the inputs with limited threads
    with threadpool_limits(limits=1, user_api='blas'):
        neighborhood_score = np.dot(neighborhoods, annotation_matrix)
    
    # Sum across rows for neighborhoods to get N, reshape for broadcasting
    cdef np.ndarray[np.float32_t, ndim=1] N = np.sum(neighborhoods, axis=1)
    cdef np.ndarray[np.float32_t, ndim=2] N_reshaped = N[:, None]
    # Mean of the dot product
    cdef np.ndarray[np.float32_t, ndim=2] M = neighborhood_score / N_reshaped
    # Compute the mean of squares (EXX) with limited threads
    with threadpool_limits(limits=1, user_api='blas'):
        EXX = np.dot(neighborhoods, np.power(annotation_matrix, 2)) / N_reshaped
    
    # Variance computation
    cdef np.ndarray[np.float32_t, ndim=2] variance = EXX - M**2
    # Standard deviation computation
    cdef np.ndarray[np.float32_t, ndim=2] stdev = np.sqrt(variance)
    # Z-score computation with error handling
    with np.errstate(divide='ignore', invalid='ignore'):
        neighborhood_score = np.divide(M, stdev)
        # Handle divisions by zero or stdev == 0
        neighborhood_score[np.isnan(neighborhood_score)] = 0  # Assuming requirement to reset NaN results to 0

    return neighborhood_score
