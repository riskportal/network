"""
risk/stats/permutation
~~~~~~~~~~~~~~~~~~~~~~
"""

from ._cython.permutation import (
    compute_neighborhood_score_by_sum_cython,
    compute_neighborhood_score_by_variance_cython,
    compute_neighborhood_score_by_z_score_cython,
)
from ._python.permutation import (
    compute_neighborhood_score_by_sum_python,
    compute_neighborhood_score_by_variance_python,
    compute_neighborhood_score_by_z_score_python,
)
