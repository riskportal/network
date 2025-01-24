"""
risk/stats
~~~~~~~~~~
"""

from .binom import compute_binom_test
from .chi2 import compute_chi2_test
from .hypergeom import compute_hypergeom_test
from .permutation import compute_permutation_test
from .poisson import compute_poisson_test
from .zscore import compute_zscore_test

from .stats import calculate_significance_matrices
