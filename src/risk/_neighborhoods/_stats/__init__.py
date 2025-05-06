"""
risk/_neighborhoods/_stats
~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from ._permutation import compute_permutation_test
from ._tests import (
    compute_binom_test,
    compute_chi2_test,
    compute_hypergeom_test,
    compute_poisson_test,
    compute_zscore_test,
)
