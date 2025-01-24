"""
risk/stats
~~~~~~~~~~
"""

from risk.stats.binom import compute_binom_test
from risk.stats.chi2 import compute_chi2_test
from risk.stats.hypergeom import compute_hypergeom_test
from risk.stats.permutation import compute_permutation_test
from risk.stats.poisson import compute_poisson_test
from risk.stats.zscore import compute_zscore_test

from risk.stats.stats import calculate_significance_matrices
