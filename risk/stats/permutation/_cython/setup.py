"""
risk/stats/permutation/_cython/setup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

# setup.py
from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(ext_modules=cythonize("permutation.pyx"), include_dirs=[np.get_include()])
