from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize

import numpy

from .risk import __version__


extensions = [
    Extension(
        name="risk.stats.permutation._cython.permutation",
        sources=["risk/stats/permutation/_cython/permutation.pyx"],
        include_dirs=[numpy.get_include()],
    ),
]

setup(
    name="risk-network",
    version=__version__,  # Updated version
    author="Ira Horecka",
    author_email="ira89@icloud.com",
    description="A Python package for biological network analysis",  # Updated description
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="GPL-3.0-or-later",
    packages=find_packages(),
    ext_modules=cythonize(extensions),
    include_package_data=True,
    install_requires=[
        "cython",
        "numpy",
        "ipywidgets",
        "markov_clustering",
        "matplotlib",
        "networkx",
        "nltk==3.8.1",
        "pandas",
        "python-louvain",
        "scikit-learn",
        "scipy",
        "statsmodels",
        "threadpoolctl",
        "tqdm",
    ],
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Development Status :: 4 - Beta",
    ],
    python_requires=">=3.7",
)
