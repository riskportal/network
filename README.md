# RISK Network

<p align="center">
  <img src="https://i.imgur.com/8TleEJs.png" width="50%" />
</p>

<br>

![Python](https://img.shields.io/badge/python-3.8%2B-yellow)
[![pypiv](https://img.shields.io/pypi/v/risk-network.svg)](https://pypi.python.org/pypi/risk-network)
![License](https://img.shields.io/badge/license-GPLv3-purple)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.xxxxxxx.svg)](https://doi.org/10.5281/zenodo.xxxxxxx)
![Downloads](https://img.shields.io/pypi/dm/risk-network)
![Tests](https://github.com/riskportal/network/actions/workflows/ci.yml/badge.svg)

**RISK** (Regional Inference of Significant Kinships) is a next-generation tool for biological network annotation and visualization. It integrates community detection algorithms, rigorous overrepresentation analysis, and a modular framework for diverse network types. RISK identifies biologically coherent relationships within networks and generates publication-ready visualizations, making it a useful tool for biological and interdisciplinary network analysis.

For a full description of RISK and its applications, see:
<br>
**Horecka and Röst (2025)**, _"RISK: a next-generation tool for biological network annotation and visualization"_.
<br>
DOI: [10.5281/zenodo.xxxxxxx](https://doi.org/10.5281/zenodo.xxxxxxx)

## Documentation and Tutorial

Full documentation is available at:

- **Docs:** [https://riskportal.github.io/network-tutorial](https://riskportal.github.io/network-tutorial)
- **Tutorial Jupyter Notebook Repository:** [https://github.com/riskportal/network-tutorial](https://github.com/riskportal/network-tutorial)

## Installation

RISK is compatible with Python 3.8 or later and runs on all major operating systems. To install the latest version of RISK, run:

```bash
pip install risk-network --upgrade
```

## Key Features of RISK

- **Broad Data Compatibility**: Accepts multiple network formats (NetworkX, Cytoscape, GPickle) and user-provided annotations formatted as term–to–gene membership tables (JSON, CSV, TSV, Excel, or Python dictionaries).
- **Flexible Clustering**: Offers Louvain, Leiden, Markov Clustering, Greedy Modularity, Label Propagation, Spinglass, and Walktrap, with user-defined resolution parameters to detect both coarse and fine-grained modules.
- **Statistical Testing**: Provides hypergeometric, chi-squared, binomial, and permutation tests, balancing speed with statistical rigor.
- **High-Resolution Visualization**: Generates publication-ready figures with contour overlays, customizable node/edge properties, and export to SVG, PNG, or PDF.

## Example Usage

We applied RISK to a _Saccharomyces cerevisiae_ protein–protein interaction (PPI) network (Michaelis _et al_., 2023; 3,839 proteins, 30,955 interactions). RISK identified compact, functional modules overrepresented in Gene Ontology Biological Process (GO BP) terms (Ashburner _et al_., 2000), revealing biological organization including ribosomal assembly, mitochondrial organization, and RNA polymerase activity (P < 0.0001).

[![RISK analysis of the yeast PPI network](https://i.imgur.com/fSNf5Ad.jpeg)](https://i.imgur.com/fSNf5Ad.jpeg)
**RISK workflow overview and analysis of the yeast PPI network**. GO BP terms are color-coded to represent key cellular processes—including ribosomal assembly, mitochondrial organization, and RNA polymerase activity (P < 0.0001).

## Citation

If you use RISK in your research, please cite the following:

**Horecka and Röst (2025)**, _"RISK: a next-generation tool for biological network annotation and visualization"_.
<br>
DOI: [10.5281/zenodo.xxxxxxx](https://doi.org/10.5281/zenodo.xxxxxxx)

## Contributing

We welcome contributions from the community:

- [Issues Tracker](https://github.com/riskportal/network/issues)
- [Source Code](https://github.com/riskportal/network/tree/main/risk)

## Support

If you encounter issues or have suggestions for new features, please use the [Issues Tracker](https://github.com/riskportal/network/issues) on GitHub.

## License

RISK is open source under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html).
