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

**RISK** (Regional Inference of Significant Kinships) is a next-generation tool for biological network annotation and visualization. RISK integrates community detection-based clustering, rigorous statistical enrichment analysis, and a modular framework to uncover biologically meaningful relationships and generate high-resolution visualizations. RISK supports diverse data formats and is optimized for large-scale network analysis, making it a valuable resource for researchers in systems biology and beyond.

## Documentation and Tutorial

Full documentation is available at:

- **Docs:** [https://riskportal.github.io/network-tutorial](https://riskportal.github.io/network-tutorial)  
- **Tutorial Jupyter Notebook Repository:** [https://github.com/riskportal/network-tutorial](https://github.com/riskportal/network-tutorial)

## Installation

RISK is compatible with Python 3.8 or later and runs on all major operating systems. To install the latest version of RISK, run:

```bash
pip install risk-network --upgrade
```

## Features

- **Comprehensive Network Analysis**: Analyze biological networks (e.g., protein–protein interaction and genetic interaction networks) as well as non-biological networks.
- **Advanced Clustering Algorithms**: Supports Louvain, Leiden, Markov Clustering, Greedy Modularity, Label Propagation, Spinglass, and Walktrap for identifying structured network regions.
- **Flexible Visualization**: Produce customizable, high-resolution network visualizations with kernel density estimate overlays, adjustable node and edge attributes, and export options in SVG, PNG, and PDF formats.
- **Efficient Data Handling**: Supports multiple input/output formats, including JSON, CSV, TSV, Excel, Cytoscape, and GPickle.
- **Statistical Analysis**: Assess functional enrichment using hypergeometric, permutation (network-aware), binomial, chi-squared, Poisson, and z-score tests, ensuring statistical adaptability across datasets.
- **Cross-Domain Applicability**: Suitable for network analysis across biological and non-biological domains, including social and communication networks.

## Example Usage

We applied RISK to a *Saccharomyces cerevisiae* protein–protein interaction network from Michaelis et al. (2023), filtering for proteins with six or more interactions to emphasize core functional relationships. RISK identified compact, statistically enriched clusters corresponding to biological processes such as ribosomal assembly and mitochondrial organization.

[![Figure 1](https://i.imgur.com/lJHJrJr.jpeg)](https://i.imgur.com/lJHJrJr.jpeg)

This figure highlights RISK’s capability to detect both established and novel functional modules within the yeast interactome.

## Citation

If you use RISK in your research, please reference the following:

**Horecka et al.**, *"RISK: a next-generation tool for biological network annotation and visualization"*, 2025.  
DOI: [10.1234/zenodo.xxxxxxx](https://doi.org/10.1234/zenodo.xxxxxxx)

## Software Architecture and Implementation

RISK features a streamlined, modular architecture designed to meet diverse research needs. RISK’s modular design enables users to run individual components—such as clustering, statistical testing, or visualization—independently or in combination, depending on the analysis workflow. It includes dedicated modules for:

- **Data I/O**: Supports JSON, CSV, TSV, Excel, Cytoscape, and GPickle formats.
- **Clustering**: Supports multiple clustering methods, including Louvain, Leiden, Markov Clustering, Greedy Modularity, Label Propagation, Spinglass, and Walktrap. Provides flexible distance metrics tailored to network structure.
- **Statistical Analysis**: Provides a suite of tests for overrepresentation analysis of annotations.
- **Visualization**: Offers customizable, high-resolution output in multiple formats, including SVG, PNG, and PDF.
- **Configuration Management**: Centralized parameters in risk.params ensure reproducibility and easy tuning for large-scale analyses.

## Performance and Efficiency

Benchmarking results demonstrate that RISK efficiently scales to networks exceeding hundreds of thousands of edges, maintaining low execution times and optimal memory usage across statistical tests.

## Contributing

We welcome contributions from the community:

- [Issues Tracker](https://github.com/riskportal/network/issues)
- [Source Code](https://github.com/riskportal/network/tree/main/risk)

## Support

If you encounter issues or have suggestions for new features, please use the [Issues Tracker](https://github.com/riskportal/network/issues) on GitHub.

## License

RISK is open source under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html).
