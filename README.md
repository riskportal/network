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
![Platforms](https://img.shields.io/badge/platform-linux%20%7C%20macos%20%7C%20windows-lightgrey)

**RISK** (Regional Inference of Significant Kinships) is a next-generation tool for biological network annotation and visualization. It leverages advanced clustering algorithms, robust statistical frameworks, and a modular design to identify biologically meaningful relationships and generate publication-quality visualizations. RISK supports diverse data formats and is optimized for large-scale network analysis, making it a valuable resource for researchers in systems biology and beyond.

## Documentation and Tutorial

An interactive Jupyter notebook tutorial can be found [here](https://github.com/riskportal/network-tutorial). We highly recommend new users to consult the documentation and tutorial early on to fully utilize RISK's capabilities.

## Installation

RISK is compatible with Python 3.8 or later and runs on all major operating systems. To install the latest version of RISK, run:

```bash
pip install risk-network --upgrade
```

## Features

- **Comprehensive Network Analysis**: Analyze biological networks (e.g., protein–protein interaction and genetic interaction networks) as well as non-biological networks.
- **Advanced Clustering Algorithms**: Utilize algorithms such as Louvain, Leiden, and Markov Clustering to detect functional modules.
- **Flexible Visualization**: Generate high-resolution, publication-quality figures with customizable node and edge attributes.
- **Efficient Data Handling**: Supports multiple input/output formats, including JSON, CSV, TSV, Excel, Cytoscape, and GPickle.
- **Statistical Analysis**: Perform overrepresentation analysis of annotations (e.g., Gene Ontology Biological Process (GO BP) terms) using statistical tests including hypergeometric, permutation, chi-squared, binomial, Poisson, and z-score.
- **Cross-Domain Applicability**: Suitable for network analysis across biological and non-biological domains, including social and communication networks.

## Example Usage

We applied RISK to a *Saccharomyces cerevisiae* protein–protein interaction network from Michaelis et al., 2023 to reveal functional modules related to biological processes such as ribosomal assembly and mitochondrial organization.

[![Figure 1](https://i.imgur.com/lJHJrJr.jpeg)](https://i.imgur.com/lJHJrJr.jpeg)

This figure highlights RISK’s capability to detect both established and novel functional modules within the yeast interactome.

## Citation

If you use RISK in your research, please cite:

**Horecka & Röst**, "RISK: a next-generation tool for biological network annotation and visualization", **Bioinformatics**, 2025. DOI: [10.1234/zenodo.xxxxxxx](https://doi.org/10.1234/zenodo.xxxxxxx)

## Software Architecture and Implementation

RISK features a streamlined, modular architecture designed to meet diverse research needs. It includes dedicated modules for:

- **Data I/O**: Supports JSON, CSV, TSV, Excel, Cytoscape, and GPickle formats.
- **Clustering**: Implements multiple algorithms (e.g., Louvain, Markov Clustering) and supports both spherical and Euclidean distance metrics.
- **Statistical Analysis**: Provides a suite of tests for overrepresentation analysis of annotations.
- **Visualization**: Offers customizable, high-resolution output in multiple formats, including SVG, PNG, and PDF.

## Performance and Efficiency

Benchmarking tests have shown that RISK is both computationally efficient and scalable, maintaining low execution times and memory usage even on large networks.

## Contributing

We welcome contributions from the community:

- [Issues Tracker](https://github.com/irahorecka/risk/issues)
- [Source Code](https://github.com/irahorecka/risk/tree/main/risk)

## Support

If you encounter issues or have suggestions for new features, please use the [Issues Tracker](https://github.com/irahorecka/risk/issues) on GitHub.

## License

RISK is open source under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html).

---

**Note**: For detailed documentation and to access the interactive tutorial, please visit the links above.
