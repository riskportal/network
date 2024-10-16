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

**RISK (RISK Infers Spatial Kinships)** is a next-generation tool designed to streamline the analysis of biological and non-biological networks. RISK enhances network analysis with its modular architecture, extensive file format support, and advanced clustering algorithms. It simplifies the creation of publication-quality figures, making it an important tool for researchers across disciplines.

## Documentation and Tutorial

- **Documentation**: Comprehensive documentation is available [here](Documentation link).
- **Tutorial**: An interactive Jupyter notebook tutorial can be found [here](https://github.com/riskportal/network-tutorial).
We highly recommend new users to consult the documentation and tutorial early on to fully leverage RISK's capabilities.

## Installation

RISK is compatible with Python 3.8 and later versions and operates on all major operating systems. Install RISK via pip:

```bash
pip install risk-network
```

## Features

- **Comprehensive Network Analysis**: Analyze biological networks such as protein–protein interaction (PPI) and gene regulatory networks, as well as non-biological networks.
- **Advanced Clustering Algorithms**: Utilize algorithms like Louvain, Markov Clustering, Spinglass, and more to identify key functional modules.
- **Flexible Visualization**: Generate clear, publication-quality figures with customizable node and edge attributes, including colors, shapes, sizes, and labels.
- **Efficient Data Handling**: Optimized for large datasets, supporting multiple file formats such as JSON, CSV, TSV, Excel, Cytoscape, and GPickle.
- **Statistical Analysis**: Integrated statistical tests, including hypergeometric, permutation, and Poisson tests, to assess the significance of enriched regions.
- **Cross-Domain Applicability**: Suitable for network analysis across biological and non-biological domains, including social and communication networks.

## Example Usage

We applied RISK to a *Saccharomyces cerevisiae* protein–protein interaction network, revealing both established and novel functional relationships. The visualization below highlights key biological processes such as ribosomal assembly and mitochondrial organization.

![RISK Main Figure](https://i.imgur.com/5OP3Hqe.jpeg)

RISK successfully detected both known and novel functional clusters within the yeast interactome. Clusters related to Golgi transport and actin nucleation were clearly defined and closely located, showcasing RISK's ability to map well-characterized interactions. Additionally, RISK identified links between mRNA processing pathways and vesicle trafficking proteins, consistent with recent studies demonstrating the role of vesicles in mRNA localization and stability.

## Citation

If you use RISK in your research, please cite the following:

**Horecka**, *et al.*, "RISK: a next-generation tool for biological network annotation and visualization", **[Journal Name]**, 2024. DOI: [10.1234/zenodo.xxxxxxx](https://doi.org/10.1234/zenodo.xxxxxxx)

## Software Architecture and Implementation

RISK features a streamlined, modular architecture designed to meet diverse research needs. Each module focuses on a specific task—such as network input/output, statistical analysis, or visualization—ensuring ease of adaptation and extension. This design enhances flexibility and reduces development overhead for users integrating RISK into their workflows.

### Supported Data Formats

- **Input/Output**: JSON, CSV, TSV, Excel, Cytoscape, GPickle.
- **Visualization Outputs**: SVG, PNG, PDF.

### Clustering Algorithms

- **Available Algorithms**:
  - Greedy Modularity
  - Label Propagation
  - Louvain
  - Markov Clustering
  - Spinglass
  - Walktrap
- **Distance Metrics**: Supports both spherical and Euclidean distance metrics.

### Statistical Tests

- **Hypergeometric Test**
- **Permutation Test** (single- or multi-process modes)
- **Poisson Test**

## Performance and Efficiency

In benchmarking tests using the yeast interactome network, RISK demonstrated substantial improvements over previous tools in both computational performance and memory efficiency. RISK processed the dataset approximately **3.25 times faster**, reducing CPU time by **69%**, and required **25% less peak memory usage**, underscoring its efficient utilization of computational resources.

## Contributing

We welcome contributions from the community. Please use the following resources:

- [Issues Tracker](https://github.com/irahorecka/risk/issues)
- [Source Code](https://github.com/irahorecka/risk/tree/main/risk)

## Support

If you encounter issues or have suggestions for new features, please use the [Issues Tracker](https://github.com/irahorecka/risk/issues) on GitHub.

## License

RISK is freely available as open-source software under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html).

---

**Note**: For detailed documentation and to access the interactive tutorial, please visit the links provided in the [Documentation and Tutorial](#documentation-and-tutorial) section.
