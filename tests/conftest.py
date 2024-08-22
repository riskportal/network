"""
tests/conftest
~~~~~~~~~~~~~~
"""

import pytest
from pathlib import Path
from risk import RISK

ROOT_PATH = Path(__file__).resolve().parent / "data"


@pytest.fixture(scope="session")
def data_path():
    """Fixture to provide the base path to the data directory

    Returns:
        Path: The base path to the data directory
    """
    return ROOT_PATH


@pytest.fixture(scope="session")
def risk_obj():
    """Fixture to initialize and return the RISK object

    Returns:
        RISK: The initialized RISK object instance
    """
    return RISK(
        compute_sphere=True,
        surface_depth=0.1,
        min_edges_per_node=0,
        include_edge_weight=False,
        weight_label="weight",
    )


@pytest.fixture(scope="session")
def network(risk_obj, data_path):
    """Fixture to load and return the network from a Cytoscape file

    Args:
        risk_obj: The RISK object instance used for loading the network
        data_path: The base path to the data directory

    Returns:
        Network: The loaded network object
    """
    network_file = data_path / "cytoscape" / "michaelis_2023.cys"
    network = risk_obj.load_cytoscape_network(
        filepath=str(network_file),
        source_label="source",
        target_label="target",
        view_name="",
    )
    return network


@pytest.fixture(scope="session")
def annotations(risk_obj, network, data_path):
    """Fixture to load and return annotations from a JSON file

    Args:
        risk_obj: The RISK object instance used for loading annotations
        network: The network object to which annotations will be applied
        data_path: The base path to the data directory

    Returns:
        Annotations: The loaded annotations object
    """
    annotation_file = data_path / "json" / "annotations" / "go_biological_process.json"
    annotations = risk_obj.load_json_annotation(filepath=str(annotation_file), network=network)
    return annotations


@pytest.fixture(scope="session")
def graph(risk_obj, data_path):
    """Fixture to load and return a graph built from a Cytoscape JSON network and annotations

    Args:
        risk_obj: The RISK object instance used for loading the graph
        data_path: The base path to the data directory

    Returns:
        Graph: The constructed graph object
    """
    network_filepath = data_path / "cyjs" / "michaelis_2023.cyjs"
    annotation_filepath = data_path / "json" / "annotations" / "go_biological_process.json"

    # Load network from the Cytoscape JSON file
    network = risk_obj.load_cytoscape_json_network(
        filepath=str(network_filepath), source_label="source", target_label="target"
    )

    # Load annotations associated with the network
    annotations = risk_obj.load_json_annotation(filepath=str(annotation_filepath), network=network)

    # Build neighborhoods based on the loaded network and annotations
    neighborhoods = risk_obj.load_neighborhoods(
        network=network,
        annotations=annotations,
        distance_metric="louvain",  # Example metric
        louvain_resolution=8,  # Example resolution
        edge_length_threshold=0.75,
        score_metric="stdev",
        null_distribution="network",
        num_permutations=100,  # Number of permutations
        random_seed=887,
        max_workers=4,  # Use 4 processes
    )

    # Build the graph using the neighborhoods
    graph = risk_obj.load_graph(
        network=network,
        annotations=annotations,
        neighborhoods=neighborhoods,
        tail="right",  # Example parameter
        pval_cutoff=0.05,
        fdr_cutoff=1.0,
        impute_depth=1,
        prune_threshold=0.1,
        linkage_criterion="distance",
        linkage_method="average",
        linkage_metric="yule",
        min_cluster_size=5,
        max_cluster_size=1000,
    )
    return graph
