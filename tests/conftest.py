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


# Network fixtures
@pytest.fixture(scope="session")
def cytoscape_network(risk_obj, data_path):
    """Fixture to load and return the network from a Cytoscape file"""
    network_file = data_path / "cytoscape" / "michaelis_2023.cys"
    return risk_obj.load_cytoscape_network(
        filepath=str(network_file),
        source_label="source",
        target_label="target",
        view_name="",
    )


@pytest.fixture(scope="session")
def cytoscape_json_network(risk_obj, data_path):
    """Fixture to load and return the network from a Cytoscape JSON file"""
    network_file = data_path / "cyjs" / "michaelis_2023.cyjs"
    return risk_obj.load_cytoscape_json_network(
        filepath=str(network_file),
        source_label="source",
        target_label="target",
    )


@pytest.fixture(scope="session")
def gpickle_network(risk_obj, data_path):
    """Fixture to load and return the network from a GPickle file"""
    network_file = data_path / "gpickle" / "michaelis_2023.gpickle"
    return risk_obj.load_gpickle_network(filepath=str(network_file))


@pytest.fixture(scope="session")
def networkx_network(risk_obj, cytoscape_network):
    """Fixture to convert and return the network from a Cytoscape file as a NetworkX graph"""
    # Here, cytoscape_network is already a loaded network using the Cytoscape loader
    # Convert the Cytoscape network (or just reuse it if already a NetworkX graph)
    return risk_obj.load_networkx_network(network=cytoscape_network)


# Annotation fixtures
@pytest.fixture(scope="session")
def json_annotation(risk_obj, cytoscape_network, data_path):
    """Fixture to load and return annotations from a JSON file"""
    annotation_file = data_path / "json" / "annotations" / "go_biological_process.json"
    return risk_obj.load_json_annotation(filepath=str(annotation_file), network=cytoscape_network)


@pytest.fixture(scope="session")
def csv_annotation(risk_obj, cytoscape_network, data_path):
    """Fixture to load and return annotations from a CSV file"""
    annotation_file = data_path / "csv" / "annotations" / "go_biological_process.csv"
    return risk_obj.load_csv_annotation(filepath=str(annotation_file), network=cytoscape_network)


@pytest.fixture(scope="session")
def tsv_annotation(risk_obj, cytoscape_network, data_path):
    """Fixture to load and return annotations from a TSV file"""
    annotation_file = data_path / "tsv" / "annotations" / "go_biological_process.tsv"
    return risk_obj.load_tsv_annotation(filepath=str(annotation_file), network=cytoscape_network)


@pytest.fixture(scope="session")
def excel_annotation(risk_obj, cytoscape_network, data_path):
    """Fixture to load and return annotations from an Excel file"""
    annotation_file = data_path / "excel" / "annotations" / "go_biological_process.xlsx"
    return risk_obj.load_excel_annotation(filepath=str(annotation_file), network=cytoscape_network)


# Combined fixture for testing graph loading
@pytest.fixture(scope="session")
def graph(risk_obj, cytoscape_network, json_annotation):
    """Fixture to load and return a graph built from a Cytoscape JSON network and annotations"""
    # Using the Cytoscape JSON network and JSON annotations as default for the graph fixture
    network = cytoscape_network
    annotations = json_annotation
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
