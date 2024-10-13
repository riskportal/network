"""
tests/conftest
~~~~~~~~~~~~~~
"""

import json
from pathlib import Path

import pytest

from risk import RISK

ROOT_PATH = Path(__file__).resolve().parent / "data"


@pytest.fixture(scope="session")
def data_path():
    """Fixture to provide the base path to the data directory.

    Returns:
        Path: The base path to the data directory.
    """
    return ROOT_PATH


@pytest.fixture(scope="session")
def risk():
    """Fixture to return the uninitialized RISK object.

    Returns:
        RISK: The uninitialized RISK object instance.
    """
    return RISK


@pytest.fixture(scope="session")
def risk_obj():
    """Fixture to initialize and return the RISK object.

    Returns:
        RISK: The initialized RISK object instance.
    """
    return RISK(verbose=False)


# Network fixtures
@pytest.fixture(scope="session")
def cytoscape_network(risk_obj, data_path):
    """Fixture to load and return a spherical network from a Cytoscape file.

    Args:
        risk_obj: The RISK object instance used for loading the network.
        data_path: The base path to the directory containing the Cytoscape file.

    Returns:
        Network: The loaded network object.
    """
    network_file = data_path / "cytoscape" / "michaelis_2023.cys"
    return risk_obj.load_cytoscape_network(
        filepath=str(network_file),
        source_label="source",
        target_label="target",
        view_name="",
        compute_sphere=True,
        surface_depth=0.1,
        min_edges_per_node=0,
        include_edge_weight=False,
        weight_label="weight",
    )


@pytest.fixture(scope="session")
def cytoscape_json_network(risk_obj, data_path):
    """Fixture to load and return the network from a Cytoscape JSON file.

    Args:
        risk_obj: The RISK object instance used for loading the network.
        data_path: The base path to the directory containing the Cytoscape JSON file.

    Returns:
        Network: The loaded network object.
    """
    network_file = data_path / "cyjs" / "michaelis_2023.cyjs"
    return risk_obj.load_cytoscape_json_network(
        filepath=str(network_file),
        source_label="source",
        target_label="target",
        compute_sphere=True,
        surface_depth=0.1,
        min_edges_per_node=0,
        include_edge_weight=False,
        weight_label="weight",
    )


@pytest.fixture(scope="session")
def gpickle_network(risk_obj, data_path):
    """Fixture to load and return the network from a GPickle file.

    Args:
        risk_obj: The RISK object instance used for loading the network.
        data_path: The base path to the directory containing the GPickle file.

    Returns:
        Network: The loaded network object.
    """
    network_file = data_path / "gpickle" / "michaelis_2023.gpickle"
    return risk_obj.load_gpickle_network(
        filepath=str(network_file),
        compute_sphere=True,
        surface_depth=0.1,
        min_edges_per_node=0,
        include_edge_weight=False,
        weight_label="weight",
    )


@pytest.fixture(scope="session")
def networkx_network(risk_obj, cytoscape_network):
    """Fixture to convert and return the network from a Cytoscape file as a NetworkX graph.

    Args:
        risk_obj: The RISK object instance used for loading the network.
        cytoscape_network: The network object loaded from a Cytoscape file.

    Returns:
        NetworkXGraph: The network object converted to a NetworkX graph.
    """
    return risk_obj.load_networkx_network(
        network=cytoscape_network,
        compute_sphere=True,
        surface_depth=0.1,
        min_edges_per_node=0,
        include_edge_weight=False,
        weight_label="weight",
    )


# Annotation fixtures
@pytest.fixture(scope="session")
def annotation_dict(data_path):
    """Fixture to load and return annotations from a JSON file as a dictionary.

    Args:
        data_path: The base path to the directory containing the annotation file.

    Returns:
        dict: The loaded annotations as a dictionary.
    """
    annotation_file = data_path / "json" / "annotations" / "go_biological_process.json"
    # Load the JSON file and return as a dictionary
    with open(annotation_file, "r") as file:
        annotation_dict = json.load(file)

    return annotation_dict


@pytest.fixture(scope="session")
def json_annotation(risk_obj, cytoscape_network, data_path):
    """Fixture to load and return annotations from a JSON file.

    Args:
        risk_obj: The RISK object instance used for loading annotations.
        cytoscape_network: The network object to which annotations will be applied.
        data_path: The base path to the directory containing the annotation file.

    Returns:
        Annotations: The loaded annotations object.
    """
    annotation_file = data_path / "json" / "annotations" / "go_biological_process.json"
    return risk_obj.load_json_annotation(filepath=str(annotation_file), network=cytoscape_network)


@pytest.fixture(scope="session")
def dict_annotation(risk_obj, cytoscape_network):
    """Load and return annotations from a dictionary.

    Args:
        risk_obj: The RISK object instance for loading annotations.
        cytoscape_network: The network to which the annotations will be applied.

    Returns:
        dict: The loaded annotations object.
    """
    annotation_content = {
        "phosphatidylinositol dephosphorylation": [
            "IST2",
            "SCS2",
            "SCS22",
            "TCB1",
            "TCB2",
            "TCB3",
            "VPS74",
        ],
        "proline catabolic process": ["MPR1", "PUT1", "PUT2", "PUT3"],
        "chromosome attachment to the nuclear envelope": [
            "MMS21",
            "NDJ1",
            "NFI1",
            "RTT107",
            "SIZ1",
            "SMC6",
        ],
    }
    return risk_obj.load_dict_annotation(content=annotation_content, network=cytoscape_network)


@pytest.fixture(scope="session")
def csv_annotation(risk_obj, cytoscape_network, data_path):
    """Fixture to load and return annotations from a CSV file.

    Args:
        risk_obj: The RISK object instance used for loading annotations.
        cytoscape_network: The network object to which annotations will be applied.
        data_path: The base path to the directory containing the annotation file.

    Returns:
        Annotations: The loaded annotations object.
    """
    annotation_file = data_path / "csv" / "annotations" / "go_biological_process.csv"
    return risk_obj.load_csv_annotation(filepath=str(annotation_file), network=cytoscape_network)


@pytest.fixture(scope="session")
def tsv_annotation(risk_obj, cytoscape_network, data_path):
    """Fixture to load and return annotations from a TSV file.

    Args:
        risk_obj: The RISK object instance used for loading annotations.
        cytoscape_network: The network object to which annotations will be applied.
        data_path: The base path to the directory containing the annotation file.

    Returns:
        Annotations: The loaded annotations object.
    """
    annotation_file = data_path / "tsv" / "annotations" / "go_biological_process.tsv"
    return risk_obj.load_tsv_annotation(filepath=str(annotation_file), network=cytoscape_network)


@pytest.fixture(scope="session")
def excel_annotation(risk_obj, cytoscape_network, data_path):
    """Fixture to load and return annotations from an Excel file.

    Args:
        risk_obj: The RISK object instance used for loading annotations.
        cytoscape_network: The network object to which annotations will be applied.
        data_path: The base path to the directory containing the annotation file.

    Returns:
        Annotations: The loaded annotations object.
    """
    annotation_file = data_path / "excel" / "annotations" / "go_biological_process.xlsx"
    return risk_obj.load_excel_annotation(filepath=str(annotation_file), network=cytoscape_network)


# Combined fixture for testing graph loading
@pytest.fixture(scope="session")
def graph(risk_obj, cytoscape_network, json_annotation):
    """Fixture to load and return a graph built from a Cytoscape JSON network and annotations.

    Args:
        risk_obj: The RISK object instance used for loading the graph.
        cytoscape_network: The network object loaded from a Cytoscape file.
        json_annotation: The JSON annotations associated with the network.

    Returns:
        Graph: The constructed graph object.
    """
    network = cytoscape_network
    annotations = json_annotation
    # Build neighborhoods based on the loaded network and annotations
    neighborhoods = risk_obj.load_neighborhoods_by_permutation(
        network=network,
        annotations=annotations,
        distance_metric="louvain",  # Example metric
        louvain_resolution=8,  # Example resolution
        edge_length_threshold=0.75,
        score_metric="stdev",
        null_distribution="network",
        num_permutations=100,  # Number of permutations
        random_seed=887,
        max_workers=4,
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
