"""
tests/test_load_network
~~~~~~~~~~~~~~~~~~~~~~~
"""

import pytest


@pytest.mark.parametrize("verbose_setting", [True, False])
def test_initialize_risk(risk, verbose_setting):
    """Test RISK instance initialization with verbose parameter.

    Args:
        verbose_setting: Boolean value to set verbosity of the RISK instance.

    Returns:
        None
    """
    try:
        risk_instance = risk(verbose=verbose_setting)
        assert risk_instance is not None
    except Exception:
        pytest.fail(f"RISK failed to initialize with verbose={verbose_setting}")


def test_load_cytoscape_network(risk_obj, data_path):
    """Test loading a Cytoscape network from a .cys file.

    Args:
        risk_obj: The RISK object instance used for loading the network.
        data_path: The base path to the directory containing the Cytoscape file.

    Returns:
        None
    """
    cys_file = data_path / "cytoscape" / "michaelis_2023.cys"
    network = risk_obj.load_cytoscape_network(
        filepath=str(cys_file), source_label="source", target_label="target", view_name=""
    )

    assert network is not None
    assert len(network.nodes) > 0  # Check that the network has nodes
    assert len(network.edges) > 0  # Check that the network has edges


def test_load_cytoscape_json_network(risk_obj, data_path):
    """Test loading a Cytoscape JSON network from a .cyjs file.

    Args:
        risk_obj: The RISK object instance used for loading the network.
        data_path: The base path to the directory containing the Cytoscape JSON file.

    Returns:
        None
    """
    cyjs_file = data_path / "cyjs" / "michaelis_2023.cyjs"
    network = risk_obj.load_cytoscape_json_network(
        filepath=str(cyjs_file), source_label="source", target_label="target"
    )

    assert network is not None
    assert len(network.nodes) > 0  # Check that the network has nodes
    assert len(network.edges) > 0  # Check that the network has edges


def test_load_gpickle_network(risk_obj, data_path):
    """Test loading a network from a .gpickle file.

    Args:
        risk_obj: The RISK object instance used for loading the network.
        data_path: The base path to the directory containing the gpickle file.

    Returns:
        None
    """
    gpickle_file = data_path / "gpickle" / "michaelis_2023.gpickle"
    network = risk_obj.load_gpickle_network(filepath=str(gpickle_file))

    assert network is not None
    assert len(network.nodes) > 0  # Check that the network has nodes
    assert len(network.edges) > 0  # Check that the network has edges


def test_load_networkx_network(risk_obj, cytoscape_network):
    """Test loading a network from a NetworkX graph object.

    Args:
        risk_obj: The RISK object instance used for loading the network.
        network: The NetworkX graph object to be loaded into the RISK network.

    Returns:
        None
    """
    network = risk_obj.load_networkx_network(network=cytoscape_network)

    assert network is not None
    assert len(network.nodes) > 0  # Check that the graph has nodes
    assert len(network.edges) > 0  # Check that the graph has edges
    # Additional checks to verify the properties of the loaded graph
    for node in network.nodes:
        # Check that each node in the original network is in the RISK network
        assert node in network.nodes

    for edge in network.edges:
        # Check that each edge in the original network is in the RISK network
        assert edge in network.edges


@pytest.mark.parametrize("min_edges", [1, 5, 10])
def test_load_network_min_edges(risk_obj, data_path, min_edges):
    """Test loading a Cytoscape network with varying min_edges_per_node.

    Args:
        risk_obj: The RISK object instance used for loading the network.
        data_path: The base path to the directory containing the Cytoscape file.
        min_edges: The minimum number of edges per node to test.

    Returns:
        None
    """
    cys_file = data_path / "cytoscape" / "michaelis_2023.cys"
    network = risk_obj.load_cytoscape_network(
        filepath=str(cys_file),
        source_label="source",
        target_label="target",
        include_edge_weight=False,
        weight_label="weight",
        min_edges_per_node=min_edges,
        compute_sphere=True,
        surface_depth=0.5,
    )

    # Check that each node has at least min_edges
    for node in network.nodes:
        assert network.degree[node] >= min_edges, f"Node {node} has fewer than {min_edges} edges"
