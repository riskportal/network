"""
tests/test_load_network
~~~~~~~~~~~~~~~~~~~~~~~
"""

def test_load_cytoscape_network(risk_obj, data_path):
    """Test loading a Cytoscape network from a .cys file

    Args:
        risk_obj: The RISK object instance used for loading the network
        data_path: The base path to the directory containing the Cytoscape file

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
    """Test loading a Cytoscape JSON network from a .cyjs file

    Args:
        risk_obj: The RISK object instance used for loading the network
        data_path: The base path to the directory containing the Cytoscape JSON file

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
    """Test loading a network from a .gpickle file

    Args:
        risk_obj: The RISK object instance used for loading the network
        data_path: The base path to the directory containing the gpickle file

    Returns:
        None
    """
    gpickle_file = data_path / "gpickle" / "michaelis_2023.gpickle"
    network = risk_obj.load_gpickle_network(filepath=str(gpickle_file))

    assert network is not None
    assert len(network.nodes) > 0  # Check that the network has nodes
    assert len(network.edges) > 0  # Check that the network has edges


def test_load_networkx_network(risk_obj, network):
    """Test loading a network from a NetworkX graph object

    Args:
        risk_obj: The RISK object instance used for loading the network
        network: The NetworkX graph object to be loaded into the RISK network

    Returns:
        None
    """
    network = risk_obj.load_networkx
