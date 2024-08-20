def test_load_cytoscape_network(risk_obj, data_path):
    cys_file = data_path / "cytoscape" / "michaelis_2023.cys"
    network = risk_obj.load_cytoscape_network(
        filepath=str(cys_file), source_label="source", target_label="target", view_name=""
    )

    assert network is not None
    assert len(network.nodes) > 0  # Check that the network has nodes
    assert len(network.edges) > 0  # Check that the network has edges


def test_load_cytoscape_json_network(risk_obj, data_path):
    cyjs_file = data_path / "cyjs" / "michaelis_2023.cyjs"
    network = risk_obj.load_cytoscape_json_network(
        filepath=str(cyjs_file), source_label="source", target_label="target"
    )

    assert network is not None
    assert len(network.nodes) > 0  # Check that the network has nodes
    assert len(network.edges) > 0  # Check that the network has edges


def test_load_gpickle_network(risk_obj, data_path):
    gpickle_file = data_path / "gpickle" / "michaelis_2023.gpickle"
    network = risk_obj.load_gpickle_network(filepath=str(gpickle_file))

    assert network is not None
    assert len(network.nodes) > 0  # Check that the network has nodes
    assert len(network.edges) > 0  # Check that the network has edges


def test_load_networkx_network(risk_obj, network):
    network = risk_obj.load_networkx_network(network=network)

    assert network is not None
    assert len(network.nodes) > 0  # Check that the graph has nodes
    assert len(network.edges) > 0  # Check that the graph has edges
    # Additional checks to verify the properties of the loaded graph
    for node in network.nodes:
        assert (
            node in network.nodes
        )  # Check that each node in the original network is in the RISK network

    for edge in network.edges:
        assert (
            edge in network.edges
        )  # Check that each edge in the original network is in the RISK network
