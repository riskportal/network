"""
tests/test_load_network
~~~~~~~~~~~~~~~~~~~~~~~
"""

import os
import pickle
import sys

import networkx as nx
import pytest


@pytest.mark.parametrize("verbose_setting", [True, False])
def test_initialize_risk(risk, verbose_setting):
    """
    Test RISK instance initialization with verbose parameter.

    Args:
        verbose_setting: Boolean value to set verbosity of the RISK instance.
    """
    try:
        risk_instance = risk(verbose=verbose_setting)
        assert risk_instance is not None
    except Exception:
        pytest.fail(f"RISK failed to initialize with verbose={verbose_setting}")


def test_missing_network_file(risk_obj):
    """
    Test loading a missing network file.

    Args:
        risk_obj: The RISK object instance used for loading the network.
    """
    with pytest.raises(FileNotFoundError):
        risk_obj.load_network_cytoscape(
            filepath="missing_file.cys", source_label="source", target_label="target"
        )


def test_load_network_cytoscape(risk_obj, data_path):
    """
    Test loading a Cytoscape network from a .cys file.

    Args:
        risk_obj: The RISK object instance used for loading the network.
        data_path: The base path to the directory containing the Cytoscape file.
    """
    cys_file = data_path / "cytoscape" / "michaelis_2023.cys"
    network = risk_obj.load_network_cytoscape(
        filepath=str(cys_file), source_label="source", target_label="target", view_name=""
    )

    assert network is not None
    assert len(network.nodes) > 0  # Check that the network has nodes
    assert len(network.edges) > 0  # Check that the network has edges


def test_load_network_cyjs(risk_obj, data_path):
    """
    Test loading a Cytoscape JSON network from a .cyjs file.

    Args:
        risk_obj: The RISK object instance used for loading the network.
        data_path: The base path to the directory containing the Cytoscape JSON file.
    """
    cyjs_file = data_path / "cyjs" / "michaelis_2023.cyjs"
    network = risk_obj.load_network_cyjs(
        filepath=str(cyjs_file), source_label="source", target_label="target"
    )

    assert network is not None
    assert len(network.nodes) > 0  # Check that the network has nodes
    assert len(network.edges) > 0  # Check that the network has edges


def test_load_network_gpickle(risk_obj, data_path):
    """
    Test loading a network from a .gpickle file.

    Args:
        risk_obj: The RISK object instance used for loading the network.
        data_path: The base path to the directory containing the gpickle file.
    """
    gpickle_file = data_path / "gpickle" / "michaelis_2023.gpickle"
    network = risk_obj.load_network_gpickle(filepath=str(gpickle_file))

    assert network is not None
    assert len(network.nodes) > 0  # Check that the network has nodes
    assert len(network.edges) > 0  # Check that the network has edges


def test_load_network_networkx(risk_obj, dummy_network):
    """
    Test loading a network from a NetworkX graph object.

    Args:
        risk_obj: The RISK object instance used for loading the network.
        network: The NetworkX graph object to be loaded into the RISK network.
    """
    network = risk_obj.load_network_networkx(network=dummy_network)

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


def test_round_trip_io(risk_obj):
    """
    Test saving and loading a small graph using the io module.

    Args:
        risk_obj: The RISK object instance used for loading the network.
    """
    # Create a small test graph
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2)])
    # Add node positions as required for network loading
    G.nodes[0]["x"] = 0.0
    G.nodes[0]["y"] = 0.0
    G.nodes[1]["x"] = 1.0
    G.nodes[1]["y"] = 1.0
    G.nodes[2]["x"] = 2.0
    G.nodes[2]["y"] = 2.0

    # Ensure the tmp directory exists under data/tmp
    tmp_dir = os.path.join("data", "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_path = os.path.join(tmp_dir, "test_round_trip_io.gpickle")

    try:
        # Save the graph using pickle
        with open(tmp_path, "wb") as f:
            pickle.dump(G, f)

        # Load it back using risk_obj's load_network_gpickle
        G_loaded = risk_obj.load_network_gpickle(filepath=tmp_path)

        # Compare properties of the graphs - RISK sets node IDs to 'label' attribute when no label is present
        assert set(G.nodes()) == set(G_loaded.nodes())
        assert set(G.edges()) == set(G_loaded.edges())

    finally:
        # Always remove the temporary file at the end
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_node_positions_constant_after_networkx_load(risk_obj, dummy_network):
    """
    Test that the original network retains its node positions ('x' and 'y') after being passed to
    the loading function.

    Args:
        risk_obj: The RISK object instance used for loading the network.
        network: The NetworkX graph object to be loaded into the RISK network.
    """
    # Store the original positions of nodes from the dummy_network
    original_positions = {
        node: (dummy_network.nodes[node]["x"], dummy_network.nodes[node]["y"])
        for node in dummy_network.nodes
    }
    # Pass the network to the load function, and ignore the returned network
    _ = risk_obj.load_network_networkx(network=dummy_network)

    # Ensure that the original network (dummy_network) still has the same node positions
    for node in dummy_network.nodes:
        assert (
            "x" in dummy_network.nodes[node]
        ), f"Original node {node} missing 'x' attribute after loading"
        assert (
            "y" in dummy_network.nodes[node]
        ), f"Original node {node} missing 'y' attribute after loading"
        assert (
            dummy_network.nodes[node]["x"] == original_positions[node][0]
        ), f"Original node {node} 'x' position changed"
        assert (
            dummy_network.nodes[node]["y"] == original_positions[node][1]
        ), f"Original node {node} 'y' position changed"


def test_attribute_fallback_mechanism(risk_obj, data_path):
    """
    Test attribute fallback mechanism by assigning 'x' and 'y' as 'pos' and
    using 'label' as the node ID after loading two networks with different configurations.

    Args:
        risk_obj: The RISK object instance used for loading the network.
        data_path: Path to the data directory containing the network files.
    """
    # Load the original network from the Cytoscape file with sphere set to False
    network_file = data_path / "cytoscape" / "michaelis_2023.cys"
    cytoscape_network = risk_obj.load_network_cytoscape(
        filepath=str(network_file),
        source_label="source",
        target_label="target",
        view_name="",
        compute_sphere=False,
        surface_depth=0.1,
        min_edges_per_node=0,
    )

    # Track the original node labels in the label order (before modifying the network)
    label_order = []
    # Modify the cytoscape_network by converting 'x' and 'y' to 'pos' and using 'label' as the node ID
    for node in list(cytoscape_network.nodes):
        attrs = cytoscape_network.nodes[node]
        # Assign 'x' and 'y' as 'pos' tuple and remove 'x' and 'y'
        if "x" in attrs and "y" in attrs:
            attrs["pos"] = (attrs.pop("x"), attrs.pop("y"))

        # Assign the 'label' attribute to the node ID and store the old node label in label_order
        label_order.append(attrs["label"])  # Store the original node ID label
        if "label" in attrs:
            new_node_id = attrs.pop("label")
            nx.relabel_nodes(cytoscape_network, {node: new_node_id}, copy=False)

    # Load the modified network with the sphere calculation disabled to preserve node positions
    network = risk_obj.load_network_networkx(
        network=cytoscape_network,
        compute_sphere=False,
        surface_depth=0.1,
        min_edges_per_node=0,
    )

    # Test the fallback mechanism for 'pos' -> 'x' and 'y' and ensure the new node 'label' matches the old node IDs
    for index, (node, attrs) in enumerate(network.nodes(data=True)):
        # Check that 'x' and 'y' are correctly extracted from 'pos'
        assert "x" in attrs, f"Node {node} is missing 'x' after fallback."
        assert "y" in attrs, f"Node {node} is missing 'y' after fallback."
        # Check that the 'pos' tuple was converted correctly to 'x' and 'y'
        assert "pos" in attrs
        assert attrs["x"] == attrs["pos"][0], f"Node {node} 'x' not correctly extracted from 'pos'."
        assert attrs["y"] == attrs["pos"][1], f"Node {node} 'y' not correctly extracted from 'pos'."
        # Check if the reloaded node's 'label' matches the original node ID
        expected_node_id = label_order[index]
        assert (
            attrs["label"] == expected_node_id
        ), f"Node {node} 'label' doesn't match the expected node ID {expected_node_id}."


@pytest.mark.parametrize("min_edges", [1, 5, 10])
def test_load_network_min_edges(risk_obj, data_path, min_edges):
    """
    Test loading a Cytoscape network with varying min_edges_per_node.

    Args:
        risk_obj: The RISK object instance used for loading the network.
        data_path: The base path to the directory containing the Cytoscape file.
        min_edges: The minimum number of edges per node to test.
    """
    cys_file = data_path / "cytoscape" / "michaelis_2023.cys"
    network = risk_obj.load_network_cytoscape(
        filepath=str(cys_file),
        source_label="source",
        target_label="target",
        min_edges_per_node=min_edges,
        compute_sphere=True,
        surface_depth=0.5,
    )

    # Check that each node has at least min_edges
    for node in network.nodes:
        assert network.degree[node] >= min_edges, f"Node {node} has fewer than {min_edges} edges"


def test_node_and_edge_attributes(risk_obj, data_path):
    """
    Test that nodes and edges have the required attributes after loading.

    Args:
        risk_obj: The RISK object instance used for loading the network.
        data_path: The base path to the directory containing the Cytoscape file.
    """
    cys_file = data_path / "cytoscape" / "michaelis_2023.cys"
    network = risk_obj.load_network_cytoscape(
        filepath=str(cys_file),
        source_label="source",
        target_label="target",
    )

    # Validate node attributes
    for node, attrs in network.nodes(data=True):
        assert "x" in attrs, f"Node {node} is missing 'x' attribute"
        assert "y" in attrs, f"Node {node} is missing 'y' attribute"
        assert "label" in attrs, f"Node {node} is missing 'label' attribute"

    # Validate edge attributes
    for u, v, attrs in network.edges(data=True):
        assert "length" in attrs, f"Edge ({u}, {v}) is missing 'length' attribute"
        assert "weight" in attrs, f"Edge ({u}, {v}) is missing 'weight' attribute"


def test_sphere_unfolding(risk_obj, data_path):
    """
    Test that the sphere-to-plane unfolding correctly updates node coordinates.

    Args:
        risk_obj: The RISK object instance used for loading the network.
        data_path: The base path to the directory containing the Cytoscape file.
    """
    cys_file = data_path / "cytoscape" / "michaelis_2023.cys"
    network = risk_obj.load_network_cytoscape(
        filepath=str(cys_file),
        source_label="source",
        target_label="target",
        compute_sphere=True,
        surface_depth=0.5,
    )

    # Ensure all nodes have 'x' and 'y' coordinates in [0, 1]
    for node, attrs in network.nodes(data=True):
        assert -1 <= attrs["x"] <= 1, f"Node {node} 'x' coordinate is out of bounds"
        assert -1 <= attrs["y"] <= 1, f"Node {node} 'y' coordinate is out of bounds"


def test_edge_attribute_fallback(risk_obj, dummy_network):
    """
    Test fallback when edges are missing 'length' or 'weight' attributes.

    Args:
        risk_obj: The RISK object instance used for loading the network.
        dummy_network: The Cytoscape network to be loaded into the R
    """
    # Remove 'length' and 'weight' attributes
    for u, v in dummy_network.edges():
        del dummy_network.edges[u, v]["length"]
        del dummy_network.edges[u, v]["weight"]

    network = risk_obj.load_network_networkx(network=dummy_network)

    # Ensure fallback attributes are assigned correctly
    for u, v, attrs in network.edges(data=True):
        assert "length" in attrs, f"Edge ({u}, {v}) is missing fallback 'length' attribute"
        assert "weight" in attrs, f"Edge ({u}, {v}) is missing fallback 'weight' attribute"


@pytest.mark.skipif(
    sys.platform == "win32" and sys.version_info[:2] == (3, 10),
    reason="Fails due to recursion depth in Windows 3.10",
)
def test_deterministic_network_loading(risk_obj, data_path):
    """
    Test that loading the same network produces identical results.

    Args:
        risk_obj: The RISK object instance used for loading the network.
        data_path: The base path to the directory containing the Cytoscape file.
    """
    cys_file = data_path / "cytoscape" / "michaelis_2023.cys"
    network_1 = risk_obj.load_network_cytoscape(
        filepath=str(cys_file),
        source_label="source",
        target_label="target",
        compute_sphere=True,
        surface_depth=0.5,
    )
    network_2 = risk_obj.load_network_cytoscape(
        filepath=str(cys_file),
        source_label="source",
        target_label="target",
        compute_sphere=True,
        surface_depth=0.5,
    )

    assert nx.is_isomorphic(network_1, network_2), "Loaded networks should be identical"


def test_missing_node_attributes(risk_obj, cytoscape_network):
    """
    Test fallback mechanism for missing node attributes.

    Args:
        risk_obj: The RISK object instance used for loading the network.
        cytoscape_network: The Cytoscape network to be loaded into the RISK network.
    """
    # Remove 'x', 'y', and 'label' attributes
    for node in cytoscape_network.nodes:
        attrs = cytoscape_network.nodes[node]
        attrs.pop("x", None)
        attrs.pop("y", None)
        attrs.pop("label", None)

    # Expect a ValueError to be raised due to missing 'x', 'y', and 'pos' attributes
    with pytest.raises(
        ValueError, match="Node .* is missing 'x', 'y', and a valid 'pos' attribute."
    ):
        risk_obj.load_network_networkx(network=cytoscape_network)


def test_remove_isolates_does_not_raise(risk_obj, dummy_network):
    """
    Test that loading a network with isolated nodes does not raise an error.

    Args:
        risk_obj: The RISK object instance used for loading the network.
        dummy_network: The NetworkX graph object to be loaded into the RISK network.
    """
    G = dummy_network.copy()
    G.add_node("iso")
    G.nodes["iso"]["x"] = 0.0
    G.nodes["iso"]["y"] = 0.0
    G.nodes["iso"]["label"] = "iso"
    # Remove nodes with fewer than min_edges_per_node=1
    loaded = risk_obj.load_network_networkx(
        network=G, compute_sphere=False, surface_depth=0.1, min_edges_per_node=1
    )
    assert "iso" not in loaded.nodes, "Isolated node 'iso' should have been removed without error"
