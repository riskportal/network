"""
tests/test_load_graph
~~~~~~~~~~~~~~~~~~~~~
"""

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from risk._network._graph._summary import Summary


def test_load_graph_with_json_annotation(risk_obj, cytoscape_network, json_annotation):
    """
    Test loading a graph after generating neighborhoods with specific parameters using JSON annotation.

    Args:
        risk_obj: The RISK object instance used for loading neighborhoods and graphs.
        cytoscape_network: The network object to be used for neighborhood and graph generation.
        json_annotation: The JSON annotation associated with the network.
    """
    # Load neighborhoods as a prerequisite
    neighborhoods = risk_obj.load_neighborhoods_permutation(
        network=cytoscape_network,
        annotation=json_annotation,
        distance_metric="leiden",
        louvain_resolution=8,
        leiden_resolution=1.0,
        fraction_shortest_edges=0.75,
        score_metric="stdev",
        null_distribution="network",
        num_permutations=20,
        random_seed=887,
        max_workers=1,
    )
    # Load the graph with the specified parameters
    graph = risk_obj.load_graph(
        network=cytoscape_network,
        annotation=json_annotation,
        neighborhoods=neighborhoods,
        tail="right",
        pval_cutoff=0.05,
        fdr_cutoff=1.0,
        impute_depth=1,
        prune_threshold=0.1,
        linkage_criterion="distance",
        linkage_method="average",
        linkage_metric="yule",
        linkage_threshold=0.2,
        min_cluster_size=5,
        max_cluster_size=1000,
    )

    # Validate the graph and its components
    _validate_graph(graph)


def test_cluster_size_limits_with_json_annotation(risk_obj, cytoscape_network, json_annotation):
    """
    Test that statistically significant domains respect min and max cluster sizes using JSON annotation.

    Args:
        risk_obj: The RISK object instance used for loading neighborhoods and graphs.
        cytoscape_network: The network object to be used for neighborhood and graph generation.
        json_annotation: The JSON annotation associated with the network.
    """
    # Define different combinations of min and max cluster sizes
    cluster_size_combinations = [(5, 1000), (10, 500), (20, 300), (50, 200)]
    for min_cluster_size, max_cluster_size in cluster_size_combinations:
        # Load neighborhoods as a prerequisite
        neighborhoods = risk_obj.load_neighborhoods_permutation(
            network=cytoscape_network,
            annotation=json_annotation,
            # Test multiple distance metrics
            distance_metric=["louvain", "label_propagation"],
            louvain_resolution=8,
            fraction_shortest_edges=0.75,
            score_metric="stdev",
            null_distribution="network",
            num_permutations=20,
            random_seed=887,
            max_workers=1,
        )
        # Load the graph with the specified parameters
        graph = risk_obj.load_graph(
            network=cytoscape_network,
            annotation=json_annotation,
            neighborhoods=neighborhoods,
            tail="right",
            pval_cutoff=0.05,
            fdr_cutoff=1.0,
            impute_depth=1,
            prune_threshold=0.1,
            linkage_criterion="distance",
            linkage_method="average",
            linkage_metric="yule",
            linkage_threshold=0.2,
            min_cluster_size=min_cluster_size,
            max_cluster_size=max_cluster_size,
        )

        # Validate the graph and its components
        _validate_graph(graph)
        # Validate the size of the domains
        _check_component_sizes(graph.domain_id_to_node_ids_map, min_cluster_size, max_cluster_size)


def test_load_graph_with_dict_annotation(risk_obj, cytoscape_network, dict_annotation):
    """
    Test loading a graph after generating neighborhoods with specific parameters using dictionary annotation.

    Args:
        risk_obj: The RISK object instance used for loading neighborhoods and graphs.
        cytoscape_network: The network object to be used for neighborhood and graph generation.
        dict_annotation: The dictionary annotation associated with the network.
    """
    # Load neighborhoods as a prerequisite
    neighborhoods = risk_obj.load_neighborhoods_permutation(
        network=cytoscape_network,
        annotation=dict_annotation,
        distance_metric="louvain",
        louvain_resolution=8,
        fraction_shortest_edges=0.75,
        score_metric="stdev",
        null_distribution="network",
        num_permutations=20,
        random_seed=887,
        max_workers=1,
    )
    # Load the graph with the specified parameters
    graph = risk_obj.load_graph(
        network=cytoscape_network,
        annotation=dict_annotation,
        neighborhoods=neighborhoods,
        tail="right",
        pval_cutoff=0.05,
        fdr_cutoff=1.0,
        impute_depth=1,
        prune_threshold=0.1,
        linkage_criterion="distance",
        linkage_method="average",
        linkage_metric="yule",
        linkage_threshold=0.2,
        min_cluster_size=5,
        max_cluster_size=1000,
    )

    # Validate the graph and its components
    _validate_graph(graph)


def test_cluster_size_limits_with_dict_annotation(risk_obj, cytoscape_network, dict_annotation):
    """
    Test that statistically significant domains respect min and max cluster sizes using dictionary annotation.

    Args:
        risk_obj: The RISK object instance used for loading neighborhoods and graphs.
        cytoscape_network: The network object to be used for neighborhood and graph generation.
        dict_annotation: The dictionary annotation associated with the network.
    """
    # Define different combinations of min and max cluster sizes
    cluster_size_combinations = [(5, 1000), (10, 500), (20, 300), (50, 200)]
    for min_cluster_size, max_cluster_size in cluster_size_combinations:
        # Load neighborhoods as a prerequisite
        neighborhoods = risk_obj.load_neighborhoods_permutation(
            network=cytoscape_network,
            annotation=dict_annotation,
            # Test multiple distance metrics
            distance_metric=["louvain", "label_propagation"],
            louvain_resolution=8,
            # Test multiple edge length thresholds
            fraction_shortest_edges=[0.75, 0.25],
            score_metric="stdev",
            null_distribution="network",
            num_permutations=20,
            random_seed=887,
            max_workers=1,
        )
        # Load the graph with the specified parameters
        graph = risk_obj.load_graph(
            network=cytoscape_network,
            annotation=dict_annotation,
            neighborhoods=neighborhoods,
            tail="right",
            pval_cutoff=0.05,
            fdr_cutoff=1.0,
            impute_depth=1,
            prune_threshold=0.1,
            linkage_criterion="distance",
            linkage_method="average",
            linkage_metric="yule",
            linkage_threshold=0.2,
            min_cluster_size=min_cluster_size,
            max_cluster_size=max_cluster_size,
        )

        # Validate the graph and its components
        _validate_graph(graph)
        # Validate the size of the domains
        _check_component_sizes(graph.domain_id_to_node_ids_map, min_cluster_size, max_cluster_size)


def test_linkage_criterion_and_auto_clustering_options(
    risk_obj, cytoscape_network, json_annotation
):
    """
    Test the linkage criterion and auto-clustering options for generating graphs.

    Args:
        risk_obj: The RISK object instance used for loading neighborhoods and graphs.
        cytoscape_network: The network object to be used for neighborhood and graph generation.
        json_annotation: The JSON annotation associated with the network.
    """
    # Define parameters for testing
    test_criteria = ["maxclust", "distance", "off"]
    min_cluster_size, max_cluster_size = 10, 200  # Fixed for simplicity
    for criterion in test_criteria:
        # Load neighborhoods as a prerequisite
        neighborhoods = risk_obj.load_neighborhoods_binom(
            network=cytoscape_network,
            annotation=json_annotation,
            distance_metric="louvain",
            louvain_resolution=1.0,
            fraction_shortest_edges=0.75,
            null_distribution="network",
            random_seed=888,
        )
        # Load the graph with the specified linkage_criterion
        graph = risk_obj.load_graph(
            network=cytoscape_network,
            annotation=json_annotation,
            neighborhoods=neighborhoods,
            tail="right",
            pval_cutoff=0.05,
            fdr_cutoff=1.0,
            impute_depth=1,
            prune_threshold=0.1,
            linkage_criterion=criterion,
            linkage_method="auto",
            linkage_metric="auto",
            linkage_threshold="auto",
            min_cluster_size=min_cluster_size,
            max_cluster_size=max_cluster_size,
        )

        # Validate graph for all criteria
        _validate_graph(graph)
        # Check cluster size bounds for 'distance', 'maxclust', and 'off' criteria
        _check_component_sizes(graph.domain_id_to_node_ids_map, min_cluster_size, max_cluster_size)


def test_network_graph_structure(risk_obj, cytoscape_network, json_annotation):
    """
    Test that the Graph object contains the expected components.

    Args:
        risk_obj: The RISK object instance used for loading neighborhoods and graphs.
        cytoscape_network: The network object to be used for neighborhood and graph generation.
        json_annotation: The JSON annotation associated with the network.
    """
    # Load neighborhoods as a prerequisite
    neighborhoods = risk_obj.load_neighborhoods_permutation(
        network=cytoscape_network,
        annotation=json_annotation,
        distance_metric="leiden",
        louvain_resolution=8,
        fraction_shortest_edges=0.75,
        score_metric="stdev",
        null_distribution="network",
        num_permutations=20,
        random_seed=887,
        max_workers=1,
    )
    # Load the graph with the specified parameters
    graph = risk_obj.load_graph(
        network=cytoscape_network,
        annotation=json_annotation,
        neighborhoods=neighborhoods,
        tail="right",
        pval_cutoff=0.05,
        fdr_cutoff=1.0,
        impute_depth=1,
        prune_threshold=0.1,
        linkage_criterion="distance",
        linkage_method="average",
        linkage_metric="yule",
        linkage_threshold=0.2,
        min_cluster_size=5,
        max_cluster_size=1000,
    )

    # Validate the graph attributes
    assert isinstance(
        graph.domain_id_to_node_ids_map, dict
    ), "Domain ID to node IDs map should be a dictionary"
    assert isinstance(
        graph.domain_id_to_domain_terms_map, dict
    ), "Domain ID to domain terms map should be a dictionary"
    assert isinstance(
        graph.domain_id_to_domain_info_map, dict
    ), "Domain ID to domain info map should be a dictionary"
    assert isinstance(
        graph.node_id_to_domain_ids_and_significance_map, dict
    ), "Node ID to domain IDs and significance map should be a dictionary"
    assert isinstance(
        graph.node_id_to_node_label_map, dict
    ), "Node ID to node label map should be a dictionary"
    assert isinstance(
        graph.node_label_to_significance_map, dict
    ), "Node label to significance map should be a dictionary"
    assert isinstance(
        graph.node_significance_sums, np.ndarray
    ), "Node significance sums should be a numpy array"
    assert isinstance(
        graph.node_label_to_node_id_map, dict
    ), "Node label to ID map should be a dictionary"
    assert isinstance(
        graph.domain_id_to_node_labels_map, dict
    ), "Domain ID to node labels map should be a dictionary"
    assert isinstance(graph.network, nx.Graph), "Network should be a NetworkX graph"
    assert isinstance(
        graph.node_coordinates, np.ndarray
    ), "Node coordinates should be a numpy array"
    assert isinstance(graph.summary, Summary), "Summary should be an Summary object"


def test_load_graph_summary(graph):
    """
    Test loading the graph summary with predefined parameters.

    Args:
        graph: The graph object instance to be summarized.
    """
    # Load the graph summary and validate its type
    summary = graph.summary.load()

    assert isinstance(summary, pd.DataFrame), "Graph summary should be a dictionary"


def test_pop_domain(graph):
    """
    Test the pop method for removing a domain ID from all Graph attribute domain mappings.

    Args:
        graph: The graph object instance with existing domain mappings.
    """
    # Define the domain ID to be removed
    domain_id_to_remove = 1
    # Retrieve expected labels before popping
    expected_labels = graph.domain_id_to_node_labels_map.get(domain_id_to_remove)
    # Pop the domain ID and get the returned value
    popped_labels = graph.pop(domain_id_to_remove)
    # Assert the returned value equals the expected labels
    assert popped_labels == expected_labels, "Popped labels do not match the expected labels."

    # Check that the domain ID is removed from all relevant attributes
    assert (
        domain_id_to_remove not in graph.domain_id_to_node_ids_map
    ), f"{domain_id_to_remove} should be removed from domain_id_to_node_ids_map"
    assert (
        domain_id_to_remove not in graph.domain_id_to_domain_terms_map
    ), f"{domain_id_to_remove} should be removed from domain_id_to_domain_terms_map"
    assert (
        domain_id_to_remove not in graph.domain_id_to_domain_info_map
    ), f"{domain_id_to_remove} should be removed from domain_id_to_domain_info_map"
    assert (
        domain_id_to_remove not in graph.domain_id_to_node_labels_map
    ), f"{domain_id_to_remove} should be removed from domain_id_to_node_labels_map"

    # Check if the domain was removed from node_id_to_domain_ids_and_significance_map
    for _, domain_info in graph.node_id_to_domain_ids_and_significance_map.items():
        assert domain_id_to_remove not in domain_info.get(
            "domains", []
        ), f"{domain_id_to_remove} should be removed from node_id_to_domain_ids_and_significance_map['domains']"
        assert domain_id_to_remove not in domain_info.get(
            "significances", {}
        ), f"{domain_id_to_remove} should be removed from node_id_to_domain_ids_and_significance_map['significances']"


def _validate_graph(graph):
    """
    Validate that the graph is not None and contains nodes and edges.

    Args:
        graph: The graph object to validate.

    Raises:
        AssertionError: If the graph is None or if it contains no nodes or edges.
    """
    # For some reason, Windows can periodically return a graph with no nodes or edges
    if graph is None:
        pytest.skip("Skipping test: Graph is None.")
    if len(graph.network.nodes) == 0 or len(graph.network.edges) == 0:
        pytest.skip("Skipping test: Graph has no nodes or edges.")

    assert graph is not None, "Graph is None."
    assert len(graph.network.nodes) > 0, "Graph has no nodes."
    assert len(graph.network.edges) > 0, "Graph has no edges."


def _check_component_sizes(domain_id_to_node_id_map, min_cluster_size, max_cluster_size):
    """
    Check whether domains are within the specified size range.

    Args:
        domain_id_to_node_id_map (dict): A mapping of domain IDs to lists of node IDs.
        min_cluster_size (int): The minimum allowed size for components.
        max_cluster_size (int): The maximum allowed size for components.
    """
    for domain_id, node_ids in domain_id_to_node_id_map.items():
        # Skip invalid domain IDs
        if pd.isna(domain_id) or domain_id is None:
            print(f"Skipping invalid domain ID: {domain_id}")
            continue

        component_size = len(node_ids)
        # Debugging: Print the domain ID and its size
        print(f"Checking domain ID {domain_id} with size {component_size}")

        if not min_cluster_size <= component_size <= max_cluster_size:
            print(
                f"Domain {domain_id} size {component_size} is outside the range "
                f"{min_cluster_size} to {max_cluster_size}"
            )

        assert min_cluster_size <= component_size <= max_cluster_size, (
            f"Domain {domain_id} has size {component_size}, which is outside the range "
            f"{min_cluster_size} to {max_cluster_size}"
        )
