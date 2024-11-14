"""
tests/test_load_graph
~~~~~~~~~~~~~~~~~~~~~
"""

import pandas as pd


def test_load_graph_with_json_annotation(risk_obj, cytoscape_network, json_annotation):
    """Test loading a graph after generating neighborhoods with specific parameters using JSON annotations.

    Args:
        risk_obj: The RISK object instance used for loading neighborhoods and graphs.
        cytoscape_network: The network object to be used for neighborhood and graph generation.
        json_annotation: The JSON annotations associated with the network.
    """
    # Load neighborhoods as a prerequisite
    neighborhoods = risk_obj.load_neighborhoods_by_permutation(
        network=cytoscape_network,
        annotations=json_annotation,
        distance_metric="leiden",
        louvain_resolution=8,
        leiden_resolution=1.0,
        edge_length_threshold=0.75,
        score_metric="stdev",
        null_distribution="network",
        num_permutations=100,
        random_seed=887,
        max_workers=4,
    )
    # Load the graph with the specified parameters
    graph = risk_obj.load_graph(
        network=cytoscape_network,
        annotations=json_annotation,
        neighborhoods=neighborhoods,
        tail="right",
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

    # Validate the graph and its components
    _validate_graph(graph)


def test_cluster_size_limits_with_json_annotation(risk_obj, cytoscape_network, json_annotation):
    """Test that statistically significant domains respect min and max cluster sizes using JSON annotations.

    Args:
        risk_obj: The RISK object instance used for loading neighborhoods and graphs.
        cytoscape_network: The network object to be used for neighborhood and graph generation.
        json_annotation: The JSON annotations associated with the network.
    """
    # Define different combinations of min and max cluster sizes
    cluster_size_combinations = [(5, 1000), (10, 500), (20, 300), (50, 200)]
    for min_cluster_size, max_cluster_size in cluster_size_combinations:
        # Load neighborhoods as a prerequisite
        neighborhoods = risk_obj.load_neighborhoods_by_permutation(
            network=cytoscape_network,
            annotations=json_annotation,
            # Test multiple distance metrics
            distance_metric=["louvain", "label_propagation"],
            louvain_resolution=8,
            edge_length_threshold=0.75,
            score_metric="stdev",
            null_distribution="network",
            num_permutations=100,
            random_seed=887,
            max_workers=4,
        )
        # Load the graph with the specified parameters
        graph = risk_obj.load_graph(
            network=cytoscape_network,
            annotations=json_annotation,
            neighborhoods=neighborhoods,
            tail="right",
            pval_cutoff=0.05,
            fdr_cutoff=1.0,
            impute_depth=1,
            prune_threshold=0.1,
            linkage_criterion="distance",
            linkage_method="average",
            linkage_metric="yule",
            min_cluster_size=min_cluster_size,
            max_cluster_size=max_cluster_size,
        )

        # Validate the graph and its components
        _validate_graph(graph)
        # Validate the size of the domains
        _check_component_sizes(graph.domain_id_to_node_ids_map, min_cluster_size, max_cluster_size)


def test_load_graph_with_dict_annotation(risk_obj, cytoscape_network, dict_annotation):
    """Test loading a graph after generating neighborhoods with specific parameters using dictionary annotations.

    Args:
        risk_obj: The RISK object instance used for loading neighborhoods and graphs.
        cytoscape_network: The network object to be used for neighborhood and graph generation.
        dict_annotation: The dictionary annotations associated with the network.
    """
    # Load neighborhoods as a prerequisite
    neighborhoods = risk_obj.load_neighborhoods_by_permutation(
        network=cytoscape_network,
        annotations=dict_annotation,
        distance_metric="louvain",
        louvain_resolution=8,
        edge_length_threshold=0.75,
        score_metric="stdev",
        null_distribution="network",
        num_permutations=100,
        random_seed=887,
        max_workers=4,
    )
    # Load the graph with the specified parameters
    graph = risk_obj.load_graph(
        network=cytoscape_network,
        annotations=dict_annotation,
        neighborhoods=neighborhoods,
        tail="right",
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

    # Validate the graph and its components
    _validate_graph(graph)


def test_cluster_size_limits_with_dict_annotation(risk_obj, cytoscape_network, dict_annotation):
    """Test that statistically significant domains respect min and max cluster sizes using dictionary annotations.

    Args:
        risk_obj: The RISK object instance used for loading neighborhoods and graphs.
        cytoscape_network: The network object to be used for neighborhood and graph generation.
        dict_annotation: The dictionary annotations associated with the network.
    """
    # Define different combinations of min and max cluster sizes
    cluster_size_combinations = [(5, 1000), (10, 500), (20, 300), (50, 200)]
    for min_cluster_size, max_cluster_size in cluster_size_combinations:
        # Load neighborhoods as a prerequisite
        neighborhoods = risk_obj.load_neighborhoods_by_permutation(
            network=cytoscape_network,
            annotations=dict_annotation,
            # Test multiple distance metrics
            distance_metric=["louvain", "label_propagation"],
            louvain_resolution=8,
            # Test multiple edge length thresholds
            edge_length_threshold=[0.75, 0.25],
            score_metric="stdev",
            null_distribution="network",
            num_permutations=100,
            random_seed=887,
            max_workers=4,
        )
        # Load the graph with the specified parameters
        graph = risk_obj.load_graph(
            network=cytoscape_network,
            annotations=dict_annotation,
            neighborhoods=neighborhoods,
            tail="right",
            pval_cutoff=0.05,
            fdr_cutoff=1.0,
            impute_depth=1,
            prune_threshold=0.1,
            linkage_criterion="distance",
            linkage_method="average",
            linkage_metric="yule",
            min_cluster_size=min_cluster_size,
            max_cluster_size=max_cluster_size,
        )

        # Validate the graph and its components
        _validate_graph(graph)
        # Validate the size of the domains
        _check_component_sizes(graph.domain_id_to_node_ids_map, min_cluster_size, max_cluster_size)


def test_load_graph_summary(graph):
    """Test loading the graph summary with predefined parameters.

    Args:
        graph: The graph object instance to be summarized.
    """
    # Load the graph summary and validate its type
    summary = graph.summary.load()

    assert isinstance(summary, pd.DataFrame), "Graph summary should be a dictionary"


def test_pop_domain(graph):
    """Test the pop method for removing a domain ID from all NetworkGraph attribute domain mappings.

    Args:
        graph: The graph object instance with existing domain mappings.
    """
    # Define the domain ID to be removed
    domain_id_to_remove = 1
    # Pop the domain ID
    graph.pop(domain_id_to_remove)

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
    for node_id, domain_info in graph.node_id_to_domain_ids_and_significance_map.items():
        assert domain_id_to_remove not in domain_info.get(
            "domains", []
        ), f"{domain_id_to_remove} should be removed from node_id_to_domain_ids_and_significance_map['domains']"
        assert domain_id_to_remove not in domain_info.get(
            "significances", {}
        ), f"{domain_id_to_remove} should be removed from node_id_to_domain_ids_and_significance_map['significances']"


def _validate_graph(graph):
    """Validate that the graph is not None and contains nodes and edges.

    Args:
        graph: The graph object to validate.

    Raises:
        AssertionError: If the graph is None or if it contains no nodes or edges.
    """
    assert graph is not None, "Graph is None."
    assert len(graph.network.nodes) > 0, "Graph has no nodes."
    assert len(graph.network.edges) > 0, "Graph has no edges."


def _check_component_sizes(domain_id_to_node_id_map, min_cluster_size, max_cluster_size):
    """Check whether domains are within the specified size range.

    Args:
        domain_id_to_node_id_map (dict): A mapping of domain IDs to lists of node IDs.
        min_cluster_size (int): The minimum allowed size for components.
        max_cluster_size (int): The maximum allowed size for components.
    """
    for domain_id, node_ids in domain_id_to_node_id_map.items():
        # Calculate the size of the current component
        component_size = len(node_ids)
        # Debugging: Print the domain ID and its size
        print(f"Checking domain ID {domain_id} with size {component_size}")
        # Ensure the component size is within the specified range
        if not (min_cluster_size <= component_size <= max_cluster_size):
            print(
                f"Domain {domain_id} size {component_size} is outside the range "
                f"{min_cluster_size} to {max_cluster_size}"
            )

        assert min_cluster_size <= component_size <= max_cluster_size, (
            f"Domain {domain_id} has size {component_size}, which is outside the range "
            f"{min_cluster_size} to {max_cluster_size}"
        )
