"""
tests/test_load_graph
~~~~~~~~~~~~~~~~~~~~~
"""


def test_load_graph_with_json_annotation(risk_obj, cytoscape_network, json_annotation):
    """Test loading a graph after generating neighborhoods with specific parameters using JSON annotations.

    Args:
        risk_obj: The RISK object instance used for loading neighborhoods and graphs.
        cytoscape_network: The network object to be used for neighborhood and graph generation.
        json_annotation: The JSON annotations associated with the network.

    Returns:
        None
    """
    # Load neighborhoods as a prerequisite
    neighborhoods = risk_obj.load_neighborhoods_by_permutation(
        network=cytoscape_network,
        annotations=json_annotation,
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


def test_top_annotations_cluster_sizes_with_json_annotation(
    risk_obj, cytoscape_network, json_annotation
):
    """Test that `top_annotations` column 'size connected components' respects min and max cluster sizes using JSON annotations.

    Args:
        risk_obj: The RISK object instance used for loading neighborhoods and graphs.
        cytoscape_network: The network object to be used for neighborhood and graph generation.
        json_annotation: The JSON annotations associated with the network.

    Returns:
        None
    """
    # Define different combinations of min and max cluster sizes
    cluster_size_combinations = [(5, 1000), (10, 500), (20, 300), (50, 200)]
    for min_cluster_size, max_cluster_size in cluster_size_combinations:
        # Load neighborhoods as a prerequisite
        neighborhoods = risk_obj.load_neighborhoods_by_permutation(
            network=cytoscape_network,
            annotations=json_annotation,
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
        # Validate the 'size connected components' in `top_annotations` DataFrame
        size_connected_components = graph.top_annotations["size connected components"]
        _check_component_sizes(size_connected_components, min_cluster_size, max_cluster_size)


def test_load_graph_with_dict_annotation(risk_obj, cytoscape_network, dict_annotation):
    """Test loading a graph after generating neighborhoods with specific parameters using dictionary annotations.

    Args:
        risk_obj: The RISK object instance used for loading neighborhoods and graphs.
        cytoscape_network: The network object to be used for neighborhood and graph generation.
        dict_annotation: The dictionary annotations associated with the network.

    Returns:
        None
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


def test_top_annotations_cluster_sizes_with_dict_annotation(
    risk_obj, cytoscape_network, dict_annotation
):
    """Test that `top_annotations` column 'size connected components' respects min and max cluster sizes using dictionary annotations.

    Args:
        risk_obj: The RISK object instance used for loading neighborhoods and graphs.
        cytoscape_network: The network object to be used for neighborhood and graph generation.
        dict_annotation: The dictionary annotations associated with the network.

    Returns:
        None
    """
    # Define different combinations of min and max cluster sizes
    cluster_size_combinations = [(5, 1000), (10, 500), (20, 300), (50, 200)]
    for min_cluster_size, max_cluster_size in cluster_size_combinations:
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
            min_cluster_size=min_cluster_size,
            max_cluster_size=max_cluster_size,
        )

        # Validate the graph and its components
        _validate_graph(graph)
        # Validate the 'size connected components' in `top_annotations` DataFrame
        size_connected_components = graph.top_annotations["size connected components"]
        _check_component_sizes(size_connected_components, min_cluster_size, max_cluster_size)


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


def _check_component_sizes(size_connected_components, min_cluster_size, max_cluster_size):
    """Check whether components in `size_connected_components` are within the size range.

    Args:
        size_connected_components (list): A list of component sizes (can be list or int).
        min_cluster_size (int): The minimum allowed size for components.
        max_cluster_size (int): The maximum allowed size for components.

    Raises:
        AssertionError: If any component size is outside the specified range.
    """
    for components in size_connected_components:
        if isinstance(components, list):
            # Debugging: Print components being checked
            print(f"Checking components: {components}")
            for size in components:
                if not (min_cluster_size <= size <= max_cluster_size):
                    print(
                        f"Component size {size} is outside the range {min_cluster_size} to {max_cluster_size}"
                    )
            # Ensure all sizes in the list are within the range
            assert all(
                min_cluster_size <= size <= max_cluster_size for size in components
            ), f"Some values in 'size connected components' are outside the range {min_cluster_size} to {max_cluster_size}"
        elif isinstance(components, int):
            # Debugging: Print single component being checked
            print(f"Checking component: {components}")
            if not (min_cluster_size <= components <= max_cluster_size):
                print(
                    f"Component size {components} is outside the range {min_cluster_size} to {max_cluster_size}"
                )
            # Ensure the single component is within the range
            assert (
                min_cluster_size <= components <= max_cluster_size
            ), f"Value {components} in 'size connected components' is outside the range {min_cluster_size} to {max_cluster_size}"
