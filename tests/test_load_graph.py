"""
tests/test_load_graph
~~~~~~~~~~~~~~~~~~~~~
"""


def test_load_graph(risk_obj, cytoscape_network, json_annotation):
    """Test loading a graph after generating neighborhoods with specific parameters.

    Args:
        risk_obj: The RISK object instance used for loading neighborhoods and graphs.
        cytoscape_network: The network object to be used for neighborhood and graph generation.
        json_annotation: The annotations associated with the network.

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

    assert graph is not None
    assert len(graph.network.nodes) > 0  # Ensure that the graph has nodes
    assert len(graph.network.edges) > 0  # Ensure that the graph has edges


def test_top_annotations_cluster_sizes(risk_obj, cytoscape_network, json_annotation):
    """Test that `top_annotations` column 'size connected components' respects the min and max cluster sizes.

    Args:
        risk_obj: The RISK object instance used for loading neighborhoods and graphs.
        cytoscape_network: The network object to be used for neighborhood and graph generation.
        json_annotation: The annotations associated with the network.

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

        assert graph is not None
        assert len(graph.network.nodes) > 0  # Ensure that the graph has nodes
        assert len(graph.network.edges) > 0  # Ensure that the graph has edges
        # Validate the 'size connected components' in `top_annotations` DataFrame
        size_connected_components = graph.top_annotations["size connected components"]
        assert size_connected_components.between(
            min_cluster_size, max_cluster_size
        ).all(), f"Some values in 'size connected components' are outside the range {min_cluster_size} to {max_cluster_size}"
