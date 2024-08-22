"""
tests/test_load_graph
~~~~~~~~~~~~~~~~~~~~~
"""

def test_load_graph(risk_obj, network, annotations):
    """Test loading a graph after generating neighborhoods with specific parameters

    Args:
        risk_obj: The RISK object instance used for loading neighborhoods and graphs
        network: The network object to be used for neighborhood and graph generation
        annotations: The annotations associated with the network

    Returns:
        None
    """
    # Load neighborhoods as a prerequisite
    neighborhoods = risk_obj.load_neighborhoods(
        network=network,
        annotations=annotations,
        distance_metric="louvain",
        louvain_resolution=8,
        edge_length_threshold=0.75,
        score_metric="stdev",
        null_distribution="network",
        num_permutations=100,  # Perform 100 permutations
        random_seed=887,
        max_workers=4,  # Use 4 processes
    )

    # Load the graph with the specified parameters
    graph = risk_obj.load_graph(
        network=network,
        annotations=annotations,
        neighborhoods=neighborhoods,
        tail="right",  # Right tail for enrichment
        pval_cutoff=0.05,  # P-value cutoff of 0.05
        fdr_cutoff=1.0,  # FDR cutoff
        impute_depth=1,  # Set impute depth to 1
        prune_threshold=0.1,  # Prune threshold set to 0.1
        linkage_criterion="distance",  # Clustering based on distance
        linkage_method="average",  # Set linkage method to average
        linkage_metric="yule",  # Set linkage metric to yule
        min_cluster_size=5,  # Minimum cluster size set to 5
        max_cluster_size=1000,  # Maximum cluster size set to 1000
    )

    assert graph is not None
    assert len(graph.network.nodes) > 0  # Ensure that the graph has nodes
    assert len(graph.network.edges) > 0  # Ensure that the graph has edges
