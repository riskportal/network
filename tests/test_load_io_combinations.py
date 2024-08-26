"""
tests/test_load_io_combinations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import pytest


@pytest.mark.parametrize(
    "network_fixture",
    [
        "cytoscape_network",
        "cytoscape_json_network",
        "gpickle_network",
        "networkx_network",
    ],
)
@pytest.mark.parametrize(
    "annotation_fixture",
    [
        "json_annotation",
        "csv_annotation",
        "tsv_annotation",
        "excel_annotation",
    ],
)
def test_load_graphs(request, risk_obj, network_fixture, annotation_fixture):
    """Test loading all possible combinations of networks and annotations followed by graph loading.

    Args:
        request: The pytest request object to access fixture values.
        risk_obj: The RISK object instance used for loading the network and annotations.
        network_fixture: The name of the fixture to load the network.
        annotation_fixture: The name of the fixture to load the annotations.
    """
    # Load the network using the specified fixture
    network = request.getfixturevalue(network_fixture)
    # Load the annotations using the specified fixture
    annotations = request.getfixturevalue(annotation_fixture)
    neighborhoods = risk_obj.load_neighborhoods(
        network=network,
        annotations=annotations,
        distance_metric="louvain",
        louvain_resolution=8,
        edge_length_threshold=0.75,
        score_metric="stdev",
        null_distribution="network",
        num_permutations=100,  # Perform 100 permutations.
        random_seed=887,
        max_workers=4,  # Use 4 processes.
    )
    graph = risk_obj.load_graph(
        network=network,
        annotations=annotations,
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
    assert len(graph.network.nodes) > 0  # Ensure that the graph has nodes.
    assert len(graph.network.edges) > 0  # Ensure that the graph has edges.
