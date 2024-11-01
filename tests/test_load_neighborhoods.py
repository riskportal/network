"""
tests/test_load_neighborhoods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import pytest


@pytest.mark.parametrize("null_distribution", ["network", "annotations"])
def test_load_neighborhoods_single_process(
    risk_obj, cytoscape_network, json_annotation, null_distribution
):
    """Test loading neighborhoods using a single process with the permutation test with multiple
    null distributions.

    Args:
        risk_obj: The RISK object instance used for loading neighborhoods.
        cytoscape_network: The network object to be used for neighborhood generation.
        json_annotation: The annotations associated with the network.
        null_distribution: Null distribution type for the permutation test (either 'network' or 'annotations').
    """
    # Load neighborhoods with 1 process
    neighborhoods = risk_obj.load_neighborhoods_by_permutation(
        network=cytoscape_network,
        annotations=json_annotation,
        distance_metric="louvain",
        louvain_resolution=0.01,
        edge_length_threshold=0.25,
        score_metric="stdev",
        null_distribution=null_distribution,
        num_permutations=10,  # Set to 10 permutations as requested
        random_seed=887,
        max_workers=1,  # Single process
    )

    assert neighborhoods is not None
    assert len(neighborhoods) > 0  # Ensure neighborhoods are loaded


def test_load_neighborhoods_permutation_multi_process(risk_obj, cytoscape_network, json_annotation):
    """Test loading neighborhoods using multiple processes with the permutation test.

    Args:
        risk_obj: The RISK object instance used for loading neighborhoods.
        cytoscape_network: The network object to be used for neighborhood generation.
        json_annotation: The annotations associated with the network.
    """
    # Load neighborhoods with 4 processes
    neighborhoods = risk_obj.load_neighborhoods_by_permutation(
        network=cytoscape_network,
        annotations=json_annotation,
        distance_metric="louvain",
        louvain_resolution=0.01,
        edge_length_threshold=0.25,
        score_metric="stdev",
        null_distribution="network",
        num_permutations=10,  # Set to 10 permutations as requested
        random_seed=887,
        max_workers=4,  # Four processes
    )

    assert neighborhoods is not None
    assert len(neighborhoods) > 0  # Ensure neighborhoods are loaded


@pytest.mark.parametrize("null_distribution", ["network", "annotations"])
def test_load_neighborhoods_hypergeom(
    risk_obj, cytoscape_network, json_annotation, null_distribution
):
    """Test loading neighborhoods using the hypergeometric test with multiple null distributions.

    Args:
        risk_obj: The RISK object instance used for loading neighborhoods.
        cytoscape_network: The network object to be used for neighborhood generation.
        json_annotation: The annotations associated with the network.
        null_distribution: Null distribution type for the hypergeometric test (either 'network' or 'annotations').
    """
    neighborhoods = risk_obj.load_neighborhoods_by_hypergeom(
        network=cytoscape_network,
        annotations=json_annotation,
        distance_metric="louvain",
        louvain_resolution=0.01,
        edge_length_threshold=0.25,
        null_distribution=null_distribution,
        random_seed=887,
    )

    assert neighborhoods is not None
    assert len(neighborhoods) > 0  # Ensure neighborhoods are loaded


@pytest.mark.parametrize("null_distribution", ["network", "annotations"])
def test_load_neighborhoods_poisson(
    risk_obj, cytoscape_network, json_annotation, null_distribution
):
    """Test loading neighborhoods using the Poisson test with multiple null distributions.

    Args:
        risk_obj: The RISK object instance used for loading neighborhoods.
        cytoscape_network: The network object to be used for neighborhood generation.
        json_annotation: The annotations associated with the network.
        null_distribution: Null distribution type for the Poisson test (either 'network' or 'annotations').
    """
    neighborhoods = risk_obj.load_neighborhoods_by_poisson(
        network=cytoscape_network,
        annotations=json_annotation,
        distance_metric="louvain",
        louvain_resolution=0.01,
        edge_length_threshold=0.15,
        null_distribution=null_distribution,
        random_seed=887,
    )

    assert neighborhoods is not None
    assert len(neighborhoods) > 0  # Ensure neighborhoods are loaded


@pytest.mark.parametrize(
    "distance_metric, edge_length_threshold",
    [
        ("greedy_modularity", 0.75),
        ("louvain", 0.80),
        ("label_propagation", 0.70),
        ("markov_clustering", 0.65),
        ("walktrap", 0.85),
        ("spinglass", 0.90),
        (["louvain"], [0.75]),
        (["louvain", "label_propagation"], [0.75, 0.70]),
        (["louvain", "markov_clustering"], [0.75, 0.65]),
        (["label_propagation", "walktrap", "spinglass"], [0.70, 0.85, 0.90]),
        (
            ["louvain", "label_propagation", "markov_clustering", "walktrap", "spinglass"],
            [0.75, 0.70, 0.65, 0.85, 0.90],
        ),
        (
            [
                "louvain",
                "label_propagation",
                "markov_clustering",
                "walktrap",
                "spinglass",
                "greedy_modularity",
            ],
            [0.75, 0.70, 0.65, 0.85, 0.90, 0.80],
        ),
    ],
)
def test_load_neighborhoods_with_various_distance_metrics(
    risk_obj, cytoscape_network, json_annotation, distance_metric, edge_length_threshold
):
    """Test loading neighborhoods using various distance metrics with matching edge length thresholds.

    Args:
        risk_obj: The RISK object instance used for loading neighborhoods.
        cytoscape_network: The network object to be used for neighborhood generation.
        json_annotation: The annotations associated with the network.
        distance_metric: The specific distance metric(s) to be used for generating neighborhoods.
        edge_length_threshold: The edge length threshold(s) corresponding to each distance metric.
    """
    # Load neighborhoods with the current distance metric(s) and matching edge length threshold(s)
    neighborhoods = risk_obj.load_neighborhoods_by_permutation(
        network=cytoscape_network,
        annotations=json_annotation,
        distance_metric=distance_metric,
        louvain_resolution=8,
        edge_length_threshold=edge_length_threshold,
        score_metric="stdev",
        null_distribution="network",
        num_permutations=100,
        random_seed=887,
        max_workers=4,
    )

    assert neighborhoods is not None
    assert len(neighborhoods) > 0  # Ensure neighborhoods are loaded


@pytest.mark.parametrize("score_metric", ["sum", "stdev"])
def test_load_neighborhoods_with_various_score_metrics(
    risk_obj, cytoscape_network, json_annotation, score_metric
):
    """Test loading neighborhoods using various score metrics.

    Args:
        risk_obj: The RISK object instance used for loading neighborhoods.
        cytoscape_network: The network object to be used for neighborhood generation.
        json_annotation: The annotations associated with the network.
        score_metric: The specific score metric to be used for generating neighborhoods.
    """
    # Load neighborhoods with the specified score metric
    neighborhoods = risk_obj.load_neighborhoods_by_permutation(
        network=cytoscape_network,
        annotations=json_annotation,
        distance_metric="louvain",  # Using markov_clustering as the distance metric
        edge_length_threshold=0.75,
        score_metric=score_metric,
        null_distribution="network",
        num_permutations=100,
        random_seed=887,
        max_workers=4,
    )

    assert neighborhoods is not None
    assert len(neighborhoods) > 0  # Ensure neighborhoods are loaded


@pytest.mark.parametrize("null_distribution", ["network", "annotations"])
def test_load_neighborhoods_with_various_null_distributions(
    risk_obj, cytoscape_network, json_annotation, null_distribution
):
    """Test loading neighborhoods using various null distributions.

    Args:
        risk_obj: The RISK object instance used for loading neighborhoods.
        cytoscape_network: The network object to be used for neighborhood generation.
        json_annotation: The annotations associated with the network.
        null_distribution: The specific null distribution to be used for generating neighborhoods.
    """
    # Load neighborhoods with the specified null distribution
    neighborhoods = risk_obj.load_neighborhoods_by_permutation(
        network=cytoscape_network,
        annotations=json_annotation,
        distance_metric="louvain",  # Using markov_clustering as the distance metric
        edge_length_threshold=0.75,
        score_metric="stdev",  # Using stdev as the score metric
        null_distribution=null_distribution,  # Parametrized null distribution
        num_permutations=100,
        random_seed=887,
        max_workers=4,
    )

    assert neighborhoods is not None
    assert len(neighborhoods) > 0  # Ensure neighborhoods are loaded
