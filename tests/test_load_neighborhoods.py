"""
tests/test_load_neighborhoods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import pytest


def test_load_neighborhoods_single_process(risk_obj, cytoscape_network, json_annotation):
    """Test loading neighborhoods using a single process with the permutation test.

    Args:
        risk_obj: The RISK object instance used for loading neighborhoods.
        cytoscape_network: The network object to be used for neighborhood generation.
        json_annotation: The annotations associated with the network.

    Returns:
        None
    """
    # Load neighborhoods with 1 process
    neighborhoods = risk_obj.load_neighborhoods_by_permutation(
        network=cytoscape_network,
        annotations=json_annotation,
        distance_metric="markov_clustering",
        louvain_resolution=0.01,
        edge_length_threshold=0.25,
        score_metric="stdev",
        null_distribution="network",
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

    Returns:
        None
    """
    # Load neighborhoods with 4 processes
    neighborhoods = risk_obj.load_neighborhoods_by_permutation(
        network=cytoscape_network,
        annotations=json_annotation,
        distance_metric="markov_clustering",
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


def test_load_neighborhoods_fisher_exact_single_process(
    risk_obj, cytoscape_network, json_annotation
):
    """Test loading neighborhoods using a single process with Fisher's exact test.

    Args:
        risk_obj: The RISK object instance used for loading neighborhoods.
        cytoscape_network: The network object to be used for neighborhood generation.
        json_annotation: The annotations associated with the network.

    Returns:
        None
    """
    # Load neighborhoods with Fisher's exact test using 1 process
    neighborhoods = risk_obj.load_neighborhoods_by_fisher_exact(
        network=cytoscape_network,
        annotations=json_annotation,
        distance_metric="markov_clustering",
        louvain_resolution=0.01,
        edge_length_threshold=0.25,
        random_seed=887,
        max_workers=1,  # Single process
    )

    assert neighborhoods is not None
    assert len(neighborhoods) > 0  # Ensure neighborhoods are loaded


def test_load_neighborhoods_fisher_exact_multi_process(
    risk_obj, cytoscape_network, json_annotation
):
    """Test loading neighborhoods using multiple processes with Fisher's exact test.

    Args:
        risk_obj: The RISK object instance used for loading neighborhoods.
        cytoscape_network: The network object to be used for neighborhood generation.
        json_annotation: The annotations associated with the network.

    Returns:
        None
    """
    # Load neighborhoods with Fisher's exact test using 4 processes
    neighborhoods = risk_obj.load_neighborhoods_by_fisher_exact(
        network=cytoscape_network,
        annotations=json_annotation,
        distance_metric="markov_clustering",
        louvain_resolution=0.01,
        edge_length_threshold=0.25,
        random_seed=887,
        max_workers=4,  # Four processes
    )

    assert neighborhoods is not None
    assert len(neighborhoods) > 0  # Ensure neighborhoods are loaded


def test_load_neighborhoods_hypergeom_single_process(risk_obj, cytoscape_network, json_annotation):
    """Test loading neighborhoods using a single process with the hypergeometric test.

    Args:
        risk_obj: The RISK object instance used for loading neighborhoods.
        cytoscape_network: The network object to be used for neighborhood generation.
        json_annotation: The annotations associated with the network.

    Returns:
        None
    """
    # Load neighborhoods with the hypergeometric test using 1 process
    neighborhoods = risk_obj.load_neighborhoods_by_hypergeom(
        network=cytoscape_network,
        annotations=json_annotation,
        distance_metric="markov_clustering",
        louvain_resolution=0.01,
        edge_length_threshold=0.25,
        random_seed=887,
        max_workers=1,  # Single process
    )

    assert neighborhoods is not None
    assert len(neighborhoods) > 0  # Ensure neighborhoods are loaded


def test_load_neighborhoods_hypergeom_multi_process(risk_obj, cytoscape_network, json_annotation):
    """Test loading neighborhoods using multiple processes with the hypergeometric test.

    Args:
        risk_obj: The RISK object instance used for loading neighborhoods.
        cytoscape_network: The network object to be used for neighborhood generation.
        json_annotation: The annotations associated with the network.

    Returns:
        None
    """
    # Load neighborhoods with the hypergeometric test using 4 processes
    neighborhoods = risk_obj.load_neighborhoods_by_hypergeom(
        network=cytoscape_network,
        annotations=json_annotation,
        distance_metric="markov_clustering",
        louvain_resolution=0.01,
        edge_length_threshold=0.25,
        random_seed=887,
        max_workers=4,  # Four processes
    )

    assert neighborhoods is not None
    assert len(neighborhoods) > 0  # Ensure neighborhoods are loaded


import pytest


@pytest.mark.parametrize(
    "distance_metric",
    ["dijkstra", "louvain", "label_propagation", "markov_clustering", "walktrap", "spinglass"],
)
def test_load_neighborhoods_with_various_distance_metrics(
    risk_obj, cytoscape_network, json_annotation, distance_metric
):
    """Test loading neighborhoods using various distance metrics.

    Args:
        risk_obj: The RISK object instance used for loading neighborhoods.
        cytoscape_network: The network object to be used for neighborhood generation.
        json_annotation: The annotations associated with the network.
        distance_metric: The specific distance metric to be used for generating neighborhoods.

    Returns:
        None
    """
    # Load neighborhoods with the current distance metric
    if distance_metric == "louvain":
        neighborhoods = risk_obj.load_neighborhoods_by_permutation(
            network=cytoscape_network,
            annotations=json_annotation,
            distance_metric=distance_metric,
            louvain_resolution=8,
            edge_length_threshold=0.75,
            score_metric="stdev",
            null_distribution="network",
            num_permutations=100,
            random_seed=887,
            max_workers=4,
        )
    else:
        neighborhoods = risk_obj.load_neighborhoods_by_permutation(
            network=cytoscape_network,
            annotations=json_annotation,
            distance_metric=distance_metric,
            edge_length_threshold=0.75,
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

    Returns:
        None
    """
    # Load neighborhoods with the specified score metric
    neighborhoods = risk_obj.load_neighborhoods_by_permutation(
        network=cytoscape_network,
        annotations=json_annotation,
        distance_metric="markov_clustering",  # Using markov_clustering as the distance metric
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

    Returns:
        None
    """
    # Load neighborhoods with the specified null distribution
    neighborhoods = risk_obj.load_neighborhoods_by_permutation(
        network=cytoscape_network,
        annotations=json_annotation,
        distance_metric="markov_clustering",  # Using markov_clustering as the distance metric
        edge_length_threshold=0.75,
        score_metric="stdev",  # Using stdev as the score metric
        null_distribution=null_distribution,  # Parametrized null distribution
        num_permutations=100,
        random_seed=887,
        max_workers=4,
    )

    assert neighborhoods is not None
    assert len(neighborhoods) > 0  # Ensure neighborhoods are loaded
