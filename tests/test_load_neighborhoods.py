"""
tests/test_load_neighborhoods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import networkx as nx
import numpy as np
import pytest


@pytest.mark.parametrize("null_distribution", ["network", "annotation"])
def test_load_neighborhoods_binom(risk_obj, cytoscape_network, json_annotation, null_distribution):
    """
    Test loading neighborhoods using the binomial test with multiple null distributions.

    Args:
        risk_obj: The RISK object instance used for loading neighborhoods.
        cytoscape_network: The network object to be used for neighborhood generation.
        json_annotation: The annotation associated with the network.
        null_distribution: Null distribution type for the binomial test (either 'network' or 'annotation').
    """
    neighborhoods = risk_obj.load_neighborhoods_binom(
        network=cytoscape_network,
        annotation=json_annotation,
        distance_metric="louvain",
        louvain_resolution=0.01,
        fraction_shortest_edges=0.25,
        null_distribution=null_distribution,
        random_seed=887,
    )

    assert neighborhoods is not None
    assert len(neighborhoods) > 0  # Ensure neighborhoods are loaded


@pytest.mark.parametrize("null_distribution", ["network", "annotation"])
def test_load_neighborhoods_chi2(risk_obj, cytoscape_network, json_annotation, null_distribution):
    """
    Test loading neighborhoods using the chi-squared test with multiple null distributions.

    Args:
        risk_obj: The RISK object instance used for loading neighborhoods.
        cytoscape_network: The network object to be used for neighborhood generation.
        json_annotation: The annotation associated with the network.
        null_distribution: Null distribution type for the chi-squared test (either 'network' or 'annotation').
    """
    neighborhoods = risk_obj.load_neighborhoods_chi2(
        network=cytoscape_network,
        annotation=json_annotation,
        distance_metric="louvain",
        louvain_resolution=0.01,
        fraction_shortest_edges=0.25,
        null_distribution=null_distribution,
        random_seed=887,
    )

    assert neighborhoods is not None
    assert len(neighborhoods) > 0  # Ensure neighborhoods are loaded


@pytest.mark.parametrize("null_distribution", ["network", "annotation"])
def test_load_neighborhoods_hypergeom(
    risk_obj, cytoscape_network, json_annotation, null_distribution
):
    """
    Test loading neighborhoods using the hypergeometric test with multiple null distributions.

    Args:
        risk_obj: The RISK object instance used for loading neighborhoods.
        cytoscape_network: The network object to be used for neighborhood generation.
        json_annotation: The annotation associated with the network.
        null_distribution: Null distribution type for the hypergeometric test (either 'network' or 'annotation').
    """
    neighborhoods = risk_obj.load_neighborhoods_hypergeom(
        network=cytoscape_network,
        annotation=json_annotation,
        distance_metric="louvain",
        louvain_resolution=0.01,
        fraction_shortest_edges=0.25,
        null_distribution=null_distribution,
        random_seed=887,
    )

    assert neighborhoods is not None
    assert len(neighborhoods) > 0  # Ensure neighborhoods are loaded


@pytest.mark.parametrize("null_distribution", ["network", "annotation"])
def test_load_neighborhoods_permutation_single_process(
    risk_obj, cytoscape_network, json_annotation, null_distribution
):
    """
    Test loading neighborhoods using a single process with the permutation test with multiple
    null distributions.

    Args:
        risk_obj: The RISK object instance used for loading neighborhoods.
        cytoscape_network: The network object to be used for neighborhood generation.
        json_annotation: The annotation associated with the network.
        null_distribution: Null distribution type for the permutation test (either 'network' or 'annotation').
    """
    # Load neighborhoods with 1 process
    neighborhoods = risk_obj.load_neighborhoods_permutation(
        network=cytoscape_network,
        annotation=json_annotation,
        distance_metric="leiden",
        louvain_resolution=0.01,
        leiden_resolution=1.0,
        fraction_shortest_edges=0.25,
        score_metric="stdev",
        null_distribution=null_distribution,
        num_permutations=10,  # Set to 10 permutations as requested
        random_seed=887,
        max_workers=1,  # Single process
    )

    assert neighborhoods is not None
    assert len(neighborhoods) > 0  # Ensure neighborhoods are loaded


def test_load_neighborhoods_permutation_multi_process(risk_obj, cytoscape_network, json_annotation):
    """
    Test loading neighborhoods using multiple processes with the permutation test.

    Args:
        risk_obj: The RISK object instance used for loading neighborhoods.
        cytoscape_network: The network object to be used for neighborhood generation.
        json_annotation: The annotation associated with the network.
    """
    # Load neighborhoods with 4 processes
    neighborhoods = risk_obj.load_neighborhoods_permutation(
        network=cytoscape_network,
        annotation=json_annotation,
        distance_metric="louvain",
        louvain_resolution=0.01,
        fraction_shortest_edges=0.25,
        score_metric="stdev",
        null_distribution="network",
        num_permutations=10,  # Set to 10 permutations as requested
        random_seed=887,
        max_workers=4,  # Four processes
    )

    assert neighborhoods is not None
    assert len(neighborhoods) > 0  # Ensure neighborhoods are loaded


@pytest.mark.parametrize("null_distribution", ["network", "annotation"])
def test_load_neighborhoods_poisson(
    risk_obj, cytoscape_network, json_annotation, null_distribution
):
    """
    Test loading neighborhoods using the Poisson test with multiple null distributions.

    Args:
        risk_obj: The RISK object instance used for loading neighborhoods.
        cytoscape_network: The network object to be used for neighborhood generation.
        json_annotation: The annotation associated with the network.
        null_distribution: Null distribution type for the Poisson test (either 'network' or 'annotation').
    """
    neighborhoods = risk_obj.load_neighborhoods_poisson(
        network=cytoscape_network,
        annotation=json_annotation,
        distance_metric="louvain",
        louvain_resolution=0.01,
        fraction_shortest_edges=0.15,
        null_distribution=null_distribution,
        random_seed=887,
    )

    assert neighborhoods is not None
    assert len(neighborhoods) > 0  # Ensure neighborhoods are loaded


@pytest.mark.parametrize("null_distribution", ["network", "annotation"])
def test_load_neighborhoods_zscore(risk_obj, cytoscape_network, json_annotation, null_distribution):
    """
    Test loading neighborhoods using the z-score test with multiple null distributions.

    Args:
        risk_obj: The RISK object instance used for loading neighborhoods.
        cytoscape_network: The network object to be used for neighborhood generation.
        json_annotation: The annotation associated with the network.
        null_distribution: Null distribution type for the z-score test (either 'network' or 'annotation').
    """
    neighborhoods = risk_obj.load_neighborhoods_zscore(
        network=cytoscape_network,
        annotation=json_annotation,
        distance_metric="louvain",
        louvain_resolution=0.01,
        fraction_shortest_edges=0.25,
        null_distribution=null_distribution,
        random_seed=887,
    )

    assert neighborhoods is not None
    assert len(neighborhoods) > 0  # Ensure neighborhoods are loaded


@pytest.mark.parametrize(
    "distance_metric, fraction_shortest_edges",
    [
        ("greedy_modularity", 0.75),
        ("louvain", 0.80),
        ("leiden", 0.85),
        ("label_propagation", 0.70),
        ("markov_clustering", 0.65),
        ("walktrap", 0.85),
        ("spinglass", 0.90),
        (["louvain"], [0.75]),
        (["louvain", "label_propagation"], [0.75, 0.70]),
        (["louvain", "markov_clustering"], [0.75, 0.65]),
        (["label_propagation", "walktrap", "spinglass"], [0.70, 0.85, 0.90]),
        (
            [
                "louvain",
                "label_propagation",
                "markov_clustering",
                "walktrap",
                "spinglass",
                "leiden",
            ],
            [0.75, 0.70, 0.65, 0.85, 0.90, 0.50],
        ),
        (
            [
                "louvain",
                "leiden",
                "label_propagation",
                "markov_clustering",
                "walktrap",
                "spinglass",
                "greedy_modularity",
            ],
            [0.75, 0.70, 0.65, 0.85, 0.90, 0.80, 0.90],
        ),
    ],
)
def test_load_neighborhoods_with_various_distance_metrics(
    risk_obj, cytoscape_network, json_annotation, distance_metric, fraction_shortest_edges
):
    """
    Test loading neighborhoods using various distance metrics with matching edge length thresholds.

    Args:
        risk_obj: The RISK object instance used for loading neighborhoods.
        cytoscape_network: The network object to be used for neighborhood generation.
        json_annotation: The annotation associated with the network.
        distance_metric: The specific distance metric(s) to be used for generating neighborhoods.
        fraction_shortest_edges: The edge length threshold(s) corresponding to each distance metric.
    """
    # Load neighborhoods with the current distance metric(s) and matching edge length threshold(s)
    neighborhoods = risk_obj.load_neighborhoods_permutation(
        network=cytoscape_network,
        annotation=json_annotation,
        distance_metric=distance_metric,
        louvain_resolution=8,
        fraction_shortest_edges=fraction_shortest_edges,
        score_metric="stdev",
        null_distribution="network",
        num_permutations=20,
        random_seed=887,
        max_workers=1,
    )

    assert neighborhoods is not None
    assert len(neighborhoods) > 0  # Ensure neighborhoods are loaded


@pytest.mark.parametrize("score_metric", ["sum", "stdev"])
def test_load_neighborhoods_with_various_score_metrics(
    risk_obj, cytoscape_network, json_annotation, score_metric
):
    """
    Test loading neighborhoods using various score metrics.

    Args:
        risk_obj: The RISK object instance used for loading neighborhoods.
        cytoscape_network: The network object to be used for neighborhood generation.
        json_annotation: The annotation associated with the network.
        score_metric: The specific score metric to be used for generating neighborhoods.
    """
    # Load neighborhoods with the specified score metric
    neighborhoods = risk_obj.load_neighborhoods_permutation(
        network=cytoscape_network,
        annotation=json_annotation,
        distance_metric="louvain",  # Using louvain as the distance metric
        louvain_resolution=8,
        fraction_shortest_edges=0.75,
        score_metric=score_metric,
        null_distribution="network",
        num_permutations=20,
        random_seed=887,
        max_workers=1,
    )

    assert neighborhoods is not None
    assert len(neighborhoods) > 0  # Ensure neighborhoods are loaded


@pytest.mark.parametrize("null_distribution", ["network", "annotation"])
def test_load_neighborhoods_with_various_null_distributions(
    risk_obj, cytoscape_network, json_annotation, null_distribution
):
    """
    Test loading neighborhoods using various null distributions.

    Args:
        risk_obj: The RISK object instance used for loading neighborhoods.
        cytoscape_network: The network object to be used for neighborhood generation.
        json_annotation: The annotation associated with the network.
        null_distribution: The specific null distribution to be used for generating neighborhoods.
    """
    # Load neighborhoods with the specified null distribution
    neighborhoods = risk_obj.load_neighborhoods_permutation(
        network=cytoscape_network,
        annotation=json_annotation,
        distance_metric="louvain",  # Using markov_clustering as the distance metric
        fraction_shortest_edges=0.75,
        score_metric="stdev",  # Using stdev as the score metric
        null_distribution=null_distribution,  # Parametrized null distribution
        num_permutations=20,
        random_seed=887,
        max_workers=1,
    )

    assert neighborhoods is not None
    assert len(neighborhoods) > 0  # Ensure neighborhoods are loaded


@pytest.mark.parametrize("null_distribution", ["network", "annotation"])
def test_load_neighborhoods_structure(
    risk_obj, cytoscape_network, json_annotation, null_distribution
):
    """Test the structure of the neighborhoods object."""
    neighborhoods = risk_obj.load_neighborhoods_permutation(
        network=cytoscape_network,
        annotation=json_annotation,
        distance_metric="louvain",
        louvain_resolution=8,
        fraction_shortest_edges=0.75,
        score_metric="stdev",
        null_distribution=null_distribution,
        num_permutations=10,
        random_seed=887,
        max_workers=1,
    )

    # Validate that the neighborhoods object has the expected keys
    assert (
        "depletion_pvals" in neighborhoods
    ), "Neighborhoods should contain a 'depletion_pvals' key"
    assert (
        "enrichment_pvals" in neighborhoods
    ), "Neighborhoods should contain an 'enrichment_pvals' key"
    assert isinstance(
        neighborhoods["depletion_pvals"], np.ndarray
    ), "'depletion_pvals' should be a numpy array"
    assert isinstance(
        neighborhoods["enrichment_pvals"], np.ndarray
    ), "'enrichment_pvals' should be a numpy array"


def test_load_neighborhoods_empty_network(risk_obj, json_annotation):
    """Test loading neighborhoods with an empty network."""
    # Create an empty network
    empty_network = nx.Graph()

    # Expect a ValueError due to missing edge lengths
    with pytest.raises(
        ValueError,
        match="No edge lengths found in the graph. Ensure edges have 'length' attributes.",
    ):
        risk_obj.load_neighborhoods_permutation(
            network=empty_network,
            annotation=json_annotation,
            distance_metric="louvain",
            louvain_resolution=8,
            fraction_shortest_edges=0.75,
            score_metric="stdev",
            null_distribution="network",
            num_permutations=10,
            random_seed=887,
            max_workers=1,
        )


@pytest.mark.parametrize("null_distribution", ["network", "annotation"])
def test_load_neighborhoods_output_dimensions(
    risk_obj, cytoscape_network, json_annotation, null_distribution
):
    """
    Test that the output dimensions of neighborhoods match expectations.

    Args:
        risk_obj: The RISK object instance used for loading neighborhoods.
        cytoscape_network: The network object to be used for neighborhood generation.
        json_annotation: The annotation associated with the network.
        null_distribution: The specific null distribution to be used for generating neighborhoods.
    """
    neighborhoods = risk_obj.load_neighborhoods_permutation(
        network=cytoscape_network,
        annotation=json_annotation,
        distance_metric="louvain",
        louvain_resolution=8,
        fraction_shortest_edges=0.75,
        score_metric="stdev",
        null_distribution=null_distribution,
        num_permutations=10,
        random_seed=887,
        max_workers=1,
    )

    # Validate dimensions of p-value matrices
    num_nodes = len(cytoscape_network.nodes)
    num_annotation = len(json_annotation["ordered_annotation"])
    assert neighborhoods["depletion_pvals"].shape == (
        num_nodes,
        num_annotation,
    ), "Depletion p-values matrix dimensions do not match the expected size"
    assert neighborhoods["enrichment_pvals"].shape == (
        num_nodes,
        num_annotation,
    ), "Enrichment p-values matrix dimensions do not match the expected size"


def test_load_neighborhoods_deterministic_output(risk_obj, cytoscape_network, json_annotation):
    """
    Test that loading neighborhoods with the same random seed produces consistent results.

    Args:
        risk_obj: The RISK object instance used for loading neighborhoods.
        cytoscape_network: The network object to be used for neighborhood generation.
        json_annotation: The annotation associated with the network.
    """
    neighborhoods_1 = risk_obj.load_neighborhoods_permutation(
        network=cytoscape_network,
        annotation=json_annotation,
        distance_metric="louvain",
        louvain_resolution=8,
        fraction_shortest_edges=0.75,
        score_metric="stdev",
        null_distribution="network",
        num_permutations=10,
        random_seed=887,
        max_workers=1,
    )
    neighborhoods_2 = risk_obj.load_neighborhoods_permutation(
        network=cytoscape_network,
        annotation=json_annotation,
        distance_metric="louvain",
        louvain_resolution=8,
        fraction_shortest_edges=0.75,
        score_metric="stdev",
        null_distribution="network",
        num_permutations=10,
        random_seed=887,  # Same seed
        max_workers=1,
    )

    # Validate that the outputs are identical
    assert np.array_equal(
        neighborhoods_1["depletion_pvals"], neighborhoods_2["depletion_pvals"]
    ), "Depletion p-values should be identical for the same random seed"
    assert np.array_equal(
        neighborhoods_1["enrichment_pvals"], neighborhoods_2["enrichment_pvals"]
    ), "Enrichment p-values should be identical for the same random seed"
