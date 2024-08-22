"""
tests/test_load_neighborhoods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

def test_load_neighborhoods_single_process(risk_obj, network, annotations):
    """Test loading neighborhoods using a single process

    Args:
        risk_obj: The RISK object instance used for loading neighborhoods
        network: The network object to be used for neighborhood generation
        annotations: The annotations associated with the network

    Returns:
        None
    """
    # Load neighborhoods with 1 process
    neighborhoods = risk_obj.load_neighborhoods(
        network=network,
        annotations=annotations,
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


def test_load_neighborhoods_multi_process(risk_obj, network, annotations):
    """Test loading neighborhoods using multiple processes

    Args:
        risk_obj: The RISK object instance used for loading neighborhoods
        network: The network object to be used for neighborhood generation
        annotations: The annotations associated with the network

    Returns:
        None
    """
    # Load neighborhoods with 4 processes
    neighborhoods = risk_obj.load_neighborhoods(
        network=network,
        annotations=annotations,
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
