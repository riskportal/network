"""
risk/config
~~~~~~~~~~~
"""

import yaml
from pathlib import Path


def read_default_config():
    """Reads the default settings from config YAML file and update the attributes in the RISK class.

    Returns:
        dict: Config file.
    """
    config_filepath = _find_nearest_git_directory().parent / "config.yaml"
    config = _open_yaml(config_filepath)
    input_config = config["input"]
    network_config = config["network"]
    return {
        # Input
        "network_filepath": input_config["network-filepath"],
        "annotation_filepath": input_config["annotation-filepath"],
        # Network
        "random_seed": network_config["properties"]["random-seed"],
        "network_source_node_label": network_config["properties"]["node"]["source-label"],
        "network_target_node_label": network_config["properties"]["node"]["target-label"],
        "network_min_edges_per_node": network_config["properties"]["node"]["min-edges"],
        "network_edge_weight_label": network_config["properties"]["edge"]["weight-label"],
        "network_enrichment_compute_sphere": network_config["enrichment"]["compute-sphere"],
        "network_enrichment_dimple_factor": network_config["enrichment"]["dimple-factor"],
        "network_enrichment_include_edge_weight": network_config["enrichment"][
            "include-edge-weight"
        ],
        "network_enrichment_null_distribution": network_config["enrichment"]["null-distribution"],
        "network_enrichment_type": network_config["enrichment"]["type"],
        "network_enrichment_num_permutations": network_config["enrichment"]["num-permutations"],
        "network_enrichment_direction": network_config["enrichment"]["direction"],
        "min_cluster_size": network_config["enrichment"]["min-cluster-size"],
        "max_cluster_size": network_config["enrichment"]["max-cluster-size"],
        "enrichment_pval_cutoff": network_config["enrichment"]["pval-cutoff"],
        "enrichment_apply_fdr": network_config["enrichment"]["apply-fdr"],
        "enrichment_fdr_cutoff": network_config["enrichment"]["fdr-cutoff"],
        "neighborhood_distance_metric": network_config["node"]["neighborhood-distance-metric"],
        "neighborhood_distance_louvaine_resolution": network_config["node"][
            "neighborhood-distance-louvaine-resolution"
        ],
        "neighborhood_score_metric": network_config["node"]["neighborhood-score-metric"],
        "neighborhood_diameter": network_config["node"]["neighborhood-diameter"],
        "group_distance_criterion": network_config["node"]["group-distance-criterion"],
        "group_distance_linkage": network_config["node"]["group-distance-linkage"],
        "group_distance_metric": network_config["node"]["group-distance-metric"],
        "unimodality_type": network_config["node"]["unimodality-type"],
    }


def validate_config(config):
    """Test the validity of the current settings in the RISK class before running the analysis.

    Returns:
        dict: Config file.
    """
    # Check if user inputs are found in accepted keywords
    _assert_user_input_in_valid_keywords(
        "Network Enrichment Background",
        user_input=config["network_enrichment_null_distribution"],
        keywords=["annotation", "network"],
    )
    _assert_user_input_in_valid_keywords(
        "Neighborhood Distance Metric",
        user_input=config["neighborhood_distance_metric"],
        keywords=[
            "euclidean",
            "shortpath",
            "louvain",
            "affinity_propagation",
        ],
    )
    _assert_user_input_in_valid_keywords(
        "Neighborhood Distance Metric",
        user_input=config["neighborhood_score_metric"],
        keywords=["sum", "variance", "zscore"],
    )
    _assert_user_input_in_valid_keywords(
        "Annotation Attribute Sign",
        user_input=config["network_enrichment_direction"],
        keywords=["highest", "lowest"],
    )

    # Check numerical user inputs
    assert config["random_seed"] > 0, "Random seed must be a positive integer greater than 0."
    assert (
        config["network_enrichment_num_permutations"] > 99
    ), "Number of enrichment permutations must be greater than or equal to 100."
    assert (
        1 >= config["enrichment_pval_cutoff"] > 0
    ), "Enrichment P-value cutoff must be greater than 0 and less than or equal to 1."
    assert (
        1 >= config["enrichment_fdr_cutoff"] > 0
    ), "Enrichment FDR cutoff must be greater than 0 and less than or equal to 1."
    assert isinstance(
        config["neighborhood_distance_louvaine_resolution"], (int, float)
    ), "Louvaine resolution must be a number."
    assert config["min_cluster_size"] > 1, "The minimum permitted annotation size is 2."
    assert config["max_cluster_size"] < 100_000, "The maximum permitted annotation size is 99,999."

    return config


def _open_yaml(yaml_filepath):
    with open(yaml_filepath, "r", encoding="utf-8") as config:
        yaml_config = yaml.safe_load(config)

    return yaml_config


def _find_nearest_git_directory(start_path=None):
    """Find the full path to the nearest '.git' directory by climbing up the directory tree.

    Args:
        start_path (str or Path, optional): The starting path for the search. If not provided,
            the current working directory is used.

    Returns:
        Path: The full path to the '.git' directory if found, or current working directory.
    """
    # If start_path is not provided, use the current working directory
    cwd = Path.cwd()
    start_path = Path(start_path) if start_path else cwd
    # Iterate through parent directories until .git is found
    current_path = start_path
    while current_path:
        git_path = current_path / ".git"
        if git_path.is_dir():
            return git_path.resolve()
        current_path = current_path.parent

    # If .git is not found in any parent directory, return None
    return cwd


def _assert_user_input_in_valid_keywords(input_class, user_input, keywords):
    assert (
        user_input in keywords
    ), f"{user_input} is not a valid input for {input_class}. Valid options are: {', '.join(keywords)}"
