import yaml
from pathlib import Path


def read_default_config():
    """Reads the default settings from config YAML file and update the attributes in the SAFE class.

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
        "annotation_id_colname": input_config["annotation-id-colname"],
        # Network
        "network_enrichment_background": network_config["enrichment"]["background"],
        "network_enrichment_type": network_config["enrichment"]["type"],
        "network_enrichment_num_permutations": network_config["enrichment"]["num-permutations"],
        "network_enrichment_direction": network_config["enrichment"]["direction"],
        "min_cluster_size": network_config["enrichment"]["min-cluster-size"],
        "enrichment_max_log10_pvalue": network_config["enrichment"]["max-log10-pvalue"],
        "enrichment_alpha_cutoff": network_config["enrichment"]["alpha-cutoff"],
        "neighborhood_distance_metric": network_config["node"]["neighborhood-distance-metric"],
        "neighborhood_score_metric": network_config["node"]["neighborhood-score-metric"],
        "neighborhood_radius": network_config["node"]["neighborhood-radius"],
        "neighborhood_radius_type": network_config["node"]["neighborhood-radius-type"],
        "group_distance_metric": network_config["node"]["group-distance-metric"],
        "group_distance_threshold": network_config["node"]["group-distance-threshold"],
        "unimodality_type": network_config["node"]["unimodality-type"],
    }


def validate_config(config):
    """Test the validity of the current settings in the SAFE class before running the analysis.

    Returns:
        dict: Config file.
    """
    # Check if user inputs are found in accepted keywords
    _assert_user_input_in_valid_keywords(
        "Network Enrichment Background",
        user_input=config["network_enrichment_background"],
        keywords=["annotation_file", "network"],
    )
    _assert_user_input_in_valid_keywords(
        "Neighborhood Distance Metric",
        user_input=config["neighborhood_distance_metric"],
        keywords=["euclidean", "shortpath", "shortpath_weighted_layout"],
    )
    _assert_user_input_in_valid_keywords(
        "Neighborhood Distance Metric",
        user_input=config["neighborhood_score_metric"],
        keywords=["sum", "variance", "zscore"],
    )
    _assert_user_input_in_valid_keywords(
        "Annotation Attribute Sign",
        user_input=config["network_enrichment_direction"],
        keywords=["highest", "lowest", "both"],
    )

    # Check numerical user inputs
    assert (
        config["network_enrichment_num_permutations"] > 99
    ), "Number of enrichment permutations must be greater than or equal to 100."
    assert (
        1 > config["enrichment_alpha_cutoff"] > 0
    ), "Enrichment alpha cutoff must be between 0 and 1."
    assert isinstance(
        config["enrichment_max_log10_pvalue"], (int, float)
    ), "Maximum enrichment Log10 P-value must be a number."
    assert config["min_cluster_size"] > 1, "The minimum permitted annotation size is 2."
    assert (
        1 > config["group_distance_threshold"] > 0
    ), "Group distance threshold must be between 0 and 1."

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
