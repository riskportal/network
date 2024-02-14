from rich import print

from danger.config import read_default_config, validate_config
from danger.network.annotation import define_top_annotations
from danger.network.io import load_cys_network, load_network_annotation
from danger.network.neighborhoods import (
    get_network_neighborhoods,
    define_domains,
    trim_domains,
)
from danger.network.plot import plot_composite_network
from danger.stats.stats import compute_pvalues_by_randomization


class SAFE:
    """SAFE"""

    def __init__(self, network_filepath="", annotation_filepath="", **kwargs):
        """
        Initiate a SAFE instance and define the main settings for analysis.
        The settings are automatically extracted from the specified (or default) INI configuration file.
        Alternatively, each setting can be changed manually after initiation.

        :param path_to_ini_file (str): Path to the configuration file. If not specified, safe_default.ini will be used.
        :param verbose (bool): Defines whether or not intermediate output will be printed out.

        """
        user_input = {
            **kwargs,
            "network_filepath": network_filepath,
            "annotation_filepath": annotation_filepath,
        }
        self.config = validate_config(merge_configs(read_default_config(), user_input))
        # TODO: What if the API is something like twosafe.load_cytoscape_network --> SAFE object
        self.network = self.load_cytoscape_network()
        self.annotation_map = self.load_network_annotation(self.network)
        self.neighborhoods = self.load_neighborhoods(self.network)
        self.neighborhood_enrichment_map = self.get_pvalues(self.neighborhoods, self.annotation_map)
        self.annotation_enrichment_matrix = self.define_top_annotations(
            self.network, self.annotation_map, self.neighborhood_enrichment_map
        )
        self.domains_matrix = self.define_domains(
            self.neighborhood_enrichment_map, self.annotation_enrichment_matrix
        )
        (
            self.annotation_enrichment_matrix,
            self.domains_matrix,
            self.trimmed_domains_matrix,
        ) = self.trim_domains(self.annotation_enrichment_matrix, self.domains_matrix)
        self.plot_composite_network(
            self.network,
            self.neighborhood_enrichment_map,
            self.annotation_enrichment_matrix,
            self.domains_matrix,
            self.trimmed_domains_matrix,
        )

    def load_cytoscape_network(self, *args, **kwargs):
        print("[cyan]Loading [yellow]'cytoscape'[/yellow] [blue]network[/blue]...")
        network_filepath = self.config["network_filepath"]
        return load_cys_network(network_filepath, *args, **kwargs)

    def load_network_annotation(self, network):
        print("[cyan]Loading [yellow]'json'[/yellow] [blue]network annotations[/blue]...")
        annotation = load_network_annotation(
            network, self.config["annotation_filepath"], self.config["annotation_id_colname"]
        )
        return annotation

    def load_neighborhoods(self, network):
        neighborhood_distance_metric = self.config["neighborhood_distance_metric"]
        print(
            f"[cyan]Computing [blue]network neighborhoods[/blue] using [yellow]'{neighborhood_distance_metric}'[/yellow] as the metric..."
        )
        neighborhoods = get_network_neighborhoods(
            network,
            neighborhood_distance_metric,
            self.config["neighborhood_radius"],
            self.config["neighborhood_distance_louvaine_resolution"],
        )
        return neighborhoods

    def get_pvalues(self, neighborhoods, annotation):
        neighborhood_score_metric = self.config["neighborhood_score_metric"]
        network_null_distribution = self.config["network_enrichment_null_distribution"]
        print(
            f"[cyan]Computing [blue]P-values by randomization[/blue] using [yellow]'{network_null_distribution}'[/yellow] as the [blue]null distribution[/blue]..."
        )
        print(
            f"[cyan]Computing [blue]test statistics[/blue] using the [yellow]'{neighborhood_score_metric}'[/yellow]-based [blue]neighborhood scoring[/blue] approach..."
        )
        annotation_matrix = annotation["annotation_matrix"]
        neighborhood_enrichment_map = compute_pvalues_by_randomization(
            neighborhoods,
            annotation_matrix,
            self.config["neighborhood_score_metric"],
            self.config["network_enrichment_direction"],
            self.config["enrichment_alpha_cutoff"],
            null_distribution=network_null_distribution,
            num_permutations=self.config["network_enrichment_num_permutations"],
            random_seed=888,
            multiple_testing=True,
        )
        return neighborhood_enrichment_map

    def define_top_annotations(self, network, annotation_map, neighborhoods_map):
        ordered_column_annotations = annotation_map["ordered_column_annotations"]
        neighborhood_enrichment_sums = neighborhoods_map["neighborhood_enrichment_sums"]
        neighborhood_binary_enrichment_matrix_below_alpha = neighborhoods_map[
            "neighborhood_binary_enrichment_matrix_below_alpha"
        ]
        top_attributes = define_top_annotations(
            network,
            ordered_column_annotations,
            neighborhood_enrichment_sums,
            neighborhood_binary_enrichment_matrix_below_alpha,
            self.config["min_cluster_size"],
            self.config["unimodality_type"],
        )
        return top_attributes

    def define_domains(self, neighborhood_enrichment_map, annotation_enrichment_matrix):
        neighborhood_enrichment_matrix = neighborhood_enrichment_map[
            "neighborhood_enrichment_matrix"
        ]
        neighborhood_binary_enrichment_matrix_below_alpha = neighborhood_enrichment_map[
            "neighborhood_binary_enrichment_matrix_below_alpha"
        ]
        group_distance_metric = self.config["group_distance_metric"]
        group_distance_threshold = self.config["group_distance_threshold"]
        print(
            f"[cyan]Defining [blue]domains[/blue] using the [yellow]'{group_distance_metric}'[/yellow] metric with [yellow]{'automatic' if not group_distance_threshold else group_distance_threshold}[/yellow] distance thresholding..."
        )
        domains_matrix = define_domains(
            neighborhood_enrichment_matrix,
            neighborhood_binary_enrichment_matrix_below_alpha,
            annotation_enrichment_matrix,
            group_distance_metric,
            group_distance_threshold,
        )
        return domains_matrix

    def trim_domains(self, annotation_matrix, domains_matrix):
        print("[cyan]Trimming [blue]domains[/blue]...")
        trimmed_domains = trim_domains(
            annotation_matrix,
            domains_matrix,
            self.config["min_cluster_size"],
        )
        return trimmed_domains

    def plot_composite_network(
        self,
        network,
        neighborhood_enrichment_map,
        annotation_matrix,
        domains_matrix,
        trimmed_domains_matrix,
    ):
        print("[cyan]Plotting [blue]composite network contours[/blue]...")
        neighborhood_enrichment_matrix = neighborhood_enrichment_map[
            "neighborhood_enrichment_matrix"
        ]
        neighborhood_binary_enrichment_matrix_below_alpha = neighborhood_enrichment_map[
            "neighborhood_binary_enrichment_matrix_below_alpha"
        ]
        return plot_composite_network(
            network,
            annotation_matrix,
            domains_matrix,
            trimmed_domains_matrix,
            neighborhood_enrichment_matrix,
            neighborhood_binary_enrichment_matrix_below_alpha,
            self.config["enrichment_max_log10_pvalue"],
        )


def merge_configs(default_config, user_input):
    """
    Merges two configuration dictionaries, overwriting keys in the default
    configuration with those from user input only if the user input value is truthy.

    :param default_config: Dictionary with default configuration settings.
    :param user_input: Dictionary with user-specified configuration settings.
    :return: A new dictionary with the merged configuration.
    """
    merged_config = default_config.copy()  # Start with a copy of the default configuration
    for key, value in user_input.items():
        if value:  # Check if the user_input value is truthy
            merged_config[key] = value  # Only overwrite if truthy
    return merged_config
