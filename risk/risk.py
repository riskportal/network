import os
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from rich import print
from rich.progress import Progress
from scipy import ndimage

from risk.config import read_default_config, validate_config
from risk.network.annotation import define_top_annotations
from risk.network.graph import calculate_edge_lengths
from risk.network.io import load_cys_network, load_network_annotation
from risk.network.neighborhoods import (
    define_domains,
    get_network_neighborhoods,
    trim_domains,
)
from risk.network.plot import plot_composite_network
from risk.stats.stats import compute_pvalues_by_randomization

from scipy.stats import mode
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform
import numpy as np
import networkx as nx
from scipy.ndimage import label


class RISK:
    """RISK API"""

    def __init__(self, network_filepath="", annotation_filepath="", **kwargs):
        user_input = {
            **kwargs,
            "network_filepath": network_filepath,
            "annotation_filepath": annotation_filepath,
        }
        self.config = validate_config(merge_configs(read_default_config(), user_input))
        # TODO: What if the API is something like twosafe.load_cytoscape_network --> RISK object
        self.network = self.load_cytoscape_network()
        self.annotation_map = self.load_network_annotation(self.network)
        self.tune_diameter_in_config(self.network)
        # ==========
        self.network, self.neighborhoods = self.tune_graph_and_get_neighborhoods(self.network)
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
        # ==========
        self.plot_composite_network(
            self.network,
            self.neighborhood_enrichment_map,
            self.annotation_enrichment_matrix,
            self.domains_matrix,
            self.trimmed_domains_matrix,
        )

    def load_cytoscape_network(self):
        network_filepath = self.config["network_filepath"]
        network_filename = Path(network_filepath).name
        network_enrichment_compute_sphere = self.config["network_enrichment_compute_sphere"]
        network_enrichment_dimple_factor = self.config["network_enrichment_dimple_factor"]
        network_min_edges_per_node = self.config["network_min_edges_per_node"]
        network_enrichment_include_edge_weight = self.config[
            "network_enrichment_include_edge_weight"
        ]
        for_print_edge_weight = (
            "[red]with[/red]" if network_enrichment_include_edge_weight else "[red]without[/red]"
        )
        for_print_sphere = (
            f"[yellow]3D[/yellow] [cyan]with[/cyan] [blue]dimple factor[/blue] [yellow]{network_enrichment_dimple_factor}[/yellow]"
            if network_enrichment_compute_sphere
            else "[yellow]2D[/yellow]"
        )
        print(
            f"[cyan]Loading [yellow]Cytoscape[/yellow] [blue]network file[/blue]: [yellow]'{network_filename}'[/yellow]...[/cyan]"
            f"\n[cyan]Removing [blue]nodes[/blue] with [blue]fewer[/blue] than [red]{network_min_edges_per_node}[/red] [blue]{'edge' if network_min_edges_per_node == 1 else 'edges'}...[/blue][/cyan]"
            f"\n[cyan]Treating the network as {for_print_sphere} {for_print_edge_weight} [yellow]edge weights[/yellow]...[/cyan]"
        )
        return load_cys_network(
            network_filepath,
            self.config["network_source_node_label"],
            self.config["network_target_node_label"],
            self.config["network_edge_weight_label"],
            compute_sphere=network_enrichment_compute_sphere,
            dimple_factor=network_enrichment_dimple_factor,
            min_edges_per_node=network_min_edges_per_node,
            include_edge_weight=network_enrichment_include_edge_weight,
        )

    def tune_graph_and_get_neighborhoods(self, G):
        network_enrichment_compute_sphere = self.config["network_enrichment_compute_sphere"]
        network_enrichment_dimple_factor = self.config["network_enrichment_dimple_factor"]
        if network_enrichment_dimple_factor == "auto" and network_enrichment_compute_sphere:
            # Dimple factors go from 0 (sphere surface) to 1000 (sphere center) - find best dimple factor within range
            G_copy = self._get_network_with_best_dimples(
                G, lower_bound=0, upper_bound=1024, tolerance=4
            )
        else:
            G_copy = calculate_edge_lengths(
                G.copy(),
                compute_sphere=network_enrichment_compute_sphere,
                include_edge_weight=self.config["network_enrichment_include_edge_weight"],
                dimple_factor=network_enrichment_dimple_factor,
            )
        neighborhoods = self.load_neighborhoods(G_copy)
        return G_copy, neighborhoods

    def load_network_annotation(self, network):
        print("[cyan]Loading [yellow]JSON[/yellow] [blue]network annotations[/blue]...")
        annotation = load_network_annotation(network, self.config["annotation_filepath"])
        return annotation

    def tune_diameter_in_config(self, network):
        if self.config["neighborhood_diameter"] == "auto":
            optimal_diameter = self.find_optimal_diameter(network)
            print(f"[yellow]Optimal search diameter:[/yellow] [red]{optimal_diameter}[/red]")
            self.config["neighborhood_diameter"] = optimal_diameter

    def load_neighborhoods(self, network):
        neighborhood_distance_metric = self.config["neighborhood_distance_metric"]
        print(
            f"[cyan]Computing [blue]network neighborhoods[/blue] using [yellow]'{neighborhood_distance_metric}'[/yellow] as the metric..."
        )
        neighborhoods = get_network_neighborhoods(
            network,
            neighborhood_distance_metric,
            self.config["neighborhood_diameter"],
            compute_sphere=self.config["network_enrichment_compute_sphere"],
            louvain_resolution=self.config["neighborhood_distance_louvaine_resolution"],
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
            self.config["enrichment_pval_cutoff"],
            self.config["enrichment_apply_fdr"],
            self.config["enrichment_fdr_cutoff"],
            null_distribution=network_null_distribution,
            num_permutations=self.config["network_enrichment_num_permutations"],
            random_seed=888,
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
        print(f"[cyan]Optimizing [blue]distance threshold[/blue] for [blue]domains[/blue]...")
        domains_matrix = define_domains(
            neighborhood_enrichment_matrix,
            neighborhood_binary_enrichment_matrix_below_alpha,
            annotation_enrichment_matrix,
            self.config["group_distance_linkage"],
            self.config["group_distance_metric"],
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
        )

    def _get_network_with_best_dimples(self, G, lower_bound=0, upper_bound=500, tolerance=1):
        print(
            "[cyan][red]Warning:[/red] [blue]Optimizing[/blue] [yellow]dimple factor[/yellow] can be an [red]expensive process[/red]."
            "[blue] Mark down[/blue] [yellow]optimal dimple factor[/yellow] for future use...[/cyan]"
        )
        max_score = -np.inf
        best_dimple_factor = lower_bound
        best_graph = None

        # Calculate total iterations for progress display
        # Add one to account for 0 dimple factor testing
        total_iterations = int(np.ceil(np.log2((upper_bound - lower_bound) / tolerance))) + 1
        with Progress() as progress:
            task_id = progress.add_task(
                f"[cyan]Optimizing [yellow]dimple factor[/yellow]...[/cyan]",
                total=total_iterations,
            )
            current_iteration = 0

            while upper_bound - lower_bound > tolerance:
                if current_iteration == 0:
                    # Explicitly test dimple factor of 0
                    G_test_zero = calculate_edge_lengths(
                        G.copy(),
                        compute_sphere=self.config["network_enrichment_compute_sphere"],
                        include_edge_weight=self.config["network_enrichment_include_edge_weight"],
                        dimple_factor=0,
                    )
                    with suppress_print():
                        neighborhoods_test_zero = self.load_neighborhoods(G_test_zero)
                    score_test = ndimage.label(neighborhoods_test_zero)[1]
                    if score_test > max_score:
                        max_score = score_test
                        best_dimple_factor = 0
                        best_graph = G_test_zero
                else:
                    mid_dimple_factor = (lower_bound + upper_bound) / 2
                    dimple_factors_to_test = [mid_dimple_factor, mid_dimple_factor + tolerance]

                    # Always cast dimple_factors to type int
                    for dimple_factor in map(lambda x: int(x), dimple_factors_to_test):
                        G_test = calculate_edge_lengths(
                            G.copy(),
                            compute_sphere=self.config["network_enrichment_compute_sphere"],
                            include_edge_weight=self.config[
                                "network_enrichment_include_edge_weight"
                            ],
                            dimple_factor=dimple_factor,
                        )
                        with suppress_print():
                            neighborhoods_test = self.load_neighborhoods(G_test)
                        score_test = ndimage.label(neighborhoods_test)[1]

                        if score_test > max_score:
                            max_score = score_test
                            best_dimple_factor = dimple_factor
                            best_graph = G_test
                    # Adjust the search bounds
                    if best_dimple_factor == mid_dimple_factor + tolerance:
                        lower_bound = mid_dimple_factor
                    else:
                        upper_bound = mid_dimple_factor

                current_iteration += 1
                progress.update(
                    task_id,
                    advance=1,
                    description=f"[cyan]Optimizing [yellow]dimple factor[/yellow]...[/cyan]",
                )

        print(f"[yellow]Optimal dimple factor:[/yellow] [red]{best_dimple_factor}[/red]")
        return best_graph

    def compute_silhouette_for_diameter(self, network, diameter):
        neighborhoods_matrix = get_network_neighborhoods(
            network,
            self.config["neighborhood_distance_metric"],
            diameter,
            compute_sphere=self.config["network_enrichment_compute_sphere"],
            louvain_resolution=self.config["neighborhood_distance_louvaine_resolution"],
        )

        # Apply ndimage.label to get the connected components
        labels, num_features = label(neighborhoods_matrix)

        # modes.mode contains the most frequent values
        labels_1d = np.amax(labels, axis=1)

        # Initialize the distance matrix with the same shape as the neighborhoods matrix
        distance_matrix = np.zeros_like(neighborhoods_matrix, dtype=float)

        # Iterate over each row to compute the distance per row
        for i, row in enumerate(neighborhoods_matrix):
            max_value_row = np.max(row)  # Max value in this row
            distance_matrix[i] = (
                max_value_row - row
            )  # Convert similarities to distances for this row

        # Ensure diagonal values are 0 (distance to itself)
        np.fill_diagonal(distance_matrix, 0)

        # Ensure all values are non-negative
        distance_matrix = np.maximum(distance_matrix, 0)

        # The silhouette_score function expects a 1D distance matrix when metric is 'precomputed'
        score = abs(silhouette_score(distance_matrix, labels_1d, metric="cosine"))
        print(diameter, score)
        return score

    def find_optimal_diameter(self, network):
        lower_bound = 0.01
        upper_bound = 1.00
        tolerance = 0.01

        best_score = float("inf")
        best_diameter = lower_bound

        G_test = calculate_edge_lengths(
            network.copy(),
            compute_sphere=self.config["network_enrichment_compute_sphere"],
            include_edge_weight=self.config["network_enrichment_include_edge_weight"],
            dimple_factor=0,
        )

        # Evaluate scores at the explicit starting points
        score_at_lower_bound = self.compute_silhouette_for_diameter(G_test, lower_bound)
        score_at_upper_bound = self.compute_silhouette_for_diameter(G_test, upper_bound)

        # Initialize best scores and diameters based on initial evaluations
        if score_at_lower_bound < score_at_upper_bound:
            best_score = score_at_lower_bound
            best_diameter = lower_bound
        else:
            best_score = score_at_upper_bound
            best_diameter = upper_bound

        # Adjust initial bounds if lower_bound performs better
        if best_diameter == lower_bound:
            upper_bound = (lower_bound + upper_bound) / 2
        else:
            lower_bound = (lower_bound + upper_bound) / 2

        best_score = float("inf")
        best_diameter = lower_bound
        while upper_bound - lower_bound > tolerance:
            midpoint = (lower_bound + upper_bound) / 2
            midpoint_score = self.compute_silhouette_for_diameter(G_test, midpoint)

            if midpoint_score < best_score:
                best_score = midpoint_score
                best_diameter = midpoint

            # Adjust bounds based on comparison with the midpoint
            if midpoint_score < score_at_lower_bound or midpoint_score < score_at_upper_bound:
                if midpoint > best_diameter:
                    upper_bound = midpoint
                else:
                    lower_bound = midpoint
            else:
                if score_at_lower_bound < score_at_upper_bound:
                    upper_bound = midpoint
                else:
                    lower_bound = midpoint

        print(
            f"Optimal diameter: {best_diameter} with a silhouette score closest to 0: {best_score}"
        )
        return best_diameter


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


@contextmanager
def suppress_print():
    original_stdout = sys.stdout  # Save a reference to the original standard output
    try:
        with open(os.devnull, "w") as devnull:
            sys.stdout = devnull  # Redirect the standard output to the null device
            yield
    finally:
        sys.stdout = original_stdout  # Restore the standard output to its original value
