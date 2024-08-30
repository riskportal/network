"""
risk/risk
~~~~~~~~~
"""

from typing import Any, Dict

import networkx as nx
import numpy as np
import pandas as pd

from risk.annotations import AnnotationsIO, define_top_annotations
from risk.log import params, print_header
from risk.neighborhoods import (
    define_domains,
    get_network_neighborhoods,
    process_neighborhoods,
    trim_domains_and_top_annotations,
)
from risk.network import NetworkIO, NetworkGraph, NetworkPlotter
from risk.stats import (
    calculate_significance_matrices,
    compute_fisher_exact_test,
    compute_hypergeom_test,
    compute_permutation_test,
)


class RISK(NetworkIO, AnnotationsIO):
    """RISK: A class for network analysis and visualization.

    The RISK class integrates functionalities for loading networks, processing annotations,
    and performing network-based statistical analysis, such as neighborhood significance testing.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the RISK class with configuration settings."""
        # Initialize and log network parameters
        params.initialize()
        # Initialize the parent classes
        super().__init__(*args, **kwargs)

    @property
    def params(self):
        """Access the logged parameters."""
        return params

    def load_neighborhoods_by_permutation(
        self,
        network: nx.Graph,
        annotations: Dict[str, Any],
        distance_metric: str = "dijkstra",
        louvain_resolution: float = 0.1,
        edge_length_threshold: float = 0.5,
        score_metric: str = "sum",
        null_distribution: str = "network",
        num_permutations: int = 1000,
        random_seed: int = 888,
        max_workers: int = 1,
    ) -> Dict[str, Any]:
        """Load significant neighborhoods for the network using the permutation test.

        Args:
            network (nx.Graph): The network graph.
            annotations (pd.DataFrame): The matrix of annotations associated with the network.
            distance_metric (str, optional): Distance metric for neighborhood analysis. Defaults to "dijkstra".
            louvain_resolution (float, optional): Resolution parameter for Louvain clustering. Defaults to 0.1.
            edge_length_threshold (float, optional): Edge length threshold for neighborhood analysis. Defaults to 0.5.
            score_metric (str, optional): Scoring metric for neighborhood significance. Defaults to "sum".
            null_distribution (str, optional): Distribution used for permutation tests. Defaults to "network".
            num_permutations (int, optional): Number of permutations for significance testing. Defaults to 1000.
            random_seed (int, optional): Seed for random number generation. Defaults to 888.
            max_workers (int, optional): Maximum number of workers for parallel computation. Defaults to 1.

        Returns:
            dict: Computed significance of neighborhoods.
        """
        print_header("Running permutation test")
        # Log neighborhood analysis parameters
        params.log_neighborhoods(
            distance_metric=distance_metric,
            louvain_resolution=louvain_resolution,
            edge_length_threshold=edge_length_threshold,
            statistical_test_function="permutation",
            score_metric=score_metric,
            null_distribution=null_distribution,
            num_permutations=num_permutations,
            random_seed=random_seed,
            max_workers=max_workers,
        )

        # Load neighborhoods based on the network and distance metric
        neighborhoods = self._load_neighborhoods(
            network,
            distance_metric,
            louvain_resolution=louvain_resolution,
            edge_length_threshold=edge_length_threshold,
            random_seed=random_seed,
        )

        # Log and display permutation test settings
        print(f"Neighborhood scoring metric: '{score_metric}'")
        print(f"Null distribution: '{null_distribution}'")
        print(f"Number of permutations: {num_permutations}")
        print(f"Maximum workers: {max_workers}")
        # Run permutation test to compute neighborhood significance
        neighborhood_significance = compute_permutation_test(
            neighborhoods=neighborhoods,
            annotations=annotations["matrix"],
            score_metric=score_metric,
            null_distribution=null_distribution,
            num_permutations=num_permutations,
            random_seed=random_seed,
            max_workers=max_workers,
        )

        return neighborhood_significance

    def load_neighborhoods_by_fisher_exact(
        self,
        network: nx.Graph,
        annotations: Dict[str, Any],
        distance_metric: str = "dijkstra",
        louvain_resolution: float = 0.1,
        edge_length_threshold: float = 0.5,
        random_seed: int = 888,
        max_workers: int = 1,
    ) -> Dict[str, Any]:
        """Load significant neighborhoods for the network using the Fisher's exact test.

        Args:
            network (nx.Graph): The network graph.
            annotations (pd.DataFrame): The matrix of annotations associated with the network.
            distance_metric (str, optional): Distance metric for neighborhood analysis. Defaults to "dijkstra".
            louvain_resolution (float, optional): Resolution parameter for Louvain clustering. Defaults to 0.1.
            edge_length_threshold (float, optional): Edge length threshold for neighborhood analysis. Defaults to 0.5.
            random_seed (int, optional): Seed for random number generation. Defaults to 888.
            max_workers (int, optional): Maximum number of workers for parallel computation. Defaults to 1.

        Returns:
            dict: Computed significance of neighborhoods.
        """
        print_header("Running Fisher's exact test")
        # Log neighborhood analysis parameters
        params.log_neighborhoods(
            distance_metric=distance_metric,
            louvain_resolution=louvain_resolution,
            edge_length_threshold=edge_length_threshold,
            statistical_test_function="fisher_exact",
            random_seed=random_seed,
            max_workers=max_workers,
        )

        # Load neighborhoods based on the network and distance metric
        neighborhoods = self._load_neighborhoods(
            network,
            distance_metric,
            louvain_resolution=louvain_resolution,
            edge_length_threshold=edge_length_threshold,
            random_seed=random_seed,
        )

        # Log and display Fisher's exact test settings
        print(f"Maximum workers: {max_workers}")
        # Run Fisher's exact test to compute neighborhood significance
        neighborhood_significance = compute_fisher_exact_test(
            neighborhoods=neighborhoods,
            annotations=annotations["matrix"],
            max_workers=max_workers,
        )

        return neighborhood_significance

    def load_neighborhoods_by_hypergeom(
        self,
        network: nx.Graph,
        annotations: Dict[str, Any],
        distance_metric: str = "dijkstra",
        louvain_resolution: float = 0.1,
        edge_length_threshold: float = 0.5,
        random_seed: int = 888,
        max_workers: int = 1,
    ) -> Dict[str, Any]:
        """Load significant neighborhoods for the network using the hypergeometric test.

        Args:
            network (nx.Graph): The network graph.
            annotations (pd.DataFrame): The matrix of annotations associated with the network.
            distance_metric (str, optional): Distance metric for neighborhood analysis. Defaults to "dijkstra".
            louvain_resolution (float, optional): Resolution parameter for Louvain clustering. Defaults to 0.1.
            edge_length_threshold (float, optional): Edge length threshold for neighborhood analysis. Defaults to 0.5.
            random_seed (int, optional): Seed for random number generation. Defaults to 888.
            max_workers (int, optional): Maximum number of workers for parallel computation. Defaults to 1.

        Returns:
            dict: Computed significance of neighborhoods.
        """
        print_header("Running hypergeometric test")
        # Log neighborhood analysis parameters
        params.log_neighborhoods(
            distance_metric=distance_metric,
            louvain_resolution=louvain_resolution,
            edge_length_threshold=edge_length_threshold,
            statistical_test_function="hypergeom",
            random_seed=random_seed,
            max_workers=max_workers,
        )

        # Load neighborhoods based on the network and distance metric
        neighborhoods = self._load_neighborhoods(
            network,
            distance_metric,
            louvain_resolution=louvain_resolution,
            edge_length_threshold=edge_length_threshold,
            random_seed=random_seed,
        )

        # Log and display hypergeometric test settings
        print(f"Maximum workers: {max_workers}")
        # Run hypergeometric test to compute neighborhood significance
        neighborhood_significance = compute_hypergeom_test(
            neighborhoods=neighborhoods,
            annotations=annotations["matrix"],
            max_workers=max_workers,
        )

        return neighborhood_significance

    def load_graph(
        self,
        network: nx.Graph,
        annotations: Dict[str, Any],
        neighborhoods: Dict[str, Any],
        tail: str = "right",  # OPTIONS: "right" (enrichment), "left" (depletion), "both"
        pval_cutoff: float = 0.01,  # OPTIONS: Any value between 0 to 1
        fdr_cutoff: float = 0.9999,  # OPTIONS: Any value between 0 to 1
        impute_depth: int = 1,
        prune_threshold: float = 0.0,
        linkage_criterion: str = "distance",
        linkage_method: str = "average",
        linkage_metric: str = "yule",
        min_cluster_size: int = 5,
        max_cluster_size: int = 1000,
    ) -> NetworkGraph:
        """Load and process the network graph, defining top annotations and domains.

        Args:
            network (nx.Graph): The network graph.
            annotations (pd.DataFrame): DataFrame containing annotation data for the network.
            neighborhoods (dict): Neighborhood enrichment data.
            tail (str, optional): Type of significance tail ("right", "left", "both"). Defaults to "right".
            pval_cutoff (float, optional): p-value cutoff for significance. Defaults to 0.01.
            fdr_cutoff (float, optional): FDR cutoff for significance. Defaults to 0.9999.
            impute_depth (int, optional): Depth for imputing neighbors. Defaults to 1.
            prune_threshold (float, optional): Distance threshold for pruning neighbors. Defaults to 0.0.
            linkage_criterion (str, optional): Clustering criterion for defining domains. Defaults to "distance".
            linkage_method (str, optional): Clustering method to use. Defaults to "average".
            linkage_metric (str, optional): Metric to use for calculating distances. Defaults to "yule".
            min_cluster_size (int, optional): Minimum size for clusters. Defaults to 5.
            max_cluster_size (int, optional): Maximum size for clusters. Defaults to 1000.

        Returns:
            NetworkGraph: A fully initialized and processed NetworkGraph object.
        """
        # Log the parameters and display headers
        print_header("Finding significant neighborhoods")
        params.log_graph(
            tail=tail,
            pval_cutoff=pval_cutoff,
            fdr_cutoff=fdr_cutoff,
            impute_depth=impute_depth,
            prune_threshold=prune_threshold,
            linkage_criterion=linkage_criterion,
            linkage_method=linkage_method,
            linkage_metric=linkage_metric,
            min_cluster_size=min_cluster_size,
            max_cluster_size=max_cluster_size,
        )

        print(f"p-value cutoff: {pval_cutoff}")
        print(f"FDR BH cutoff: {fdr_cutoff}")
        print(
            f"Significance tail: '{tail}' ({'enrichment' if tail == 'right' else 'depletion' if tail == 'left' else 'both'})"
        )
        # Calculate significant neighborhoods based on the provided parameters
        significant_neighborhoods = calculate_significance_matrices(
            neighborhoods["depletion_pvals"],
            neighborhoods["enrichment_pvals"],
            tail=tail,
            pval_cutoff=pval_cutoff,
            fdr_cutoff=fdr_cutoff,
        )

        print_header("Processing neighborhoods")
        # Process neighborhoods by imputing and pruning based on the given settings
        processed_neighborhoods = process_neighborhoods(
            network=network,
            neighborhoods=significant_neighborhoods,
            impute_depth=impute_depth,
            prune_threshold=prune_threshold,
        )

        print_header("Finding top annotations")
        print(f"Min cluster size: {min_cluster_size}")
        print(f"Max cluster size: {max_cluster_size}")
        # Define top annotations based on processed neighborhoods
        top_annotations = self._define_top_annotations(
            network=network,
            annotations=annotations,
            neighborhoods=processed_neighborhoods,
            min_cluster_size=min_cluster_size,
            max_cluster_size=max_cluster_size,
        )

        print_header(f"Optimizing distance threshold for domains")
        # Define domains in the network using the specified clustering settings
        domains = self._define_domains(
            neighborhoods=processed_neighborhoods,
            top_annotations=top_annotations,
            linkage_criterion=linkage_criterion,
            linkage_method=linkage_method,
            linkage_metric=linkage_metric,
        )
        # Trim domains and top annotations based on cluster size constraints
        top_annotations, domains, trimmed_domains = trim_domains_and_top_annotations(
            domains=domains,
            top_annotations=top_annotations,
            min_cluster_size=min_cluster_size,
            max_cluster_size=max_cluster_size,
        )

        # Prepare node mapping and enrichment sums for the final NetworkGraph object
        ordered_nodes = annotations["ordered_nodes"]
        node_label_to_id = dict(zip(ordered_nodes, range(len(ordered_nodes))))
        node_enrichment_sums = processed_neighborhoods["node_enrichment_sums"]

        # Return the fully initialized NetworkGraph object
        return NetworkGraph(
            network=network,
            top_annotations=top_annotations,
            domains=domains,
            trimmed_domains=trimmed_domains,
            node_label_to_id_map=node_label_to_id,
            node_enrichment_sums=node_enrichment_sums,
        )

    def load_plotter(
        self,
        graph: NetworkGraph,
        figsize: tuple = (10, 10),
        background_color: str = "white",
        plot_outline: bool = True,
        outline_color: str = "black",
        outline_scale: float = 1.00,
    ) -> NetworkPlotter:
        """Get a NetworkPlotter object for plotting.

        Args:
            graph (NetworkGraph): The graph to plot.
            figsize (tuple, optional): Size of the figure. Defaults to (10, 10).
            background_color (str, optional): Background color of the plot. Defaults to "white".
            plot_outline (bool, optional): Whether to plot the network outline. Defaults to True.
            outline_color (str, optional): Color of the outline. Defaults to "black".
            outline_scale (float, optional): Scaling factor for the outline. Defaults to 1.00.

        Returns:
            NetworkPlotter: A NetworkPlotter object configured with the given parameters.
        """
        print_header("Loading plotter")
        # Log the plotter settings
        params.log_plotter(
            figsize=figsize,
            background_color=background_color,
            plot_outline=plot_outline,
            outline_color=outline_color,
            outline_scale=outline_scale,
        )

        # Initialize and return a NetworkPlotter object
        return NetworkPlotter(
            graph,
            figsize=figsize,
            background_color=background_color,
            plot_outline=plot_outline,
            outline_color=outline_color,
            outline_scale=outline_scale,
        )

    def _load_neighborhoods(
        self,
        network: nx.Graph,
        distance_metric: str = "dijkstra",
        louvain_resolution: float = 0.1,
        edge_length_threshold: float = 0.5,
        random_seed: int = 888,
    ) -> np.ndarray:
        """Load significant neighborhoods for the network.

        Args:
            network (nx.Graph): The network graph.
            annotations (pd.DataFrame): The matrix of annotations associated with the network.
            distance_metric (str, optional): Distance metric for neighborhood analysis. Defaults to "dijkstra".
            louvain_resolution (float, optional): Resolution parameter for Louvain clustering. Defaults to 0.1.
            edge_length_threshold (float, optional): Edge length threshold for neighborhood analysis. Defaults to 0.5.
            random_seed (int, optional): Seed for random number generation. Defaults to 888.

        Returns:
            np.ndarray: Neighborhood matrix calculated based on the selected distance metric.
        """
        # Display the chosen distance metric
        if distance_metric == "louvain":
            for_print_distance_metric = f"louvain (resolution={louvain_resolution})"
        else:
            for_print_distance_metric = distance_metric
        # Log and display neighborhood settings
        print(f"Distance metric: '{for_print_distance_metric}'")
        print(f"Edge length threshold: {edge_length_threshold}")
        print(f"Random seed: {random_seed}")

        # Compute neighborhoods based on the network and distance metric
        neighborhoods = get_network_neighborhoods(
            network,
            distance_metric,
            edge_length_threshold,
            louvain_resolution=louvain_resolution,
            random_seed=random_seed,
        )

        return neighborhoods

    def _define_top_annotations(
        self,
        network: nx.Graph,
        annotations: Dict[str, Any],
        neighborhoods: Dict[str, Any],
        min_cluster_size: int = 5,
        max_cluster_size: int = 1000,
    ) -> pd.DataFrame:
        """Define top annotations for the network.

        Args:
            network (nx.Graph): The network graph.
            annotations (dict): Annotations data for the network.
            neighborhoods (dict): Neighborhood enrichment data.
            min_cluster_size (int, optional): Minimum size for clusters. Defaults to 5.
            max_cluster_size (int, optional): Maximum size for clusters. Defaults to 1000.

        Returns:
            dict: Top annotations identified within the network.
        """
        # Extract necessary data from annotations and neighborhoods
        ordered_annotations = annotations["ordered_annotations"]
        neighborhood_enrichment_sums = neighborhoods["neighborhood_enrichment_counts"]
        neighborhoods_binary_enrichment_matrix = neighborhoods["binary_enrichment_matrix"]
        # Call external function to define top annotations
        return define_top_annotations(
            network=network,
            ordered_annotation_labels=ordered_annotations,
            neighborhood_enrichment_sums=neighborhood_enrichment_sums,
            binary_enrichment_matrix=neighborhoods_binary_enrichment_matrix,
            min_cluster_size=min_cluster_size,
            max_cluster_size=max_cluster_size,
        )

    def _define_domains(
        self,
        neighborhoods: Dict[str, Any],
        top_annotations: pd.DataFrame,
        linkage_criterion: str,
        linkage_method: str,
        linkage_metric: str,
    ) -> pd.DataFrame:
        """Define domains in the network based on enrichment data.

        Args:
            neighborhoods (dict): Enrichment data for neighborhoods.
            top_annotations (pd.DataFrame): Enrichment matrix for top annotations.
            linkage_criterion (str): Clustering criterion for defining domains.
            linkage_method (str): Clustering method to use.
            linkage_metric (str): Metric to use for calculating distances.

        Returns:
            pd.DataFrame: Matrix of defined domains.
        """
        # Extract the significant enrichment matrix from the neighborhoods data
        significant_neighborhoods_enrichment = neighborhoods["significant_enrichment_matrix"]
        # Call external function to define domains based on the extracted data
        return define_domains(
            top_annotations=top_annotations,
            significant_neighborhoods_enrichment=significant_neighborhoods_enrichment,
            linkage_criterion=linkage_criterion,
            linkage_method=linkage_method,
            linkage_metric=linkage_metric,
        )
