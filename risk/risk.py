"""
risk/risk
~~~~~~~~~
"""

import copy
from typing import Any, Dict, List, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd

from risk.annotations import AnnotationsIO, define_top_annotations
from risk.log import params, logger, log_header, set_global_verbosity
from risk.neighborhoods import (
    define_domains,
    get_network_neighborhoods,
    process_neighborhoods,
    trim_domains,
)
from risk.network import NetworkIO, NetworkGraph, NetworkPlotter
from risk.stats import (
    calculate_significance_matrices,
    compute_hypergeom_test,
    compute_permutation_test,
    compute_poisson_test,
)


class RISK(NetworkIO, AnnotationsIO):
    """RISK: A class for network analysis and visualization.

    The RISK class integrates functionalities for loading networks, processing annotations,
    and performing network-based statistical analysis, such as neighborhood significance testing.
    """

    def __init__(self, verbose: bool = True):
        """Initialize the RISK class with configuration settings.

        Args:
            verbose (bool): If False, suppresses all log messages to the console. Defaults to True.
        """
        # Set global verbosity for logging
        set_global_verbosity(verbose)
        # Provide public access to the logged network parameters
        self.params = params
        super().__init__()

    def load_neighborhoods_by_hypergeom(
        self,
        network: nx.Graph,
        annotations: Dict[str, Any],
        distance_metric: Union[str, List, Tuple, np.ndarray] = "louvain",
        louvain_resolution: float = 0.1,
        edge_length_threshold: Union[float, List, Tuple, np.ndarray] = 0.5,
        null_distribution: str = "network",
        random_seed: int = 888,
    ) -> Dict[str, Any]:
        """Load significant neighborhoods for the network using the hypergeometric test.

        Args:
            network (nx.Graph): The network graph.
            annotations (Dict[str, Any]): The annotations associated with the network.
            distance_metric (str, List, Tuple, or np.ndarray, optional): The distance metric(s) to use. Can be a string for one
                metric or a list/tuple/ndarray of metrics ('greedy_modularity', 'louvain', 'label_propagation',
                'markov_clustering', 'walktrap', 'spinglass'). Defaults to 'louvain'.
            louvain_resolution (float, optional): Resolution parameter for Louvain clustering. Defaults to 0.1.
            edge_length_threshold (float, List, Tuple, or np.ndarray, optional): Edge length threshold(s) for creating subgraphs.
                Can be a single float for one threshold or a list/tuple of floats corresponding to multiple thresholds.
                Defaults to 0.5.
            null_distribution (str, optional): Type of null distribution ('network' or 'annotations'). Defaults to "network".
            random_seed (int, optional): Seed for random number generation. Defaults to 888.

        Returns:
            Dict[str, Any]: Computed significance of neighborhoods.
        """
        log_header("Running hypergeometric test")
        # Log neighborhood analysis parameters
        params.log_neighborhoods(
            distance_metric=distance_metric,
            louvain_resolution=louvain_resolution,
            edge_length_threshold=edge_length_threshold,
            statistical_test_function="hypergeom",
            null_distribution=null_distribution,
            random_seed=random_seed,
        )

        # Make a copy of the network to avoid modifying the original
        network = copy.deepcopy(network)

        # Load neighborhoods based on the network and distance metric
        neighborhoods = self._load_neighborhoods(
            network,
            distance_metric,
            louvain_resolution=louvain_resolution,
            edge_length_threshold=edge_length_threshold,
            random_seed=random_seed,
        )
        # Run hypergeometric test to compute neighborhood significance
        neighborhood_significance = compute_hypergeom_test(
            neighborhoods=neighborhoods,
            annotations=annotations["matrix"],
            null_distribution=null_distribution,
        )

        # Return the computed neighborhood significance
        return neighborhood_significance

    def load_neighborhoods_by_poisson(
        self,
        network: nx.Graph,
        annotations: Dict[str, Any],
        distance_metric: Union[str, List, Tuple, np.ndarray] = "louvain",
        louvain_resolution: float = 0.1,
        edge_length_threshold: Union[float, List, Tuple, np.ndarray] = 0.5,
        null_distribution: str = "network",
        random_seed: int = 888,
    ) -> Dict[str, Any]:
        """Load significant neighborhoods for the network using the Poisson test.

        Args:
            network (nx.Graph): The network graph.
            annotations (Dict[str, Any]): The annotations associated with the network.
            distance_metric (str, List, Tuple, or np.ndarray, optional): The distance metric(s) to use. Can be a string for one
                metric or a list/tuple/ndarray of metrics ('greedy_modularity', 'louvain', 'label_propagation',
                'markov_clustering', 'walktrap', 'spinglass'). Defaults to 'louvain'.
            louvain_resolution (float, optional): Resolution parameter for Louvain clustering. Defaults to 0.1.
            edge_length_threshold (float, List, Tuple, or np.ndarray, optional): Edge length threshold(s) for creating subgraphs.
                Can be a single float for one threshold or a list/tuple of floats corresponding to multiple thresholds.
                Defaults to 0.5.
            null_distribution (str, optional): Type of null distribution ('network' or 'annotations'). Defaults to "network".
            random_seed (int, optional): Seed for random number generation. Defaults to 888.

        Returns:
            Dict[str, Any]: Computed significance of neighborhoods.
        """
        log_header("Running Poisson test")
        # Log neighborhood analysis parameters
        params.log_neighborhoods(
            distance_metric=distance_metric,
            louvain_resolution=louvain_resolution,
            edge_length_threshold=edge_length_threshold,
            statistical_test_function="poisson",
            null_distribution=null_distribution,
            random_seed=random_seed,
        )

        # Make a copy of the network to avoid modifying the original
        network = copy.deepcopy(network)

        # Load neighborhoods based on the network and distance metric
        neighborhoods = self._load_neighborhoods(
            network,
            distance_metric,
            louvain_resolution=louvain_resolution,
            edge_length_threshold=edge_length_threshold,
            random_seed=random_seed,
        )
        # Run Poisson test to compute neighborhood significance
        neighborhood_significance = compute_poisson_test(
            neighborhoods=neighborhoods,
            annotations=annotations["matrix"],
            null_distribution=null_distribution,
        )

        # Return the computed neighborhood significance
        return neighborhood_significance

    def load_neighborhoods_by_permutation(
        self,
        network: nx.Graph,
        annotations: Dict[str, Any],
        distance_metric: Union[str, List, Tuple, np.ndarray] = "louvain",
        louvain_resolution: float = 0.1,
        edge_length_threshold: Union[float, List, Tuple, np.ndarray] = 0.5,
        score_metric: str = "sum",
        null_distribution: str = "network",
        num_permutations: int = 1000,
        random_seed: int = 888,
        max_workers: int = 1,
    ) -> Dict[str, Any]:
        """Load significant neighborhoods for the network using the permutation test.

        Args:
            network (nx.Graph): The network graph.
            annotations (Dict[str, Any]): The annotations associated with the network.
            distance_metric (str, List, Tuple, or np.ndarray, optional): The distance metric(s) to use. Can be a string for one
                metric or a list/tuple/ndarray of metrics ('greedy_modularity', 'louvain', 'label_propagation',
                'markov_clustering', 'walktrap', 'spinglass'). Defaults to 'louvain'.
            louvain_resolution (float, optional): Resolution parameter for Louvain clustering. Defaults to 0.1.
            edge_length_threshold (float, List, Tuple, or np.ndarray, optional): Edge length threshold(s) for creating subgraphs.
                Can be a single float for one threshold or a list/tuple of floats corresponding to multiple thresholds.
                Defaults to 0.5.
            score_metric (str, optional): Scoring metric for neighborhood significance. Defaults to "sum".
            null_distribution (str, optional): Type of null distribution ('network' or 'annotations'). Defaults to "network".
            num_permutations (int, optional): Number of permutations for significance testing. Defaults to 1000.
            random_seed (int, optional): Seed for random number generation. Defaults to 888.
            max_workers (int, optional): Maximum number of workers for parallel computation. Defaults to 1.

        Returns:
            Dict[str, Any]: Computed significance of neighborhoods.
        """
        log_header("Running permutation test")
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

        # Make a copy of the network to avoid modifying the original
        network = copy.deepcopy(network)

        # Load neighborhoods based on the network and distance metric
        neighborhoods = self._load_neighborhoods(
            network,
            distance_metric,
            louvain_resolution=louvain_resolution,
            edge_length_threshold=edge_length_threshold,
            random_seed=random_seed,
        )

        # Log and display permutation test settings
        logger.debug(f"Neighborhood scoring metric: '{score_metric}'")
        logger.debug(f"Null distribution: '{null_distribution}'")
        logger.debug(f"Number of permutations: {num_permutations}")
        logger.debug(f"Maximum workers: {max_workers}")
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

        # Return the computed neighborhood significance
        return neighborhood_significance

    def load_graph(
        self,
        network: nx.Graph,
        annotations: Dict[str, Any],
        neighborhoods: Dict[str, Any],
        tail: str = "right",
        pval_cutoff: float = 0.01,
        fdr_cutoff: float = 0.9999,
        impute_depth: int = 0,
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
            annotations (Dict[str, Any]): The annotations associated with the network.
            neighborhoods (Dict[str, Any]): Neighborhood significance data.
            tail (str, optional): Type of significance tail ("right", "left", "both"). Defaults to "right".
            pval_cutoff (float, optional): p-value cutoff for significance. Defaults to 0.01.
            fdr_cutoff (float, optional): FDR cutoff for significance. Defaults to 0.9999.
            impute_depth (int, optional): Depth for imputing neighbors. Defaults to 0.
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
        log_header("Finding significant neighborhoods")
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

        # Make a copy of the network to avoid modifying the original
        network = copy.deepcopy(network)

        logger.debug(f"p-value cutoff: {pval_cutoff}")
        logger.debug(f"FDR BH cutoff: {fdr_cutoff}")
        logger.debug(
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

        log_header("Processing neighborhoods")
        # Process neighborhoods by imputing and pruning based on the given settings
        processed_neighborhoods = process_neighborhoods(
            network=network,
            neighborhoods=significant_neighborhoods,
            impute_depth=impute_depth,
            prune_threshold=prune_threshold,
        )

        log_header("Finding top annotations")
        logger.debug(f"Min cluster size: {min_cluster_size}")
        logger.debug(f"Max cluster size: {max_cluster_size}")
        # Define top annotations based on processed neighborhoods
        top_annotations = self._define_top_annotations(
            network=network,
            annotations=annotations,
            neighborhoods=processed_neighborhoods,
            min_cluster_size=min_cluster_size,
            max_cluster_size=max_cluster_size,
        )

        log_header("Optimizing distance threshold for domains")
        # Extract the significant significance matrix from the neighborhoods data
        significant_neighborhoods_significance = processed_neighborhoods[
            "significant_significance_matrix"
        ]
        # Define domains in the network using the specified clustering settings
        domains = define_domains(
            top_annotations=top_annotations,
            significant_neighborhoods_significance=significant_neighborhoods_significance,
            linkage_criterion=linkage_criterion,
            linkage_method=linkage_method,
            linkage_metric=linkage_metric,
        )
        # Trim domains and top annotations based on cluster size constraints
        domains, trimmed_domains = trim_domains(
            domains=domains,
            top_annotations=top_annotations,
            min_cluster_size=min_cluster_size,
            max_cluster_size=max_cluster_size,
        )

        # Prepare node mapping and significance sums for the final NetworkGraph object
        ordered_nodes = annotations["ordered_nodes"]
        node_label_to_id = dict(zip(ordered_nodes, range(len(ordered_nodes))))
        node_significance_sums = processed_neighborhoods["node_significance_sums"]

        # Return the fully initialized NetworkGraph object
        return NetworkGraph(
            network=network,
            annotations=annotations,
            neighborhoods=neighborhoods,
            domains=domains,
            trimmed_domains=trimmed_domains,
            node_label_to_node_id_map=node_label_to_id,
            node_significance_sums=node_significance_sums,
        )

    def load_plotter(
        self,
        graph: NetworkGraph,
        figsize: Union[List, Tuple, np.ndarray] = (10, 10),
        background_color: str = "white",
        background_alpha: Union[float, None] = 1.0,
        pad: float = 0.3,
    ) -> NetworkPlotter:
        """Get a NetworkPlotter object for plotting.

        Args:
            graph (NetworkGraph): The graph to plot.
            figsize (List, Tuple, or np.ndarray, optional): Size of the plot. Defaults to (10, 10)., optional): Size of the figure. Defaults to (10, 10).
            background_color (str, optional): Background color of the plot. Defaults to "white".
            background_alpha (float, None, optional): Transparency level of the background color. If provided, it overrides
                any existing alpha values found in background_color. Defaults to 1.0.
            pad (float, optional): Padding value to adjust the axis limits. Defaults to 0.3.

        Returns:
            NetworkPlotter: A NetworkPlotter object configured with the given parameters.
        """
        log_header("Loading plotter")

        # Initialize and return a NetworkPlotter object
        return NetworkPlotter(
            graph,
            figsize=figsize,
            background_color=background_color,
            background_alpha=background_alpha,
            pad=pad,
        )

    def _load_neighborhoods(
        self,
        network: nx.Graph,
        distance_metric: Union[str, List, Tuple, np.ndarray] = "louvain",
        louvain_resolution: float = 0.1,
        edge_length_threshold: Union[float, List, Tuple, np.ndarray] = 0.5,
        random_seed: int = 888,
    ) -> np.ndarray:
        """Load significant neighborhoods for the network.

        Args:
            network (nx.Graph): The network graph.
            annotations (pd.DataFrame): The matrix of annotations associated with the network.
            distance_metric (str, List, Tuple, or np.ndarray, optional): The distance metric(s) to use. Can be a string for one
                metric or a list/tuple/ndarray of metrics ('greedy_modularity', 'louvain', 'label_propagation',
                'markov_clustering', 'walktrap', 'spinglass'). Defaults to 'louvain'.
            louvain_resolution (float, optional): Resolution parameter for Louvain clustering. Defaults to 0.1.
            edge_length_threshold (float, List, Tuple, or np.ndarray, optional): Edge length threshold(s) for creating subgraphs.
                Can be a single float for one threshold or a list/tuple of floats corresponding to multiple thresholds.
                Defaults to 0.5.
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
        logger.debug(f"Distance metric: '{for_print_distance_metric}'")
        logger.debug(f"Edge length threshold: {edge_length_threshold}")
        logger.debug(f"Random seed: {random_seed}")

        # Compute neighborhoods based on the network and distance metric
        neighborhoods = get_network_neighborhoods(
            network,
            distance_metric,
            edge_length_threshold,
            louvain_resolution=louvain_resolution,
            random_seed=random_seed,
        )

        # Return the computed neighborhoods
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
            annotations (Dict[str, Any]): Annotations data for the network.
            neighborhoods (Dict[str, Any]): Neighborhood significance data.
            min_cluster_size (int, optional): Minimum size for clusters. Defaults to 5.
            max_cluster_size (int, optional): Maximum size for clusters. Defaults to 1000.

        Returns:
            Dict[str, Any]: Top annotations identified within the network.
        """
        # Extract necessary data from annotations and neighborhoods
        ordered_annotations = annotations["ordered_annotations"]
        neighborhood_significance_sums = neighborhoods["neighborhood_significance_counts"]
        significant_significance_matrix = neighborhoods["significant_significance_matrix"]
        significant_binary_significance_matrix = neighborhoods[
            "significant_binary_significance_matrix"
        ]
        # Call external function to define top annotations
        return define_top_annotations(
            network=network,
            ordered_annotation_labels=ordered_annotations,
            neighborhood_significance_sums=neighborhood_significance_sums,
            significant_significance_matrix=significant_significance_matrix,
            significant_binary_significance_matrix=significant_binary_significance_matrix,
            min_cluster_size=min_cluster_size,
            max_cluster_size=max_cluster_size,
        )
