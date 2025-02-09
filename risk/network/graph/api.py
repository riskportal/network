"""
risk/network/graph/api
~~~~~~~~~~~~~~~~~~~~~~
"""

import copy
from typing import Any, Dict, Union

import networkx as nx
import pandas as pd

from risk.annotations import define_top_annotations
from risk.log import logger, log_header, params
from risk.neighborhoods import (
    define_domains,
    process_neighborhoods,
    trim_domains,
)
from risk.network.graph.graph import Graph
from risk.stats import calculate_significance_matrices


class GraphAPI:
    """Handles the loading of network graphs and associated data.

    The GraphAPI class provides methods to load and process network graphs, annotations, and neighborhoods.
    """

    def __init__() -> None:
        pass

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
        linkage_threshold: Union[float, str] = 0.2,
        min_cluster_size: int = 5,
        max_cluster_size: int = 1000,
    ) -> Graph:
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
            linkage_method (str, optional): Clustering method to use. Choose "auto" to optimize. Defaults to "average".
            linkage_metric (str, optional): Metric to use for calculating distances. Choose "auto" to optimize.
                Defaults to "yule".
            linkage_threshold (float, str, optional): Threshold for clustering. Choose "auto" to optimize.
                Defaults to 0.2.
            min_cluster_size (int, optional): Minimum size for clusters. Defaults to 5.
            max_cluster_size (int, optional): Maximum size for clusters. Defaults to 1000.

        Returns:
            Graph: A fully initialized and processed Graph object.
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
            linkage_threshold=linkage_threshold,
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
            linkage_threshold=linkage_threshold,
        )
        # Trim domains and top annotations based on cluster size constraints
        domains, trimmed_domains = trim_domains(
            domains=domains,
            top_annotations=top_annotations,
            min_cluster_size=min_cluster_size,
            max_cluster_size=max_cluster_size,
        )

        # Prepare node mapping and significance sums for the final Graph object
        ordered_nodes = annotations["ordered_nodes"]
        node_label_to_id = dict(zip(ordered_nodes, range(len(ordered_nodes))))
        node_significance_sums = processed_neighborhoods["node_significance_sums"]

        # Return the fully initialized Graph object
        return Graph(
            network=network,
            annotations=annotations,
            neighborhoods=neighborhoods,
            domains=domains,
            trimmed_domains=trimmed_domains,
            node_label_to_node_id_map=node_label_to_id,
            node_significance_sums=node_significance_sums,
        )

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
