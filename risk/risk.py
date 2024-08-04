from risk.annotations import define_top_annotations
from risk.graph import NetworkGraph
from risk.io import AnnotationsIO, NetworkIO
from risk.log import params, print_header
from risk.neighborhoods import (
    define_domains,
    get_network_neighborhoods,
    process_neighborhoods,
    trim_domains_and_top_annotations,
)
from risk.plot import NetworkPlotter
from risk.stats import calculate_significance_matrices, compute_permutation


class RISK(NetworkIO, AnnotationsIO):
    """RISK: A class for network analysis and visualization."""

    def __init__(
        self,
        compute_sphere=True,
        surface_depth="auto",
        neighborhood_diameter=0.5,
        distance_metric="dijkstra",
        louvain_resolution=0.1,
        random_walk_length=3,
        random_walk_num=250,
        include_edge_weight=True,
        min_edges_per_node=0,
    ):
        """Initialize RISK with configuration settings.

        Args:
            network_filepath (str): Path to the network file.
            annotation_filepath (str): Path to the annotations file.
            **kwargs: Additional configuration parameters.
        """
        # Always clear / initialize all logged parameters upon RISK instantiation
        params.initialize()
        params.log_network(
            compute_sphere=compute_sphere,
            surface_depth=surface_depth,
            include_edge_weight=include_edge_weight,
            distance_metric=distance_metric,
            neighborhood_diameter=neighborhood_diameter,
            louvain_resolution=louvain_resolution,
            random_walk_length=random_walk_length,
            random_walk_num=random_walk_num,
            min_edges_per_node=min_edges_per_node,
        )
        NetworkIO.__init__(
            self,
            compute_sphere=compute_sphere,
            surface_depth=surface_depth,
            include_edge_weight=include_edge_weight,
            distance_metric=distance_metric,
            neighborhood_diameter=neighborhood_diameter,
            louvain_resolution=louvain_resolution,
            random_walk_length=random_walk_length,
            random_walk_num=random_walk_num,
            min_edges_per_node=min_edges_per_node,
        )
        AnnotationsIO.__init__(self)
        self.compute_sphere = compute_sphere
        self.surface_depth = surface_depth
        self.include_edge_weight = include_edge_weight
        self.distance_metric = distance_metric
        self.neighborhood_diameter = neighborhood_diameter
        self.louvain_resolution = louvain_resolution
        self.random_walk_length = random_walk_length
        self.random_walk_num = random_walk_num
        self.min_edges_per_node = min_edges_per_node

    @property
    def params(self):
        return params

    def load_neighborhoods(
        self,
        network,
        annotations,
        score_metric="sum",
        null_distribution="network",
        num_permutations=1000,
        random_seed=888,
        max_workers=1,
    ):
        """Load significant neighborhoods for the network.

        Args:
            network (NetworkX graph): The network graph.

        Returns:
            dict: Neighborhoods of the network.
        """
        print_header("Running permutation test")
        params.log_neighborhoods(
            score_metric=score_metric,
            null_distribution=null_distribution,
            num_permutations=num_permutations,
            random_seed=random_seed,
            max_workers=max_workers,
        )
        for_print_distance_metric = (
            f"louvain (resolution={self.louvain_resolution})"
            if self.distance_metric == "louvain"
            else f"random walk (length={self.random_walk_length}, num={self.random_walk_num})"
            if self.distance_metric == "random_walk"
            else self.distance_metric
        )
        print(f"Distance metric: '{for_print_distance_metric}'")
        neighborhoods = get_network_neighborhoods(
            network,
            self.distance_metric,
            self.neighborhood_diameter,
            compute_sphere=self.compute_sphere,
            louvain_resolution=self.louvain_resolution,
            random_walk_length=self.random_walk_length,
            random_walk_num=self.random_walk_num,
            random_seed=random_seed,
        )
        print(f"Null distribution: '{null_distribution}'")
        print(f"Neighborhood scoring metric: '{score_metric}'")
        neighborhood_significance = compute_permutation(
            neighborhoods=neighborhoods,
            annotations=annotations["matrix"],
            score_metric=score_metric,
            null_distribution=null_distribution,
            num_permutations=num_permutations,
            random_seed=random_seed,
            max_workers=max_workers,
        )
        return neighborhood_significance

    def load_graph(
        self,
        network,
        annotations,
        neighborhoods,
        tail="right",  # OPTIONS: right (enrichment), left (depletion), both
        pval_cutoff=0.01,  # OPTIONS: Any value between 0 to 1
        apply_fdr=False,
        fdr_cutoff=0.9999,  # OPTIONS: Any value between 0 to 1
        impute_depth=1,
        prune_threshold=0.0,
        linkage_criterion="distance",
        linkage_method="average",
        linkage_metric="yule",
        min_cluster_size=5,
        max_cluster_size=1000,
    ):
        """Get a NetworkGraph object for plotting.

        Args:
            network (NetworkX graph): The network graph.
            annotations (pd.DataFrame): Annotations matrix.
            neighborhoods (dict): Neighborhood enrichment data.

        Returns:
            NetworkGraph: A NetworkGraph object.
        """
        print_header("Finding significant neighborhoods")
        params.log_graph(
            tail=tail,
            pval_cutoff=pval_cutoff,
            apply_fdr=apply_fdr,
            fdr_cutoff=fdr_cutoff,
            impute_depth=impute_depth,
            prune_threshold=prune_threshold,
            linkage_criterion=linkage_criterion,
            linkage_method=linkage_method,
            linkage_metric=linkage_metric,
            min_cluster_size=min_cluster_size,
            max_cluster_size=max_cluster_size,
        )
        print(f"P-value cutoff: {pval_cutoff}")
        print(f"FDR cutoff: {'N/A' if not apply_fdr else apply_fdr}")
        print(
            f"Significance tail: '{tail}' ({'enrichment' if tail == 'right' else 'depletion' if tail == 'left' else 'both'})"
        )
        significant_neighborhoods = calculate_significance_matrices(
            neighborhoods["depletion_pvals"],
            neighborhoods["enrichment_pvals"],
            tail=tail,
            pval_cutoff=pval_cutoff,
            apply_fdr=apply_fdr,
            fdr_cutoff=fdr_cutoff,
        )
        # Process neighborhoods based on the imputation and pruning settings
        processed_neighborhoods = self._process_neighborhoods(
            network,
            significant_neighborhoods,
            impute_depth=impute_depth,
            prune_threshold=prune_threshold,
        )
        print_header("Trimming domains and finding top annotations")
        print(f"Min cluster size: {min_cluster_size}")
        print(f"Max cluster size: {max_cluster_size}")
        # Define top annotations based on the network and neighborhoods
        top_annotations = self._define_top_annotations(
            network=network,
            annotations=annotations,
            neighborhoods=processed_neighborhoods,
            min_cluster_size=min_cluster_size,
            max_cluster_size=max_cluster_size,
        )
        print_header(f"Optimizing distance threshold for domains")
        # Define domains in the network
        domains = self._define_domains(
            neighborhoods=processed_neighborhoods,
            top_annotations=top_annotations,
            linkage_criterion=linkage_criterion,
            linkage_method=linkage_method,
            linkage_metric=linkage_metric,
        )
        # Trim domains and top annotations based on the cluster size
        top_annotations, domains, trimmed_domains = trim_domains_and_top_annotations(
            domains=domains,
            top_annotations=top_annotations,
            min_cluster_size=min_cluster_size,
            max_cluster_size=max_cluster_size,
        )
        # Get the ordered nodes for the network
        ordered_nodes = annotations["ordered_nodes"]
        node_label_to_id = dict(zip(ordered_nodes, range(len(ordered_nodes))))
        # Get the significant binary enrichment matrix for neighborhoods
        node_enrichment_sums = processed_neighborhoods["node_enrichment_sums"]
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
        graph,
        figsize=(10, 10),
        background_color="white",
        plot_outline=True,
        outline_color="black",
        outline_scale=1.00,
    ):
        """Get a NetworkPlotter object for plotting.

        Args:
            graph (NetworkGraph): A NetworkGraph object.

        Returns:
            NetworkPlotter: A NetworkPlotter object.
        """
        print_header("Loading plotter")
        params.log_plotter(
            figsize=figsize,
            background_color=background_color,
            plot_outline=plot_outline,
            outline_color=outline_color,
            outline_scale=outline_scale,
        )
        return NetworkPlotter(
            graph,
            figsize=figsize,
            background_color=background_color,
            plot_outline=plot_outline,
            outline_color=outline_color,
            outline_scale=outline_scale,
        )

    def _process_neighborhoods(
        self,
        network,
        neighborhoods,
        impute_depth=1,
        prune_threshold=0.0,
    ):
        """Process neighborhoods based on the imputation and pruning settings.

        Args:
            neighborhoods (dict): Neighborhoods data.
            impute_depth (int): Depth for imputing neighbors.
            prune_threshold (float): Distance threshold for pruning neighbors.

        Returns:
            dict: Adjusted neighborhoods data.
        """
        enrichment_matrix = neighborhoods["significance_matrix"]
        binary_enrichment_matrix = neighborhoods["binary_significance_matrix"]
        return process_neighborhoods(
            network=network,
            enrichment_matrix=enrichment_matrix,
            binary_enrichment_matrix=binary_enrichment_matrix,
            impute_depth=impute_depth,
            prune_threshold=prune_threshold,
        )

    def _define_top_annotations(
        self, network, annotations, neighborhoods, min_cluster_size=5, max_cluster_size=1000
    ):
        """Define top annotations for the network.

        Args:
            network (NetworkX graph): The network graph.
            annotations (dict): Annotations for the network.
            neighborhoods (dict): Neighborhoods map with enrichment data.

        Returns:
            dict: Top annotations.
        """
        ordered_annotations = annotations["ordered_annotations"]
        neighborhood_enrichment_sums = neighborhoods["neighborhood_enrichment_counts"]
        neighborhoods_binary_enrichment_matrix = neighborhoods["binary_significance_matrix"]
        return define_top_annotations(
            network=network,
            ordered_annotation_labels=ordered_annotations,
            neighborhood_enrichment_sums=neighborhood_enrichment_sums,
            binary_enrichment_matrix=neighborhoods_binary_enrichment_matrix,
            min_cluster_size=min_cluster_size,
            max_cluster_size=max_cluster_size,
        )

    def _define_domains(
        self, neighborhoods, top_annotations, linkage_criterion, linkage_method, linkage_metric
    ):
        """Define domains in the network based on enrichment data.

        Args:
            neighborhoods (dict): Enrichment data for neighborhoods.
            top_annotations (pd.DataFrame): Enrichment matrix for top annotations.

        Returns:
            pd.DataFrame: Domains matrix.
        """
        significant_neighborhoods_enrichment = neighborhoods["significance_matrix"]
        return define_domains(
            top_annotations=top_annotations,
            significant_neighborhoods_enrichment=significant_neighborhoods_enrichment,
            linkage_criterion=linkage_criterion,
            linkage_method=linkage_method,
            linkage_metric=linkage_metric,
        )
