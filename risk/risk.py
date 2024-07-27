from rich import print

from risk.network.annotation import define_top_annotations
from risk.network.io import (
    load_annotation,
    NetworkIO,
)
from risk.network.neighborhoods import (
    define_domains,
    get_network_neighborhoods,
    trim_domains_and_top_annotations,
)
from risk.network.plot import NetworkGraph, NetworkPlotter
from risk.stats.stats import compute_pvalues_by_randomization


class RISK(NetworkIO):
    """RISK: A class for network analysis and visualization."""

    def __init__(
        self,
        compute_sphere=True,
        dimple_factor=None,
        neighborhood_diameter=0.5,
        distance_metric="shortpath",
        louvain_resolution=0.1,
        include_edge_weight=True,
        min_edges_per_node=0,
        min_cluster_size=5,
        max_cluster_size=1000,
    ):
        """Initialize RISK with configuration settings.

        Args:
            network_filepath (str): Path to the network file.
            annotation_filepath (str): Path to the annotation file.
            **kwargs: Additional configuration parameters.
        """
        super().__init__(
            compute_sphere=compute_sphere,
            dimple_factor=dimple_factor,
            include_edge_weight=include_edge_weight,
            distance_metric=distance_metric,
            neighborhood_diameter=neighborhood_diameter,
            louvain_resolution=louvain_resolution,
            min_edges_per_node=min_edges_per_node,
        )
        self.compute_sphere = compute_sphere
        self.dimple_factor = dimple_factor
        self.include_edge_weight = include_edge_weight
        self.distance_metric = distance_metric
        self.neighborhood_diameter = neighborhood_diameter
        self.louvain_resolution = louvain_resolution
        self.min_edges_per_node = min_edges_per_node
        self.min_cluster_size = min_cluster_size
        self.max_cluster_size = max_cluster_size

        # print(
        #     f"[cyan]Loading [yellow]Cytoscape[/yellow] [blue]network file[/blue]: [yellow]'{network_filename}'[/yellow]...[/cyan]"
        #     f"\n[cyan]Removing [blue]nodes[/blue] with [blue]fewer[/blue] than [red]{network_min_edges_per_node}[/red] [blue]{'edge' if network_min_edges_per_node == 1 else 'edges'}...[/blue][/cyan]"
        #     f"\n[cyan]Treating the network as {for_print_sphere} {for_print_edge_weight} [yellow]edge weights[/yellow]...[/cyan]"
        # )

    def load_annotation(self, filepath, network):
        """Load network annotations from a file.

        Args:
            network (NetworkX graph): The network graph.

        Returns:
            dict: Annotations for the network.
        """
        print("[cyan]Loading [yellow]JSON[/yellow] [blue]network annotations[/blue]...")
        return load_annotation(network, filepath)

    def load_significant_neighborhoods(
        self,
        network,
        annotations,
        score_metric="sum",
        null_distribution="network",
        tail="right",
        num_permutations=1000,
        pval_cutoff=1.00,
        apply_fdr=False,
        fdr_cutoff=1.00,
        random_seed=888,
    ):
        """Load significant neighborhoods for the network.

        Args:
            network (NetworkX graph): The network graph.

        Returns:
            dict: Neighborhoods of the network.
        """
        print(
            f"[cyan]Computing [blue]network neighborhoods[/blue] using [yellow]'{self.distance_metric}'[/yellow] as the metric..."
        )
        neighborhoods = get_network_neighborhoods(
            network,
            self.distance_metric,
            self.neighborhood_diameter,
            compute_sphere=self.compute_sphere,
            louvain_resolution=self.louvain_resolution,
        )
        print(
            f"[cyan]Computing [blue]P-values by randomization[/blue] using [yellow]'{null_distribution}'[/yellow] as the [blue]null distribution[/blue]..."
        )
        print(
            f"[cyan]Computing [blue]test statistics[/blue] using the [yellow]'{score_metric}'[/yellow]-based [blue]neighborhood scoring[/blue] approach...[/cyan]"
        )
        significant_neighborhoods = compute_pvalues_by_randomization(
            neighborhoods=neighborhoods,
            annotations=annotations["matrix"],
            score_metric=score_metric,
            tail=tail,
            num_permutations=num_permutations,
            pval_cutoff=pval_cutoff,
            apply_fdr=apply_fdr,
            fdr_cutoff=fdr_cutoff,
            random_seed=random_seed,
        )
        return significant_neighborhoods

    def define_top_annotations(self, network, annotations, neighborhoods):
        """Define top annotations for the network.

        Args:
            network (NetworkX graph): The network graph.
            annotations (dict): Annotations for the network.
            neighborhoods (dict): Neighborhoods map with enrichment data.

        Returns:
            dict: Top annotations.
        """
        ordered_annotations = annotations["ordered_annotations"]
        neighborhood_enrichment_sums = neighborhoods["enrichment_sums"]
        neighborhood_binary_enrichment_matrix_below_alpha = neighborhoods[
            "binary_enrichment_matrix_below_alpha"
        ]
        return define_top_annotations(
            network=network,
            ordered_annotation_labels=ordered_annotations,
            neighborhood_enrichment_sums=neighborhood_enrichment_sums,
            binary_enrichment_matrix_below_alpha=neighborhood_binary_enrichment_matrix_below_alpha,
            min_cluster_size=self.min_cluster_size,
            max_cluster_size=self.max_cluster_size,
        )

    def define_domains(
        self, neighborhoods, top_annotations, linkage_criterion, linkage_method, linkage_metric
    ):
        """Define domains in the network based on enrichment data.

        Args:
            neighborhoods (dict): Enrichment data for neighborhoods.
            top_annotations (pd.DataFrame): Enrichment matrix for top annotations.

        Returns:
            pd.DataFrame: Domains matrix.
        """
        neighborhoods_enrichment = neighborhoods["enrichment_matrix"]
        significant_neighborhoods_enrichment = neighborhoods["binary_enrichment_matrix_below_alpha"]
        print(f"[cyan]Optimizing [blue]distance threshold[/blue] for [blue]domains[/blue]...")
        return define_domains(
            top_annotations=top_annotations,
            neighborhoods_enrichment=neighborhoods_enrichment,
            significant_neighborhoods_enrichment=significant_neighborhoods_enrichment,
            linkage_criterion=linkage_criterion,
            linkage_method=linkage_method,
            linkage_metric=linkage_metric,
        )

    def get_network_graph(
        self,
        network,
        neighborhoods,
        top_annotations,
        domains,
    ):
        """Get a NetworkGraph object for plotting.

        Args:
            network (NetworkX graph): The network graph.
            neighborhoods (dict): Neighborhood enrichment data.
            annotations (pd.DataFrame): Annotation matrix.
            domains (pd.DataFrame): Domains matrix.
            trimmed_domains_matrix (pd.DataFrame): Trimmed domains matrix.

        Returns:
            NetworkGraph: A NetworkGraph object.
        """
        neighborhood_binary_enrichment_matrix_below_alpha = neighborhoods[
            "binary_enrichment_matrix_below_alpha"
        ]
        print("[cyan]Trimming [blue]domains[/blue]...")
        top_annotations, domains, trimmed_domains = trim_domains_and_top_annotations(
            domains=domains,
            top_annotations=top_annotations,
            min_cluster_size=self.min_cluster_size,
            max_cluster_size=self.max_cluster_size,
        )
        return NetworkGraph(
            network,
            top_annotations,
            domains,
            trimmed_domains,
            neighborhood_binary_enrichment_matrix_below_alpha,
        )

    def get_network_plotter(self, network_graph):
        """Get a NetworkPlotter object for plotting.

        Args:
            network_graph (NetworkGraph): A NetworkGraph object.

        Returns:
            NetworkPlotter: A NetworkPlotter object.
        """
        return NetworkPlotter(network_graph)
