"""
risk/network/io/network
~~~~~~~~~~~~~~~~~~~~~~~

This file contains the code for the RISK class and command-line access.
"""

import pickle
import shutil
import warnings
import zipfile
from xml.dom import minidom

import networkx as nx
import pandas as pd
from rich import print

from risk.graph.metrics import calculate_edge_lengths, get_best_surface_depth


class NetworkIO:
    def __init__(
        self,
        compute_sphere=True,
        surface_depth=None,
        include_edge_weight=True,
        distance_metric="dijkstra",
        neighborhood_diameter=0.5,
        louvain_resolution=0.1,
        min_edges_per_node=0,
    ):
        _log_initialization(compute_sphere, surface_depth, include_edge_weight)
        self.compute_sphere = compute_sphere
        self.surface_depth = surface_depth
        self.include_edge_weight = include_edge_weight
        self.distance_metric = distance_metric
        self.neighborhood_diameter = neighborhood_diameter
        self.louvain_resolution = louvain_resolution
        self.min_edges_per_node = min_edges_per_node

    def load_gpickle_network(self, filepath):
        """Load a network from a GPickle file.

        Args:
            filepath (str): Path to the GPickle file.

        Returns:
            NetworkX graph: Loaded network.
        """
        _log_network_loading("GPickle", filepath)
        with open(filepath, "rb") as f:
            G = pickle.load(f)
        return self._load_networkx(G)

    def load_networkx_network(self, G):
        """Load a NetworkX graph.

        Args:
            G (NetworkX graph): A NetworkX graph object.

        Returns:
            NetworkX graph: Processed network.
        """
        _log_network_loading("NetworkX")
        return self._load_networkx(G)

    def load_cytoscape_network(
        self,
        filepath,
        source_label="source",
        target_label="target",
        weight_label="weight",
        view_name=None,
    ):
        """Load a network from a Cytoscape file.

        Args:
            filepath (str): Path to the Cytoscape file.
            source_label (str, optional): Source node label. Default is "source".
            target_label (str, optional): Target node label. Default is "target".
            weight_label (str, optional): Edge weight label. Default is "weight".
            view_name (str, optional): Specific view name to load. Default is None.
            min_edges_per_node (int, optional): Minimum number of edges per node. Default is 0.

        Returns:
            NetworkX graph: Loaded and processed network.
        """
        _log_network_loading("Cytoscape", filepath)
        # Unzip CYS file
        with zipfile.ZipFile(filepath, "r") as zip_ref:
            cys_files = zip_ref.namelist()
            zip_ref.extractall("./")
        # Get first view and network instances
        cys_view_files = [cf for cf in cys_files if "/views/" in cf]
        cys_view_file = (
            cys_view_files[0]
            if not view_name
            else [cvf for cvf in cys_view_files if cvf.endswith(view_name + ".xgmml")][0]
        )
        # Parse nodes
        cys_view_dom = minidom.parse(cys_view_file)
        cys_nodes = cys_view_dom.getElementsByTagName("node")
        node_xs = {}
        node_ys = {}
        for node in cys_nodes:
            # Node ID is found in 'label'
            node_id = str(node.attributes["label"].value)
            for child in node.childNodes:
                if child.nodeType == 1 and child.tagName == "graphics":
                    node_xs[node_id] = float(child.attributes["x"].value)
                    node_ys[node_id] = float(child.attributes["y"].value)

        # Read the node attributes (from /tables/)
        attribute_metadata_keywords = ["/tables/", "SHARED_ATTRS", "edge.cytable"]
        attribute_metadata = [
            cf for cf in cys_files if all(keyword in cf for keyword in attribute_metadata_keywords)
        ][0]
        # Load attributes file from Cytoscape as pandas data frame
        attribute_table = pd.read_csv(attribute_metadata, sep=",", header=None, skiprows=1)
        # Set columns
        attribute_table.columns = attribute_table.iloc[0]
        # Skip first four rows
        attribute_table = attribute_table.iloc[4:, :]
        attribute_table = attribute_table[[source_label, target_label, weight_label]]
        attribute_table = attribute_table.dropna().reset_index(drop=True)

        # Create a graph
        G = nx.Graph()
        # Add edges and nodes with weights
        for _, row in attribute_table.iterrows():
            source, target, weight = (
                row[source_label],
                row[target_label],
                float(row[weight_label]),
            )
            if source not in G:
                G.add_node(source)  # Optionally add x, y coordinates here if available
            if target not in G:
                G.add_node(target)  # Optionally add x, y coordinates here if available
            G.add_edge(source, target, weight=weight)

        # Remove invalid graph attributes / properties as soon as edges are added
        self._remove_invalid_graph_properties(G)

        for node in G.nodes():
            G.nodes[node]["label"] = node
            G.nodes[node]["x"] = node_xs[
                node
            ]  # Assuming you have a dict `node_xs` for x coordinates
            G.nodes[node]["y"] = node_ys[
                node
            ]  # Assuming you have a dict `node_ys` for y coordinates

        # Relabel the node ids to sequential numbers to make calculations faster
        G = nx.relabel_nodes(G, {node: idx for idx, node in enumerate(G.nodes)})
        self._validate_graph(G)
        G = self._process_graph(G)
        # Remove unzipped files/directories
        cys_dirnames = list(set([cf.split("/")[0] for cf in cys_files]))
        for dirname in cys_dirnames:
            shutil.rmtree(dirname)

        return G

    def _load_networkx(self, G):
        """Internal method to process NetworkX graph.

        Args:
            G (NetworkX graph): A NetworkX graph object.

        Returns:
            NetworkX graph: Processed network.
        """
        self._remove_invalid_graph_properties(G)
        G = nx.relabel_nodes(G, {node: idx for idx, node in enumerate(G.nodes)})
        self._validate_graph(G)
        return G

    def _remove_invalid_graph_properties(self, G):
        """Remove invalid properties from the graph.

        Args:
            G (NetworkX graph): A NetworkX graph object.
            min_edges_per_node (int): Minimum number of edges per node.
        """
        print(
            f"[cyan]Removing [blue]nodes[/blue] with [blue]fewer[/blue] than [red]{self.min_edges_per_node}[/red] [blue]edge(s)...[/blue][/cyan]"
        )
        nodes_with_few_edges = [
            node for node in G.nodes() if G.degree(node) <= self.min_edges_per_node
        ]
        G.remove_nodes_from(nodes_with_few_edges)
        self_loops = list(nx.selfloop_edges(G))
        G.remove_edges_from(self_loops)

    @staticmethod
    def _validate_graph(graph):
        """Validate the graph structure and attributes.

        Args:
            graph (NetworkX graph): A NetworkX graph object.
        """
        for node, attrs in graph.nodes(data=True):
            assert (
                "x" in attrs and "y" in attrs
            ), f"Node {node} is missing 'x' or 'y' position attributes."
            assert "label" in attrs, f"Node {node} is missing a 'label' attribute."

        missing_weights = [edge for edge in graph.edges(data=True) if "weight" not in edge[2]]
        if missing_weights:
            warnings.warn(
                "Some edges are missing weights; default weight of 1 will be used for missing weights."
            )

    def _process_graph(self, G):
        """Prepare network by adjusting surface depth and get neighborhoods.

        Args:
            G (NetworkX graph): The input network.

        Returns:
            Tuple: Processed network and neighborhoods.
        """
        if self.surface_depth is None and self.compute_sphere:
            self.surface_depth = get_best_surface_depth(
                G.copy(),
                compute_sphere=self.compute_sphere,
                include_edge_weight=self.include_edge_weight,
                distance_metric=self.distance_metric,
                neighborhood_diameter=self.neighborhood_diameter,
                louvain_resolution=self.louvain_resolution,
            )

        G = calculate_edge_lengths(
            G.copy(),
            compute_sphere=self.compute_sphere,
            surface_depth=self.surface_depth,
            include_edge_weight=self.include_edge_weight,
        )
        return G


def _log_initialization(compute_sphere, surface_depth, include_edge_weight):
    """Log the initialization of the RISK class."""
    for_print_sphere = "[yellow]spherical[/yellow]" if compute_sphere else "[yellow]planar[/yellow]"
    for_print_surface_depth = (
        f"with a [yellow]surface depth[/yellow] of [red]{surface_depth}[/red]"
        if surface_depth and compute_sphere
        else "without a [yellow]surface depth[/yellow]"
    )
    for_print_edge_weight = (
        "with [yellow]edge weights[/yellow]"
        if include_edge_weight
        else "without [yellow]edge weights[/yellow]"
    )
    print(
        f"[cyan]Treating the network as {for_print_sphere} {for_print_surface_depth} {for_print_edge_weight}...[/cyan]"
    )


def _log_network_loading(filetype, filepath=None):
    """Log information about the network file being loaded.

    Args:
        filetype (str): The type of the file being loaded (e.g., 'Cytoscape').
        filepath (str, optional): The path to the file being loaded.
    """
    if filepath:
        message = f"[cyan]Loading [yellow]{filetype}[/yellow] [blue]network file[/blue]: [yellow]'{filepath}'[/yellow]...[/cyan]"
    else:
        message = f"[cyan]Loading [yellow]{filetype}[/yellow] [blue]network file[/blue]...[/cyan]"
    print(message)
