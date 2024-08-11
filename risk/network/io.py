"""
risk/network/io
~~~~~~~~~~~~~~~

This file contains the code for the RISK class and command-line access.
"""

import json
import pickle
import shutil
import zipfile
from xml.dom import minidom

import networkx as nx
import pandas as pd

from risk.network.geometry import apply_edge_lengths
from risk.log import params, print_header


class NetworkIO:
    """A class for loading, processing, and managing network data.

    The NetworkIO class provides methods to load network data from various formats (e.g., GPickle, NetworkX)
    and process the network by adjusting node coordinates, calculating edge lengths, and validating graph structure.
    """

    def __init__(
        self,
        compute_sphere: bool = True,
        surface_depth: float = 0.0,
        distance_metric: str = "dijkstra",
        edge_length_threshold: float = 0.5,
        louvain_resolution: float = 0.1,
        min_edges_per_node: int = 0,
        include_edge_weight: bool = True,
        weight_label: str = "weight",
    ):
        self.compute_sphere = compute_sphere
        self.surface_depth = surface_depth
        self.include_edge_weight = include_edge_weight
        self.weight_label = weight_label
        self.distance_metric = distance_metric
        self.edge_length_threshold = edge_length_threshold
        self.louvain_resolution = louvain_resolution
        self.min_edges_per_node = min_edges_per_node

    def load_gpickle_network(self, filepath: str) -> nx.Graph:
        """Load a network from a GPickle file.

        Args:
            filepath (str): Path to the GPickle file.

        Returns:
            nx.Graph: Loaded and processed network.
        """
        filetype = "GPickle"
        params.log_network(filetype=filetype, filepath=filepath)
        self._log_loading(filetype, filepath=filepath)
        with open(filepath, "rb") as f:
            G = pickle.load(f)

        return self._initialize_graph(G)

    def load_networkx_network(self, G: nx.Graph) -> nx.Graph:
        """Load a NetworkX graph.

        Args:
            G (nx.Graph): A NetworkX graph object.

        Returns:
            nx.Graph: Processed network.
        """
        filetype = "NetworkX"
        params.log_network(filetype=filetype)
        self._log_loading(filetype)
        return self._initialize_graph(G)

    def load_cytoscape_network(
        self,
        filepath: str,
        source_label: str = "source",
        target_label: str = "target",
        view_name: str = "",
    ) -> nx.Graph:
        """Load a network from a Cytoscape file.

        Args:
            filepath (str): Path to the Cytoscape file.
            source_label (str, optional): Source node label. Defaults to "source".
            target_label (str, optional): Target node label. Defaults to "target".
            view_name (str, optional): Specific view name to load. Defaults to None.

        Returns:
            nx.Graph: Loaded and processed network.
        """
        filetype = "Cytoscape"
        params.log_network(filetype=filetype, filepath=str(filepath))
        self._log_loading(filetype, filepath=filepath)
        cys_files = []
        # Try / finally to remove unzipped files
        try:
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
            node_x_positions = {}
            node_y_positions = {}
            for node in cys_nodes:
                # Node ID is found in 'label'
                node_id = str(node.attributes["label"].value)
                for child in node.childNodes:
                    if child.nodeType == 1 and child.tagName == "graphics":
                        node_x_positions[node_id] = float(child.attributes["x"].value)
                        node_y_positions[node_id] = float(child.attributes["y"].value)

            # Read the node attributes (from /tables/)
            attribute_metadata_keywords = ["/tables/", "SHARED_ATTRS", "edge.cytable"]
            attribute_metadata = [
                cf
                for cf in cys_files
                if all(keyword in cf for keyword in attribute_metadata_keywords)
            ][0]
            # Load attributes file from Cytoscape as pandas data frame
            attribute_table = pd.read_csv(attribute_metadata, sep=",", header=None, skiprows=1)
            # Set columns
            attribute_table.columns = attribute_table.iloc[0]
            # Skip first four rows
            attribute_table = attribute_table.iloc[4:, :]
            # Conditionally select columns based on include_edge_weight
            if self.include_edge_weight:
                attribute_table = attribute_table[[source_label, target_label, self.weight_label]]
            else:
                attribute_table = attribute_table[[source_label, target_label]]

            attribute_table = attribute_table.dropna().reset_index(drop=True)
            # Create a graph
            G = nx.Graph()
            # Add edges and nodes, conditionally including weights
            for _, row in attribute_table.iterrows():
                source = row[source_label]
                target = row[target_label]
                if self.include_edge_weight:
                    weight = float(row[self.weight_label])
                    G.add_edge(source, target, weight=weight)
                else:
                    G.add_edge(source, target)

                if source not in G:
                    G.add_node(source)  # Optionally add x, y coordinates here if available
                if target not in G:
                    G.add_node(target)  # Optionally add x, y coordinates here if available

            # Add node attributes
            for node in G.nodes():
                G.nodes[node]["label"] = node
                G.nodes[node]["x"] = node_x_positions[
                    node
                ]  # Assuming you have a dict `node_x_positions` for x coordinates
                G.nodes[node]["y"] = node_y_positions[
                    node
                ]  # Assuming you have a dict `node_y_positions` for y coordinates

            return self._initialize_graph(G)

        finally:
            # Remove unzipped files/directories
            cys_dirnames = list(set([cf.split("/")[0] for cf in cys_files]))
            for dirname in cys_dirnames:
                shutil.rmtree(dirname)

    def load_cytoscape_json_network(self, filepath, source_label="source", target_label="target"):
        """Load a network from a Cytoscape JSON (.cyjs) file.

        Args:
            filepath (str): Path to the Cytoscape JSON file.
            source_label (str, optional): Source node label. Default is "source".
            target_label (str, optional): Target node label. Default is "target".

        Returns:
            NetworkX graph: Loaded and processed network.
        """
        filetype = "Cytoscape JSON"
        params.log_network(filetype=filetype, filepath=str(filepath))
        self._log_loading(filetype, filepath=filepath)
        # Load the Cytoscape JSON file
        with open(filepath, "r") as f:
            cyjs_data = json.load(f)

        # Create a graph
        G = nx.Graph()
        # Process nodes
        node_x_positions = {}
        node_y_positions = {}
        for node in cyjs_data["elements"]["nodes"]:
            node_data = node["data"]
            node_id = node_data["id"]
            node_x_positions[node_id] = node["position"]["x"]
            node_y_positions[node_id] = node["position"]["y"]
            G.add_node(node_id)
            G.nodes[node_id]["label"] = node_data.get("name", node_id)
            G.nodes[node_id]["x"] = node["position"]["x"]
            G.nodes[node_id]["y"] = node["position"]["y"]

        # Process edges
        for edge in cyjs_data["elements"]["edges"]:
            edge_data = edge["data"]
            source = edge_data[source_label]
            target = edge_data[target_label]
            if self.weight_label is not None and self.weight_label in edge_data:
                weight = float(edge_data[self.weight_label])
                G.add_edge(source, target, weight=weight)
            else:
                G.add_edge(source, target)

        # Initialize the graph
        return self._initialize_graph(G)

    def _initialize_graph(self, G: nx.Graph) -> nx.Graph:
        """Initialize the graph by processing and validating its nodes and edges.

        Args:
            G (nx.Graph): The input NetworkX graph.

        Returns:
            nx.Graph: The processed and validated graph.
        """
        # IMPORTANT: This is where the graph node labels are converted to integers
        G = nx.relabel_nodes(G, {node: idx for idx, node in enumerate(G.nodes)})
        self._remove_invalid_graph_properties(G)
        self._validate_edges(G)
        self._validate_nodes(G)
        self._process_graph(G)
        return G

    def _remove_invalid_graph_properties(self, G: nx.Graph) -> None:
        """Remove invalid properties from the graph.

        Args:
            G (nx.Graph): A NetworkX graph object.
        """
        print(f"Minimum edges per node: {self.min_edges_per_node}")
        # Remove nodes with fewer edges than the specified threshold
        nodes_with_few_edges = [
            node for node in G.nodes() if G.degree(node) <= self.min_edges_per_node
        ]
        G.remove_nodes_from(nodes_with_few_edges)
        # Remove self-loop edges
        self_loops = list(nx.selfloop_edges(G))
        G.remove_edges_from(self_loops)

    def _validate_edges(self, G: nx.Graph) -> None:
        """Validate and assign weights to the edges in the graph.

        Args:
            G (nx.Graph): A NetworkX graph object.
        """
        missing_weights = 0
        # Assign user-defined edge weights to the "weight" attribute
        for _, _, data in G.edges(data=True):
            if self.weight_label not in data:
                missing_weights += 1
            data["weight"] = data.get(
                self.weight_label, 1.0
            )  # Default to 1.0 if 'weight' not present

        if self.include_edge_weight and missing_weights:
            print(f"Total edges missing weights: {missing_weights}")

    def _validate_nodes(self, G: nx.Graph) -> None:
        """Validate the graph structure and attributes.

        Args:
            G (nx.Graph): A NetworkX graph object.
        """
        for node, attrs in G.nodes(data=True):
            assert (
                "x" in attrs and "y" in attrs
            ), f"Node {node} is missing 'x' or 'y' position attributes."
            assert "label" in attrs, f"Node {node} is missing a 'label' attribute."

    def _process_graph(self, G: nx.Graph) -> None:
        """Prepare the network by adjusting surface depth and calculating edge lengths.

        Args:
            G (nx.Graph): The input network graph.
        """
        apply_edge_lengths(
            G,
            compute_sphere=self.compute_sphere,
            surface_depth=self.surface_depth,
            include_edge_weight=self.include_edge_weight,
        )

    def _log_loading(
        self,
        filetype: str,
        filepath: str = "",
    ) -> None:
        """Log the initialization details of the RISK class.

        Args:
            filetype (str): The type of the file being loaded (e.g., 'CSV', 'JSON').
            filepath (str, optional): The path to the file being loaded. Defaults to "".
        """
        print_header("Loading network")
        print(f"Filetype: {filetype}")
        if filepath:
            print(f"Filepath: {filepath}")
        print(f"Project to sphere: {self.compute_sphere}")
        if self.compute_sphere:
            print(f"Surface depth: {self.surface_depth}")
        print(f"Edge length threshold: {self.edge_length_threshold}")
        print(f"Include edge weights: {self.include_edge_weight}")
        if self.include_edge_weight:
            print(f"Weight label: {self.weight_label}")
