"""
risk/network/io
~~~~~~~~~~~~~~~

This file contains the code for the RISK class and command-line access.
"""

import copy
import json
import os
import pickle
import shutil
import zipfile
from xml.dom import minidom

import networkx as nx
import numpy as np
import pandas as pd

from risk.network.geometry import assign_edge_lengths
from risk.log import params, logger, log_header


class NetworkIO:
    """A class for loading, processing, and managing network data.

    The NetworkIO class provides methods to load network data from various formats (e.g., GPickle, NetworkX)
    and process the network by adjusting node coordinates, calculating edge lengths, and validating graph structure.
    """

    def __init__(
        self,
        compute_sphere: bool = True,
        surface_depth: float = 0.0,
        min_edges_per_node: int = 0,
        include_edge_weight: bool = True,
        weight_label: str = "weight",
    ):
        """Initialize the NetworkIO class.

        Args:
            compute_sphere (bool, optional): Whether to map nodes to a sphere. Defaults to True.
            surface_depth (float, optional): Surface depth for the sphere. Defaults to 0.0.
            min_edges_per_node (int, optional): Minimum number of edges per node. Defaults to 0.
            include_edge_weight (bool, optional): Whether to include edge weights in calculations. Defaults to True.
            weight_label (str, optional): Label for edge weights. Defaults to "weight".
        """
        self.compute_sphere = compute_sphere
        self.surface_depth = surface_depth
        self.min_edges_per_node = min_edges_per_node
        self.include_edge_weight = include_edge_weight
        self.weight_label = weight_label
        # Log the initialization of the NetworkIO class
        params.log_network(
            compute_sphere=compute_sphere,
            surface_depth=surface_depth,
            min_edges_per_node=min_edges_per_node,
            include_edge_weight=include_edge_weight,
            weight_label=weight_label,
        )

    @staticmethod
    def load_gpickle_network(
        filepath: str,
        compute_sphere: bool = True,
        surface_depth: float = 0.0,
        min_edges_per_node: int = 0,
        include_edge_weight: bool = True,
        weight_label: str = "weight",
    ) -> nx.Graph:
        """Load a network from a GPickle file.

        Args:
            filepath (str): Path to the GPickle file.
            compute_sphere (bool, optional): Whether to map nodes to a sphere. Defaults to True.
            surface_depth (float, optional): Surface depth for the sphere. Defaults to 0.0.
            min_edges_per_node (int, optional): Minimum number of edges per node. Defaults to 0.
            include_edge_weight (bool, optional): Whether to include edge weights in calculations. Defaults to True.
            weight_label (str, optional): Label for edge weights. Defaults to "weight".

        Returns:
            nx.Graph: Loaded and processed network.
        """
        networkio = NetworkIO(
            compute_sphere=compute_sphere,
            surface_depth=surface_depth,
            min_edges_per_node=min_edges_per_node,
            include_edge_weight=include_edge_weight,
            weight_label=weight_label,
        )
        return networkio._load_gpickle_network(filepath=filepath)

    def _load_gpickle_network(self, filepath: str) -> nx.Graph:
        """Private method to load a network from a GPickle file.

        Args:
            filepath (str): Path to the GPickle file.

        Returns:
            nx.Graph: Loaded and processed network.
        """
        filetype = "GPickle"
        # Log the loading of the GPickle file
        params.log_network(filetype=filetype, filepath=filepath)
        self._log_loading(filetype, filepath=filepath)

        with open(filepath, "rb") as f:
            G = pickle.load(f)

        # Initialize the graph
        return self._initialize_graph(G)

    @staticmethod
    def load_networkx_network(
        network: nx.Graph,
        compute_sphere: bool = True,
        surface_depth: float = 0.0,
        min_edges_per_node: int = 0,
        include_edge_weight: bool = True,
        weight_label: str = "weight",
    ) -> nx.Graph:
        """Load a NetworkX graph.

        Args:
            network (nx.Graph): A NetworkX graph object.
            compute_sphere (bool, optional): Whether to map nodes to a sphere. Defaults to True.
            surface_depth (float, optional): Surface depth for the sphere. Defaults to 0.0.
            min_edges_per_node (int, optional): Minimum number of edges per node. Defaults to 0.
            include_edge_weight (bool, optional): Whether to include edge weights in calculations. Defaults to True.
            weight_label (str, optional): Label for edge weights. Defaults to "weight".

        Returns:
            nx.Graph: Loaded and processed network.
        """
        networkio = NetworkIO(
            compute_sphere=compute_sphere,
            surface_depth=surface_depth,
            min_edges_per_node=min_edges_per_node,
            include_edge_weight=include_edge_weight,
            weight_label=weight_label,
        )
        return networkio._load_networkx_network(network=network)

    def _load_networkx_network(self, network: nx.Graph) -> nx.Graph:
        """Private method to load a NetworkX graph.

        Args:
            network (nx.Graph): A NetworkX graph object.

        Returns:
            nx.Graph: Processed network.
        """
        filetype = "NetworkX"
        # Log the loading of the NetworkX graph
        params.log_network(filetype=filetype)
        self._log_loading(filetype)

        # Important: Make a copy of the network to avoid modifying the original
        network_copy = copy.deepcopy(network)
        # Initialize the graph
        return self._initialize_graph(network_copy)

    @staticmethod
    def load_cytoscape_network(
        filepath: str,
        source_label: str = "source",
        target_label: str = "target",
        view_name: str = "",
        compute_sphere: bool = True,
        surface_depth: float = 0.0,
        min_edges_per_node: int = 0,
        include_edge_weight: bool = True,
        weight_label: str = "weight",
    ) -> nx.Graph:
        """Load a network from a Cytoscape file.

        Args:
            filepath (str): Path to the Cytoscape file.
            source_label (str, optional): Source node label. Defaults to "source".
            target_label (str, optional): Target node label. Defaults to "target".
            view_name (str, optional): Specific view name to load. Defaults to "".
            compute_sphere (bool, optional): Whether to map nodes to a sphere. Defaults to True.
            surface_depth (float, optional): Surface depth for the sphere. Defaults to 0.0.
            min_edges_per_node (int, optional): Minimum number of edges per node. Defaults to 0.
            include_edge_weight (bool, optional): Whether to include edge weights in calculations. Defaults to True.
            weight_label (str, optional): Label for edge weights. Defaults to "weight".

        Returns:
            nx.Graph: Loaded and processed network.
        """
        networkio = NetworkIO(
            compute_sphere=compute_sphere,
            surface_depth=surface_depth,
            min_edges_per_node=min_edges_per_node,
            include_edge_weight=include_edge_weight,
            weight_label=weight_label,
        )
        return networkio._load_cytoscape_network(
            filepath=filepath,
            source_label=source_label,
            target_label=target_label,
            view_name=view_name,
        )

    def _load_cytoscape_network(
        self,
        filepath: str,
        source_label: str = "source",
        target_label: str = "target",
        view_name: str = "",
    ) -> nx.Graph:
        """Private method to load a network from a Cytoscape file.

        Args:
            filepath (str): Path to the Cytoscape file.
            source_label (str, optional): Source node label. Defaults to "source".
            target_label (str, optional): Target node label. Defaults to "target".
            view_name (str, optional): Specific view name to load. Defaults to "".

        Returns:
            nx.Graph: Loaded and processed network.
        """
        filetype = "Cytoscape"
        # Log the loading of the Cytoscape file
        params.log_network(filetype=filetype, filepath=str(filepath))
        self._log_loading(filetype, filepath=filepath)

        cys_files = []
        tmp_dir = ".tmp_cytoscape"
        # Try / finally to remove unzipped files
        try:
            # Create the temporary directory if it doesn't exist
            if not os.path.exists(tmp_dir):
                os.makedirs(tmp_dir)

            # Unzip CYS file into the temporary directory
            with zipfile.ZipFile(filepath, "r") as zip_ref:
                cys_files = zip_ref.namelist()
                zip_ref.extractall(tmp_dir)

            # Get first view and network instances
            cys_view_files = [os.path.join(tmp_dir, cf) for cf in cys_files if "/views/" in cf]
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
                os.path.join(tmp_dir, cf)
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

            # Initialize the graph
            return self._initialize_graph(G)

        finally:
            # Remove the temporary directory and its contents
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)

    @staticmethod
    def load_cytoscape_json_network(
        filepath: str,
        source_label: str = "source",
        target_label: str = "target",
        compute_sphere: bool = True,
        surface_depth: float = 0.0,
        min_edges_per_node: int = 0,
        include_edge_weight: bool = True,
        weight_label: str = "weight",
    ) -> nx.Graph:
        """Load a network from a Cytoscape JSON (.cyjs) file.

        Args:
            filepath (str): Path to the Cytoscape JSON file.
            source_label (str, optional): Source node label. Default is "source".
            target_label (str, optional): Target node label. Default is "target".
            compute_sphere (bool, optional): Whether to map nodes to a sphere. Defaults to True.
            surface_depth (float, optional): Surface depth for the sphere. Defaults to 0.0.
            min_edges_per_node (int, optional): Minimum number of edges per node. Defaults to 0.
            include_edge_weight (bool, optional): Whether to include edge weights in calculations. Defaults to True.
            weight_label (str, optional): Label for edge weights. Defaults to "weight".

        Returns:
            NetworkX graph: Loaded and processed network.
        """
        networkio = NetworkIO(
            compute_sphere=compute_sphere,
            surface_depth=surface_depth,
            min_edges_per_node=min_edges_per_node,
            include_edge_weight=include_edge_weight,
            weight_label=weight_label,
        )
        return networkio._load_cytoscape_json_network(
            filepath=filepath,
            source_label=source_label,
            target_label=target_label,
        )

    def _load_cytoscape_json_network(self, filepath, source_label="source", target_label="target"):
        """Private method to load a network from a Cytoscape JSON (.cyjs) file.

        Args:
            filepath (str): Path to the Cytoscape JSON file.
            source_label (str, optional): Source node label. Default is "source".
            target_label (str, optional): Target node label. Default is "target".

        Returns:
            NetworkX graph: Loaded and processed network.
        """
        filetype = "Cytoscape JSON"
        # Log the loading of the Cytoscape JSON file
        params.log_network(filetype=filetype, filepath=str(filepath))
        self._log_loading(filetype, filepath=filepath)

        # Load the Cytoscape JSON file
        with open(filepath, "r") as f:
            cyjs_data = json.load(f)

        # Create a graph
        G = nx.Graph()
        # Store node positions for later use
        node_x_positions = {}
        node_y_positions = {}
        for node in cyjs_data["elements"]["nodes"]:
            node_data = node["data"]
            node_id = node_data["id_original"]
            node_x_positions[node_id] = node["position"]["x"]
            node_y_positions[node_id] = node["position"]["y"]

        # Process edges and add them to the graph
        for edge in cyjs_data["elements"]["edges"]:
            edge_data = edge["data"]
            source = edge_data[f"{source_label}_original"]
            target = edge_data[f"{target_label}_original"]
            # Add the edge to the graph, optionally including weights
            if self.weight_label is not None and self.weight_label in edge_data:
                weight = float(edge_data[self.weight_label])
                G.add_edge(source, target, weight=weight)
            else:
                G.add_edge(source, target)

            # Ensure nodes exist in the graph and add them if not present
            if source not in G:
                G.add_node(source)
            if target not in G:
                G.add_node(target)

        # Add node attributes (like label, x, y positions)
        for node in G.nodes():
            G.nodes[node]["label"] = node
            G.nodes[node]["x"] = node_x_positions.get(node, 0)  # Use stored positions
            G.nodes[node]["y"] = node_y_positions.get(node, 0)  # Use stored positions

        # Initialize the graph
        return self._initialize_graph(G)

    def _initialize_graph(self, G: nx.Graph) -> nx.Graph:
        """Initialize the graph by processing and validating its nodes and edges.

        Args:
            G (nx.Graph): The input NetworkX graph.

        Returns:
            nx.Graph: The processed and validated graph.
        """
        self._validate_nodes(G)
        self._assign_edge_weights(G)
        self._assign_edge_lengths(G)
        self._remove_invalid_graph_properties(G)
        # IMPORTANT: This is where the graph node labels are converted to integers
        # Make sure to perform this step after all other processing
        G = nx.relabel_nodes(G, {node: idx for idx, node in enumerate(G.nodes)})
        return G

    def _remove_invalid_graph_properties(self, G: nx.Graph) -> None:
        """Remove invalid properties from the graph, including self-loops, nodes with fewer edges than
        the threshold, and isolated nodes.

        Args:
            G (nx.Graph): A NetworkX graph object.
        """
        # Count number of nodes and edges before cleaning
        num_initial_nodes = G.number_of_nodes()
        num_initial_edges = G.number_of_edges()
        # Remove self-loops to ensure correct edge count
        G.remove_edges_from(list(nx.selfloop_edges(G)))
        # Iteratively remove nodes with fewer edges than the threshold
        while True:
            nodes_to_remove = [node for node in G.nodes if G.degree(node) < self.min_edges_per_node]
            if not nodes_to_remove:
                break  # Exit loop if no more nodes need removal
            G.remove_nodes_from(nodes_to_remove)

        # Remove isolated nodes
        isolated_nodes = list(nx.isolates(G))
        if isolated_nodes:
            G.remove_nodes_from(isolated_nodes)

        # Log the number of nodes and edges before and after cleaning
        num_final_nodes = G.number_of_nodes()
        num_final_edges = G.number_of_edges()
        logger.debug(f"Initial node count: {num_initial_nodes}")
        logger.debug(f"Final node count: {num_final_nodes}")
        logger.debug(f"Initial edge count: {num_initial_edges}")
        logger.debug(f"Final edge count: {num_final_edges}")

    def _assign_edge_weights(self, G: nx.Graph) -> None:
        """Assign weights to the edges in the graph.

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
            logger.debug(f"Total edges missing weights: {missing_weights}")

    def _validate_nodes(self, G: nx.Graph) -> None:
        """Validate the graph structure and attributes with attribute fallback for positions and labels.

        Args:
            G (nx.Graph): A NetworkX graph object.
        """
        # Keep track of nodes missing labels
        nodes_with_missing_labels = []

        for node, attrs in G.nodes(data=True):
            # Attribute fallback for 'x' and 'y' attributes
            if "x" not in attrs or "y" not in attrs:
                if (
                    "pos" in attrs
                    and isinstance(attrs["pos"], (list, tuple, np.ndarray))
                    and len(attrs["pos"]) >= 2
                ):
                    attrs["x"], attrs["y"] = attrs["pos"][
                        :2
                    ]  # Use only x and y, ignoring z if present
                else:
                    raise ValueError(
                        f"Node {node} is missing 'x', 'y', and a valid 'pos' attribute."
                    )

            # Attribute fallback for 'label' attribute
            if "label" not in attrs:
                # Try alternative attribute names for label
                if "name" in attrs:
                    attrs["label"] = attrs["name"]
                elif "id" in attrs:
                    attrs["label"] = attrs["id"]
                else:
                    # Collect nodes with missing labels
                    nodes_with_missing_labels.append(node)
                    attrs["label"] = str(node)  # Use node ID as the label

        # Issue a single warning if any labels were missing
        if nodes_with_missing_labels:
            total_nodes = len(G.nodes)
            fraction_missing_labels = len(nodes_with_missing_labels) / total_nodes
            logger.warning(
                f"{len(nodes_with_missing_labels)} out of {total_nodes} nodes "
                f"({fraction_missing_labels:.2%}) were missing 'label' attributes and were assigned node IDs."
            )

    def _assign_edge_lengths(self, G: nx.Graph) -> None:
        """Prepare the network by adjusting surface depth and calculating edge lengths.

        Args:
            G (nx.Graph): The input network graph.
        """
        assign_edge_lengths(
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
        log_header("Loading network")
        logger.debug(f"Filetype: {filetype}")
        if filepath:
            logger.debug(f"Filepath: {filepath}")
        logger.debug(f"Edge weight: {'Included' if self.include_edge_weight else 'Excluded'}")
        if self.include_edge_weight:
            logger.debug(f"Weight label: {self.weight_label}")
        logger.debug(f"Minimum edges per node: {self.min_edges_per_node}")
        logger.debug(f"Projection: {'Sphere' if self.compute_sphere else 'Plane'}")
        if self.compute_sphere:
            logger.debug(f"Surface depth: {self.surface_depth}")
