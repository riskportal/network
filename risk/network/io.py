"""
risk/network/io
~~~~~~~~~~~~~~~
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
    ):
        """Initialize the NetworkIO class.

        Args:
            compute_sphere (bool, optional): Whether to map nodes to a sphere. Defaults to True.
            surface_depth (float, optional): Surface depth for the sphere. Defaults to 0.0.
            min_edges_per_node (int, optional): Minimum number of edges per node. Defaults to 0.
        """
        self.compute_sphere = compute_sphere
        self.surface_depth = surface_depth
        self.min_edges_per_node = min_edges_per_node
        # Log the initialization of the NetworkIO class
        params.log_network(
            compute_sphere=compute_sphere,
            surface_depth=surface_depth,
            min_edges_per_node=min_edges_per_node,
        )

    @staticmethod
    def load_gpickle_network(
        filepath: str,
        compute_sphere: bool = True,
        surface_depth: float = 0.0,
        min_edges_per_node: int = 0,
    ) -> nx.Graph:
        """Load a network from a GPickle file.

        Args:
            filepath (str): Path to the GPickle file.
            compute_sphere (bool, optional): Whether to map nodes to a sphere. Defaults to True.
            surface_depth (float, optional): Surface depth for the sphere. Defaults to 0.0.
            min_edges_per_node (int, optional): Minimum number of edges per node. Defaults to 0.

        Returns:
            nx.Graph: Loaded and processed network.
        """
        networkio = NetworkIO(
            compute_sphere=compute_sphere,
            surface_depth=surface_depth,
            min_edges_per_node=min_edges_per_node,
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
    ) -> nx.Graph:
        """Load a NetworkX graph.

        Args:
            network (nx.Graph): A NetworkX graph object.
            compute_sphere (bool, optional): Whether to map nodes to a sphere. Defaults to True.
            surface_depth (float, optional): Surface depth for the sphere. Defaults to 0.0.
            min_edges_per_node (int, optional): Minimum number of edges per node. Defaults to 0.

        Returns:
            nx.Graph: Loaded and processed network.
        """
        networkio = NetworkIO(
            compute_sphere=compute_sphere,
            surface_depth=surface_depth,
            min_edges_per_node=min_edges_per_node,
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

        Returns:
            nx.Graph: Loaded and processed network.
        """
        networkio = NetworkIO(
            compute_sphere=compute_sphere,
            surface_depth=surface_depth,
            min_edges_per_node=min_edges_per_node,
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

        Raises:
            ValueError: If no matching attribute metadata file is found.
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
            # Use a generator to find the first matching file
            attribute_metadata = next(
                (
                    os.path.join(tmp_dir, cf)
                    for cf in cys_files
                    if all(keyword in cf for keyword in attribute_metadata_keywords)
                ),
                None,  # Default if no file matches
            )
            if attribute_metadata:
                # Optimize `read_csv` by leveraging proper options
                attribute_table = pd.read_csv(
                    attribute_metadata,
                    sep=",",
                    header=None,
                    skiprows=1,
                    dtype=str,  # Use specific dtypes to reduce memory usage
                    engine="c",  # Use the C engine for parsing if compatible
                    low_memory=False,  # Optimize memory handling for large files
                )
            else:
                raise ValueError("No matching attribute metadata file found.")

            # Set columns
            attribute_table.columns = attribute_table.iloc[0]
            # Skip first four rows, select source and target columns, and reset index
            attribute_table = attribute_table.iloc[4:, :]
            try:
                # Attempt to filter the attribute_table with the given labels
                attribute_table = attribute_table[[source_label, target_label]]
            except KeyError as e:
                # Find which key(s) caused the issue
                missing_keys = [
                    key
                    for key in [source_label, target_label]
                    if key not in attribute_table.columns
                ]
                # Raise the KeyError with details about the issue and available options
                available_columns = ", ".join(attribute_table.columns)
                raise KeyError(
                    f"The column(s) '{', '.join(missing_keys)}' do not exist in the table. "
                    f"Available columns are: {available_columns}."
                ) from e

            attribute_table = attribute_table.dropna().reset_index(drop=True)

            # Create a graph
            G = nx.Graph()
            # Add edges and nodes
            for _, row in attribute_table.iterrows():
                source = row[source_label]
                target = row[target_label]
                G.add_edge(source, target)
                if source not in G:
                    G.add_node(source)  # Optionally add x, y coordinates here if available
                if target not in G:
                    G.add_node(target)  # Optionally add x, y coordinates here if available

            # Add node attributes
            for node in G.nodes():
                G.nodes[node]["label"] = node
                G.nodes[node]["x"] = node_x_positions[node]
                G.nodes[node]["y"] = node_y_positions[node]

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
    ) -> nx.Graph:
        """Load a network from a Cytoscape JSON (.cyjs) file.

        Args:
            filepath (str): Path to the Cytoscape JSON file.
            source_label (str, optional): Source node label. Default is "source".
            target_label (str, optional): Target node label. Default is "target".
            compute_sphere (bool, optional): Whether to map nodes to a sphere. Defaults to True.
            surface_depth (float, optional): Surface depth for the sphere. Defaults to 0.0.
            min_edges_per_node (int, optional): Minimum number of edges per node. Defaults to 0.

        Returns:
            NetworkX graph: Loaded and processed network.
        """
        networkio = NetworkIO(
            compute_sphere=compute_sphere,
            surface_depth=surface_depth,
            min_edges_per_node=min_edges_per_node,
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
            # Use the original node ID if available, otherwise use the default ID
            node_id = node_data.get("id_original", node_data.get("id"))
            node_x_positions[node_id] = node["position"]["x"]
            node_y_positions[node_id] = node["position"]["y"]

        # Process edges and add them to the graph
        for edge in cyjs_data["elements"]["edges"]:
            edge_data = edge["data"]
            # Use the original source and target labels if available, otherwise fall back to default labels
            source = edge_data.get(f"{source_label}_original", edge_data.get(source_label))
            target = edge_data.get(f"{target_label}_original", edge_data.get(target_label))
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
        G = nx.convert_node_labels_to_integers(G)
        return G

    def _remove_invalid_graph_properties(self, G: nx.Graph) -> None:
        """Remove invalid properties from the graph, including self-loops, nodes with fewer edges than
        the threshold, and isolated nodes.

        Args:
            G (nx.Graph): A NetworkX graph object.
        """
        # Count the number of nodes and edges before cleaning
        num_initial_nodes = G.number_of_nodes()
        num_initial_edges = G.number_of_edges()
        # Remove self-loops to ensure correct edge count
        G.remove_edges_from(nx.selfloop_edges(G))
        # Iteratively remove nodes with fewer edges than the threshold
        while True:
            nodes_to_remove = [
                node
                for node, degree in dict(G.degree()).items()
                if degree < self.min_edges_per_node
            ]
            if not nodes_to_remove:
                break  # Exit loop if no nodes meet the condition
            G.remove_nodes_from(nodes_to_remove)

        # Remove isolated nodes
        G.remove_nodes_from(nx.isolates(G))

        # Log the number of nodes and edges before and after cleaning
        num_final_nodes = G.number_of_nodes()
        num_final_edges = G.number_of_edges()
        logger.debug(f"Initial node count: {num_initial_nodes}")
        logger.debug(f"Final node count: {num_final_nodes}")
        logger.debug(f"Initial edge count: {num_initial_edges}")
        logger.debug(f"Final edge count: {num_final_edges}")

    def _assign_edge_weights(self, G: nx.Graph) -> None:
        """Assign default edge weights to the graph.

        Args:
            G (nx.Graph): A NetworkX graph object.
        """
        # Set default weight for all edges in bulk
        default_weight = 1
        nx.set_edge_attributes(G, default_weight, "weight")

    def _validate_nodes(self, G: nx.Graph) -> None:
        """Validate the graph structure and attributes with attribute fallback for positions and labels.

        Args:
            G (nx.Graph): A NetworkX graph object.

        Raises:
            ValueError: If a node is missing 'x', 'y', and a valid 'pos' attribute.
        """
        # Retrieve all relevant attributes in bulk
        pos_attrs = nx.get_node_attributes(G, "pos")
        name_attrs = nx.get_node_attributes(G, "name")
        id_attrs = nx.get_node_attributes(G, "id")
        # Dictionaries to hold missing or fallback attributes
        x_attrs = {}
        y_attrs = {}
        label_attrs = {}
        nodes_with_missing_labels = []

        # Iterate through nodes to validate and assign missing attributes
        for node in G.nodes:
            attrs = G.nodes[node]
            # Validate and assign 'x' and 'y' attributes
            if "x" not in attrs or "y" not in attrs:
                if (
                    node in pos_attrs
                    and isinstance(pos_attrs[node], (list, tuple, np.ndarray))
                    and len(pos_attrs[node]) >= 2
                ):
                    x_attrs[node], y_attrs[node] = pos_attrs[node][:2]
                else:
                    raise ValueError(
                        f"Node {node} is missing 'x', 'y', and a valid 'pos' attribute."
                    )

            # Validate and assign 'label' attribute
            if "label" not in attrs:
                if node in name_attrs:
                    label_attrs[node] = name_attrs[node]
                elif node in id_attrs:
                    label_attrs[node] = id_attrs[node]
                else:
                    # Assign node ID as label and log the missing label
                    label_attrs[node] = str(node)
                    nodes_with_missing_labels.append(node)

        # Batch update attributes in the graph
        nx.set_node_attributes(G, x_attrs, "x")
        nx.set_node_attributes(G, y_attrs, "y")
        nx.set_node_attributes(G, label_attrs, "label")

        # Log a warning if any labels were missing
        if nodes_with_missing_labels:
            total_nodes = G.number_of_nodes()
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
        logger.debug(f"Minimum edges per node: {self.min_edges_per_node}")
        logger.debug(f"Projection: {'Sphere' if self.compute_sphere else 'Plane'}")
        if self.compute_sphere:
            logger.debug(f"Surface depth: {self.surface_depth}")
