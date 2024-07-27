"""
risk/network/io/network
~~~~~~~~~~~~~~~~~~~~~~~

This file contains the code for the RISK class and command-line access.
"""

import os
import pickle
import shutil
import sys
import warnings
import zipfile
from contextlib import contextmanager
from xml.dom import minidom

import networkx as nx
import numpy as np
import pandas as pd
from rich import print
from rich.progress import Progress
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist

from risk.network.graph import calculate_edge_lengths
from risk.network.neighborhoods import get_network_neighborhoods


class NetworkIO:
    def __init__(
        self,
        compute_sphere=True,
        dimple_factor=None,
        include_edge_weight=True,
        distance_metric="shortpath",
        neighborhood_diameter=0.5,
        louvain_resolution=0.1,
        min_edges_per_node=0,
    ):
        self.compute_sphere = compute_sphere
        self.dimple_factor = dimple_factor
        self.include_edge_weight = include_edge_weight
        self.distance_metric = distance_metric
        self.neighborhood_diameter = neighborhood_diameter
        self.louvain_resolution = louvain_resolution
        self.min_edges_per_node = min_edges_per_node

    def load_gpickle(self, filepath):
        """Load a network from a GPickle file.

        Args:
            filepath (str): Path to the GPickle file.

        Returns:
            NetworkX graph: Loaded network.
        """
        self._log_network_loading("GPickle", filepath)
        with open(filepath, "rb") as f:
            G = pickle.load(f)
        return self._load_networkx(G)

    def load_networkx(self, G):
        """Load a NetworkX graph.

        Args:
            G (NetworkX graph): A NetworkX graph object.

        Returns:
            NetworkX graph: Processed network.
        """
        self._log_network_loading("NetworkX")
        return self._load_networkx(G)

    def load_cytoscape(
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
        self._log_network_loading("Cytoscape", filepath)
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
        """Prepare network by adjusting dimple factor and get neighborhoods.

        Args:
            G (NetworkX graph): The input network.

        Returns:
            Tuple: Processed network and neighborhoods.
        """
        if self.dimple_factor is None and self.compute_sphere:
            self.dimple_factor = self._get_best_dimple_factor(
                G,
                include_edge_weight=self.include_edge_weight,
                compute_sphere=self.compute_sphere,
                distance_metric=self.distance_metric,
                neighborhood_diameter=self.neighborhood_diameter,
                louvain_resolution=self.louvain_resolution,
            )

        G = calculate_edge_lengths(
            G,
            include_edge_weight=self.include_edge_weight,
            compute_sphere=self.compute_sphere,
            dimple_factor=self.dimple_factor,
        )
        return G

    def _get_best_dimple_factor(
        self,
        G,
        include_edge_weight=True,
        compute_sphere=True,
        distance_metric="euclidean",
        neighborhood_diameter=0.5,
        louvain_resolution=None,
        lower_bound=0,
        upper_bound=1024,
        tolerance=4,
    ):
        """Find the optimal dimple factor for the network.

        Args:
            G (NetworkX graph): The network graph.
            lower_bound (int): Lower bound for dimple factor.
            upper_bound (int): Upper bound for dimple factor.
            tolerance (int): Tolerance for dimple factor optimization.

        Returns:
            int: The best dimple factor.
        """
        print(
            "[cyan][red]Warning:[/red] [blue]Optimizing[/blue] [yellow]dimple factor[/yellow] can be an [red]expensive process[/red]. "
            "[blue]Mark down[/blue] [yellow]optimal dimple factor[/yellow] for future use...[/cyan]"
        )

        # Initialize variables to keep track of the best score and corresponding dimple factor
        max_score = -np.inf
        best_dimple_factor = lower_bound
        # Calculate the total number of iterations for progress tracking
        total_iterations = int(np.ceil(np.log2((upper_bound - lower_bound) / tolerance))) + 1
        # Start the progress tracking
        with Progress() as progress:
            task_id = progress.add_task(
                "[cyan]Optimizing [yellow]dimple factor[/yellow]...[/cyan]", total=total_iterations
            )
            current_iteration = 0

            while upper_bound - lower_bound > tolerance:
                # Compute the midpoint of the current search interval
                mid_dimple_factor = (lower_bound + upper_bound) / 2
                # Generate dimple factors to test (midpoint and midpoint + tolerance)
                dimple_factors_to_test = [mid_dimple_factor, mid_dimple_factor + tolerance]

                for dimple_factor in map(int, dimple_factors_to_test):
                    # Calculate edge lengths with the current dimple factor
                    G_test = calculate_edge_lengths(
                        G,
                        include_edge_weight=include_edge_weight,
                        compute_sphere=compute_sphere,
                        dimple_factor=dimple_factor,
                    )

                    # Suppress print output for loading neighborhoods
                    with self._suppress_print():
                        neighborhoods_test = get_network_neighborhoods(
                            network=G_test,
                            distance_metric=distance_metric,
                            neighborhood_diameter=neighborhood_diameter,
                            compute_sphere=compute_sphere,
                            louvain_resolution=louvain_resolution,
                        )

                    # Compute the silhouette score for the test graph
                    score_test = self._compute_silhouette_score(neighborhoods_test)

                    # Update the best score and dimple factor if the current score is better
                    if score_test > max_score:
                        max_score = score_test
                        best_dimple_factor = dimple_factor

                # Adjust the search interval based on the test results
                if best_dimple_factor == mid_dimple_factor + tolerance:
                    lower_bound = mid_dimple_factor
                else:
                    upper_bound = mid_dimple_factor

                # Update the progress tracker
                current_iteration += 1
                progress.update(
                    task_id,
                    advance=1,
                    description="[cyan]Optimizing [yellow]dimple factor[/yellow]...[/cyan]",
                )

        # Print the optimal dimple factor
        print(f"[yellow]Optimal dimple factor:[/yellow] [red]{best_dimple_factor}[/red]")
        return best_dimple_factor

    def _compute_silhouette_score(self, neighborhoods, linkage_method="average"):
        """Compute the silhouette score for a given graph and neighborhoods.

        Args:
            neighborhoods (numpy.ndarray): Neighborhood matrix.
            linkage_method (str): The linkage method to use for clustering.

        Returns:
            float: The silhouette score.
        """
        # Calculate the maximum value in the neighborhoods matrix
        max_value = np.max(neighborhoods)
        # Compute the distance matrix by subtracting neighborhoods from the max value
        distance_matrix = max_value - neighborhoods
        # Ensure the diagonal elements are zero (distance to itself is zero)
        np.fill_diagonal(distance_matrix, 0)
        # Ensure all distance values are non-negative
        distance_matrix = np.maximum(distance_matrix, 0)
        # Compute the silhouette score using hierarchical clustering
        return self._compute_silhouette_with_validation(distance_matrix, linkage_method)

    @staticmethod
    def _compute_silhouette_with_validation(distance_matrix, linkage_method="average"):
        """Compute the silhouette score for hierarchical clustering.

        Args:
            distance_matrix (np.ndarray): The distance matrix.
            linkage_method (str): The linkage method to use for hierarchical clustering.

        Returns:
            float: The best silhouette score found.
        """
        # Ensure all values in the distance matrix are non-negative
        distance_matrix = np.maximum(distance_matrix, 0)

        # Perform hierarchical clustering
        Z = linkage(pdist(distance_matrix), method=linkage_method)

        # Initialize variables to keep track of the best score
        best_score = float("-inf")
        num_clusters = 2

        while num_clusters < len(distance_matrix):
            # Generate cluster labels for the current number of clusters
            labels = fcluster(Z, t=num_clusters, criterion="maxclust")

            # Ensure there is more than one cluster
            if len(set(labels)) > 1:
                try:
                    # Compute the silhouette score for the current clustering
                    score = silhouette_score(distance_matrix, labels, metric="precomputed")
                    # Update the best score if the current score is better
                    if score > best_score:
                        best_score = score
                    break
                except ValueError:
                    pass  # Continue to the next iteration if an error occurs

            num_clusters += 1

        # Assign a default score if no valid clustering is found
        if best_score == float("-inf"):
            best_score = 0.0
            print("Valid clustering could not be achieved. Returning default score of 0.0.")

        return best_score

    @staticmethod
    def _log_network_loading(filetype, filepath=None):
        """Log information about the network file being loaded.

        Args:
            filetype (str): The type of the file being loaded (e.g., 'Cytoscape').
            filepath (str, optional): The path to the file being loaded.
        """
        if filepath:
            message = f"[cyan]Loading [yellow]{filetype}[/yellow] [blue]network file[/blue]: [yellow]'{filepath}'[/yellow]...[/cyan]"
        else:
            message = (
                f"[cyan]Loading [yellow]{filetype}[/yellow] [blue]network file[/blue]...[/cyan]"
            )
        print(message)

    @staticmethod
    @contextmanager
    def _suppress_print():
        """Context manager to suppress print statements."""
        original_stdout = sys.stdout
        try:
            with open(os.devnull, "w") as devnull:
                sys.stdout = devnull
                yield
        finally:
            sys.stdout = original_stdout
