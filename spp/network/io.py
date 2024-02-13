#! /usr/bin/env python
"""This file contains the code for the SAFE class and command-line access."""
import contextlib
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import zipfile
import shutil

from scipy.spatial.distance import pdist, squareform
from xml.dom import minidom


def load_cys_network(cys_filepath, view_name=None):
    # Unzip CYS file
    cys_filepath = Path(str(cys_filepath))
    with zipfile.ZipFile(cys_filepath, "r") as zip_ref:
        cys_files = zip_ref.namelist()
        zip_ref.extractall("./")
    # Get first view and network instances
    cys_view_files = [cf for cf in cys_files if "/views/" in cf]
    cys_view_file = (
        cys_view_files[0]
        if not view_name
        else [cvf for cvf in cys_view_files if cvf.endswith(view_name + ".xgmml")][0]
    )
    cys_network_file = [cf for cf in cys_files if "/networks/" in cf][0]
    # Parse edges
    cys_network_dom = minidom.parse(cys_network_file)
    cys_edges = cys_network_dom.getElementsByTagName("edge")
    edge_list = []
    for edge in cys_edges:
        # Only add edges if both 'source' and 'target' keys exist in the edge attribute
        with contextlib.suppress(KeyError):
            edge_list.append(
                (int(edge.attributes["source"].value), int(edge.attributes["target"].value))
            )

    # Parse nodes
    cys_view_dom = minidom.parse(cys_view_file)
    cys_nodes = cys_view_dom.getElementsByTagName("node")
    node_labels = {}
    node_xs = {}
    node_ys = {}
    for node in cys_nodes:
        # Node ID is SUID
        node_id = int(node.attributes["cy:nodeId"].value)
        node_labels[node_id] = node.attributes["label"].value
        for child in node.childNodes:
            if child.nodeType == 1 and child.tagName == "graphics":
                node_xs[node_id] = float(child.attributes["x"].value)
                node_ys[node_id] = float(child.attributes["y"].value)

    # Build graph
    G = nx.Graph()
    G.add_edges_from(edge_list)
    # Modify graph to add additional attributes and remove nodes not found in nodelist but in edgelist
    for node in G.nodes:
        try:
            G.nodes[node]["label"] = node_labels[node]
            G.nodes[node]["x"] = node_xs[node]
            G.nodes[node]["y"] = node_ys[node]
        # Node not found in G network - remove node from network
        except KeyError:
            G.remove_node(node)

    # Read the node attributes (from /tables/)
    attribute_metadata_keywords = ["/tables/", "SHARED_ATTRS", "node.cytable"]
    attribute_metadata = [
        cf for cf in cys_files if all(keyword in cf for keyword in attribute_metadata_keywords)
    ][0]
    # Load attributes file from Cytoscape as pandas data frame
    attribute_table = pd.read_csv(attribute_metadata, sep=",", header=None, skiprows=1)
    # Set columns
    attribute_table.columns = attribute_table.iloc[0]
    # Skip first four rows
    attribute_table = attribute_table.iloc[4:, :].reset_index(drop=True)
    # Maps SUID to the rest of the column headers and values
    attributes_map = {attr.pop("SUID"): attr for attr in attribute_table.T.to_dict().values()}
    # Add attributes to graph
    for node_id, attributes in attributes_map.items():
        if node_id in G.nodes:
            for attr_header, attr_value in attributes.items():
                G.nodes[node_id][attr_header] = attr_value

    # Relabel the node ids to sequential numbers to make calculations faster
    G = nx.relabel_nodes(G, {node: idx for idx, node in enumerate(G.nodes)})
    G = calculate_edge_lengths(G)
    # Remove unzipped files/directories
    cys_dirnames = list(set([cf.split("/")[0] for cf in cys_files]))
    for dirname in cys_dirnames:
        shutil.rmtree(dirname)

    return G


def calculate_edge_lengths(G):
    # Extract node coordinates
    nodes_data = dict(G.nodes(data=True))
    x = np.array([data["x"] for _, data in nodes_data.items()])
    y = np.array([data["y"] for _, data in nodes_data.items()])
    # Calculate node distances
    node_coordinates = np.column_stack((x, y))
    node_distances = squareform(pdist(node_coordinates, "euclidean"))
    # Create an adjacency matrix with NaN for non-adjacent nodes
    adjacency_matrix = np.array(nx.adjacency_matrix(G).todense(), dtype=float)
    adjacency_matrix[adjacency_matrix == 0] = np.nan
    # Multiply node distances by adjacency matrix to get edge lengths
    edge_lengths = node_distances * adjacency_matrix
    # Update edge attributes in the graph
    edge_attr_dict = {
        (i, j): length
        for i, row in enumerate(edge_lengths)
        for j, length in enumerate(row)
        if ~np.isnan(length)
    }
    nx.set_edge_attributes(G, edge_attr_dict, "length")

    return G


def load_network_annotation(network, annotation_filepath, node_colname="label"):
    # Convert JSON data to a Python dictionary
    with open(annotation_filepath, "r") as file:
        annotation_input = json.load(file)
    # Flatten the dictionary for easier DataFrame creation
    flattened_annotation = [
        (node, annotation) for annotation, nodes in annotation_input.items() for node in nodes
    ]
    # Create a DataFrame
    annotation = pd.DataFrame(flattened_annotation, columns=["Node", "Annotation"])
    annotation["Is Member"] = 1
    annotation_pivot = annotation.pivot_table(
        index="Node", columns="Annotation", values="Is Member", fill_value=0, dropna=False
    )
    # Get list of node labels as ordered in a graph object
    node_label_order = list(nx.get_node_attributes(network, node_colname).values())
    # NOTE: This sets the node annotation background (union of all annotation nodes) to nodes as found in the network
    annotation_pivot = annotation_pivot.reindex(index=node_label_order)
    annotation_pivot[np.isnan(annotation_pivot)] = 0
    # Pivot the DataFrame to achieve the desired format
    ordered_nodes = tuple(annotation_pivot.index)
    ordered_annotations = tuple(annotation_pivot.columns)
    return {
        "ordered_row_nodes": ordered_nodes,
        "ordered_column_annotations": ordered_annotations,
        "annotation_matrix": annotation_pivot.to_numpy(),
    }
