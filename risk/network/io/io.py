"""
risk/network/io/io
~~~~~~~~~~~~~~~~~~

This file contains the code for the RISK class and command-line access.
"""

from pathlib import Path
import shutil
import zipfile

import networkx as nx
import pandas as pd
from xml.dom import minidom

from .tidy import remove_invalid_graph_properties


def load_networkx_network(
    G,
    min_edges_per_node=0,
):
    # Remove invalid graph attributes / properties as soon as edges are added
    remove_invalid_graph_properties(G, min_edges_per_node=min_edges_per_node)
    # Relabel the node ids to sequential numbers to make calculations faster
    G = nx.relabel_nodes(G, {node: idx for idx, node in enumerate(G.nodes)})
    return G


def load_cys_network(
    cys_filepath,
    source_node_label,
    target_node_label,
    edge_weight_label,
    view_name=None,
    min_edges_per_node=0,
):
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
    attribute_table = attribute_table[[source_node_label, target_node_label, edge_weight_label]]
    attribute_table = attribute_table.dropna().reset_index(drop=True)

    # Create a graph
    G = nx.Graph()
    # Add edges and nodes with weights
    for _, row in attribute_table.iterrows():
        source, target, weight = (
            row[source_node_label],
            row[target_node_label],
            float(row[edge_weight_label]),
        )
        if source not in G:
            G.add_node(source)  # Optionally add x, y coordinates here if available
        if target not in G:
            G.add_node(target)  # Optionally add x, y coordinates here if available
        G.add_edge(source, target, weight=weight)

    # Remove invalid graph attributes / properties as soon as edges are added
    remove_invalid_graph_properties(G, min_edges_per_node=min_edges_per_node)

    for node in G.nodes():
        G.nodes[node]["label"] = node
        G.nodes[node]["x"] = node_xs[node]  # Assuming you have a dict `node_xs` for x coordinates
        G.nodes[node]["y"] = node_ys[node]  # Assuming you have a dict `node_ys` for y coordinates

    # Relabel the node ids to sequential numbers to make calculations faster
    G = nx.relabel_nodes(G, {node: idx for idx, node in enumerate(G.nodes)})
    # Remove unzipped files/directories
    cys_dirnames = list(set([cf.split("/")[0] for cf in cys_files]))
    for dirname in cys_dirnames:
        shutil.rmtree(dirname)

    return G
