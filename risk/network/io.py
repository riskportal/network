#! /usr/bin/env python
"""This file contains the code for the RISK class and command-line access."""
import json
from pathlib import Path

import networkx as nx
import pandas as pd
import zipfile
import shutil

from xml.dom import minidom


def load_cys_network(
    cys_filepath,
    source_node_label,
    target_node_label,
    edge_weight_label,
    view_name=None,
    compute_sphere=False,
    dimple_factor=0.0,
    min_edges_per_node=0,
    include_edge_weight=False,
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


def remove_invalid_graph_properties(G, min_edges_per_node=0):
    # Remove nodes with `min_edges_per_node` or fewer edges
    nodes_with_few_edges = [node for node in G.nodes() if G.degree(node) <= min_edges_per_node]
    G.remove_nodes_from(nodes_with_few_edges)
    self_loops = list(nx.selfloop_edges(G))
    G.remove_edges_from(self_loops)


def load_network_annotation(network, annotation_filepath):
    # Convert JSON data to a Python dictionary
    with open(annotation_filepath, "r") as file:
        annotation_input = json.load(file)
    # Flatten the dictionary for easier DataFrame creation
    flattened_annotation = [
        (node, annotation) for annotation, nodes in annotation_input.items() for node in nodes
    ]
    # Create a DataFrame
    annotation = pd.DataFrame(flattened_annotation, columns=["Node", "Annotation"])
    # annotation_to_node_id_map = dict(zip(annotation['Node'], annotation.index))
    # add_edges_within_annotations(network, annotation_input, annotation_to_node_id_map)
    annotation["Is Member"] = 1
    # Pivot the DataFrame to achieve the desired format
    annotation_pivot = annotation.pivot_table(
        index="Node", columns="Annotation", values="Is Member", fill_value=0, dropna=False
    )
    # Get list of node labels as ordered in a graph object
    node_label_order = list(nx.get_node_attributes(network, "label").values())
    # This will reindex the annotation matrix with node labels as found in annotation file - those that are not found,
    # (i.e., rows) will be set to NaN values
    annotation_pivot = annotation_pivot.reindex(index=node_label_order)
    ordered_nodes = tuple(annotation_pivot.index)
    ordered_annotations = tuple(annotation_pivot.columns)
    return {
        "ordered_row_nodes": ordered_nodes,
        "ordered_column_annotations": ordered_annotations,
        "annotation_matrix": annotation_pivot.to_numpy(),
    }


# def add_edges_within_annotations(G, annotation_input, annotation_to_node_id_map):
#     for _, values in annotation_input.items():
#         # Generate all unique pairs of values to add as edges
#         for i in range(len(values)):
#             for j in range(i + 1, len(values)):
#                 node_id_1 = annotation_to_node_id_map.get(values[i])
#                 node_id_2 = annotation_to_node_id_map.get(values[j])

#                 # Check if both nodes exist in the graph before adding the edge
#                 if node_id_1 and node_id_2 and G.has_node(node_id_1) and G.has_node(node_id_2):
#                     G.add_edge(node_id_1, node_id_2, weight=0)
