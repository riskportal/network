#! /usr/bin/env python
"""This file contains the code for the SAFE class and command-line access."""
import json
from pathlib import Path

import networkx as nx
import numpy as np
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

    for node in G.nodes():
        G.nodes[node]["label"] = node
        G.nodes[node]["x"] = node_xs[node]  # Assuming you have a dict `node_xs` for x coordinates
        G.nodes[node]["y"] = node_ys[node]  # Assuming you have a dict `node_ys` for y coordinates

    # Relabel the node ids to sequential numbers to make calculations faster
    G = nx.relabel_nodes(G, {node: idx for idx, node in enumerate(G.nodes)})
    G = calculate_edge_lengths(
        G, compute_sphere=compute_sphere, include_edge_weight=include_edge_weight
    )
    # Remove unzipped files/directories
    cys_dirnames = list(set([cf.split("/")[0] for cf in cys_files]))
    for dirname in cys_dirnames:
        shutil.rmtree(dirname)

    return G


def map_to_sphere(G):
    # Extract x, y coordinates from the graph nodes
    xy_coords = np.array([[G.nodes[node]["x"], G.nodes[node]["y"]] for node in G.nodes()])

    # Normalize the coordinates between [0, 1]
    min_vals = np.min(xy_coords, axis=0)
    max_vals = np.max(xy_coords, axis=0)
    normalized_xy = (xy_coords - min_vals) / (max_vals - min_vals)

    # Map normalized coordinates to theta and phi on a sphere
    theta = normalized_xy[:, 0] * np.pi * 2
    phi = normalized_xy[:, 1] * np.pi

    # Convert spherical coordinates to Cartesian coordinates for 3D sphere
    for i, node in enumerate(G.nodes()):
        x = np.sin(phi[i]) * np.cos(theta[i])
        y = np.sin(phi[i]) * np.sin(theta[i])
        z = np.cos(phi[i])
        G.nodes[node]["x"] = x
        G.nodes[node]["y"] = y
        G.nodes[node]["z"] = z


def normalize_graph_coordinates(G):
    # Extract x, y coordinates from the graph nodes
    xy_coords = np.array([[G.nodes[node]["x"], G.nodes[node]["y"]] for node in G.nodes()])

    # Calculate min and max values for x and y
    min_vals = np.min(xy_coords, axis=0)
    max_vals = np.max(xy_coords, axis=0)

    # Normalize the coordinates to [0, 1]
    normalized_xy = (xy_coords - min_vals) / (max_vals - min_vals)

    # Update the node coordinates with the normalized values
    for i, node in enumerate(G.nodes()):
        G.nodes[node]["x"], G.nodes[node]["y"] = normalized_xy[i]


def normalize_weights(G):
    weights = [data["weight"] for _, _, data in G.edges(data=True) if "weight" in data]
    if weights:  # Ensure there are weighted edges
        min_weight = min(weights)
        max_weight = max(weights)
        range_weight = max_weight - min_weight if max_weight > min_weight else 1
        for u, v, data in G.edges(data=True):
            if "weight" in data:
                data["normalized_weight"] = (data["weight"] - min_weight) / range_weight


def spherical_distance(u_coords, v_coords):
    distance = np.arccos(np.dot(u_coords, v_coords))
    return distance


def calculate_edge_lengths(G, include_edge_weight=False, compute_sphere=True):
    # Normalize graph coordinates
    normalize_graph_coordinates(G)
    # Normalize weights
    normalize_weights(G)
    # Conditionally map nodes to a sphere based on `compute_sphere`
    if compute_sphere:
        map_to_sphere(G)

    for u, v, edge_data in G.edges(data=True):
        if compute_sphere:
            u_coords = np.array([G.nodes[u]["x"], G.nodes[u]["y"], G.nodes[u].get("z", 0)])
            v_coords = np.array([G.nodes[v]["x"], G.nodes[v]["y"], G.nodes[v].get("z", 0)])
            # Calculate the spherical distance
            dist = np.arccos(np.clip(np.dot(u_coords, v_coords), -1.0, 1.0))
        else:
            # If not computing sphere, use only x, y for planar distance
            u_coords = np.array([G.nodes[u]["x"], G.nodes[u]["y"]])
            v_coords = np.array([G.nodes[v]["x"], G.nodes[v]["y"]])
            # Calculate the planar distance
            dist = np.linalg.norm(u_coords - v_coords)

        if include_edge_weight and "normalized_weight" in edge_data:
            # Invert the weight influence such that higher weights bring nodes closer
            G.edges[u, v]["length"] = dist / (
                edge_data["normalized_weight"] + 10e-12  # Avoid division by zero
            )
        else:
            G.edges[u, v]["length"] = dist  # Use calculated distance directly

    return G


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
