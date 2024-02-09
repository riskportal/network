#! /usr/bin/env python
"""This file contains the code for the SAFE class and command-line access."""
import contextlib
import re
import os
from pathlib import Path
import logging

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.io as spio
import pandas as pd
import zipfile
import random
import shutil
import ast

from os.path import expanduser
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist, squareform
from scipy.stats import gaussian_kde
from scipy.optimize import fmin
from collections import Counter
from xml.dom import minidom


def load_cys_network(cys_filepath, view_name=None):
    # Unzip CYS file
    cys_files = zipfile.ZipFile(cys_filepath, "r").namelist()
    # Get first view and network instances
    cys_view_files = [cf for cf in cys_files if "/views/" in cf]
    cys_view_file = (
        cys_view_files[0]
        if not view_name
        else [cvf for cvf in cys_view_files if cvf.endswith(view_name + ".xgmml")][0]
    )
    cys_network_file = [cf for cf in cys_files if "/networks/" in cf][0]

    # Parse nodes
    cys_view_dom = minidom.parse(cys_view_file)
    cys_nodes = cys_view_dom.getElementsByTagName("node")
    node_labels = dict()
    node_xs = dict()
    node_ys = dict()
    for node in cys_nodes:
        node_id = int(node.attributes["cy:nodeId"].value)
        node_labels[node_id] = node.attributes["label"].value
        for child in node.childNodes:
            if child.nodeType == 1 and child.tagName == "graphics":
                node_xs[node_id] = float(child.attributes["x"].value)
                node_ys[node_id] = float(child.attributes["y"].value)

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

    # Build graph
    G = nx.Graph()
    G.add_edges_from(edge_list)
    # Modify graph to add additional attributes and remove nodes not found in nodelist but in edgelist
    for node in G.nodes:
        if node in node_labels.keys():
            G.nodes[node]["label"] = node_labels[node]
            G.nodes[node]["x"] = node_xs[node]
            G.nodes[node]["y"] = node_ys[node]
        else:
            G.remove_node(node)

    # Read the node attributes (from /tables/)
    network_filename = cys_filepath.name
    attribute_meta_keywords = ["/tables/", network_filename, "SHARED_ATTRS", "node.cytable"]
    attribute_metadata = [
        cf for cf in cys_files if all(keyword in cf for keyword in attribute_meta_keywords)
    ][0]
    attribute_table = pd.read_csv(attribute_metadata, sep=",", header=None, skiprows=1)
    col_headers = []
    for row_idx in np.arange(7):
        attribute_row = attribute_table.iloc[row_idx]
        if attribute_row.iloc[0] == "SUID":
            col_headers = list(attribute_row)
            break

    # Add attributes to nodes
    attribute_table = attribute_table.iloc[row_idx, :]
    attribute_table.columns = col_headers
    attribute_table["SUID"] = attribute_table["SUID"].astype(int)
    # Then, if SSUID is present, add all of its row contents to appropriate column keys to the graph
    for row in attribute_table.iterrows():
        if row["SUID"] in G.nodes:
            for c in col_headers[1:]:
                G.nodes[row["SUID"]][c] = row[c]

    # Relabel the node ids to sequential numbers to make calculations faster
    mapping = dict()
    for idx, node in enumerate(G.nodes):
        mapping[node] = idx
    G = nx.relabel_nodes(G, mapping)
    G = calculate_edge_lengths(G, verbose=verbose)
    # Remove unzipped files/directories
    cys_dirnames = list(set([cf.split("/")[0] for cf in cys_files]))
    for dirname in cys_dirnames:
        shutil.rmtree(dirname)

    return G


def apply_network_layout(G, layout="kamada_kawai", verbose=True):
    if layout == "kamada_kawai":
        if verbose:
            logging.info("Applying the Kamada-Kawai network layout... (may take several minutes)")

        pos = nx.kamada_kawai_layout(G)

    elif layout == "spring_embedded":
        if verbose:
            logging.info(
                "Applying the spring-embedded network layout... (may take several minutes)"
            )

        pos = nx.spring_layout(G, k=0.2, iterations=100)

    for n in G:
        G.nodes[n]["x"] = pos[n][0]
        G.nodes[n]["y"] = pos[n][1]

    return G


def calculate_edge_lengths(G, verbose=True):
    # Calculate the lengths of the edges

    if verbose:
        logging.info("Calculating edge lengths...")

    x = np.matrix(G.nodes.data("x"))[:, 1]
    y = np.matrix(G.nodes.data("y"))[:, 1]

    node_coordinates = np.concatenate([x, y], axis=1)
    node_distances = squareform(pdist(node_coordinates, "euclidean"))

    adjacency_matrix = np.array(nx.adjacency_matrix(G).todense())
    adjacency_matrix = adjacency_matrix.astype("float")
    adjacency_matrix[adjacency_matrix == 0] = np.nan

    edge_lengths = np.multiply(node_distances, adjacency_matrix)

    edge_attr_dict = {index: v for index, v in np.ndenumerate(edge_lengths) if ~np.isnan(v)}
    nx.set_edge_attributes(G, edge_attr_dict, "length")

    return G


def load_network_attributes(network, annotation_filepath, annotation_key_colname):
    # Load node mapping to attributes
    node_to_attributes = pd.read_csv(annotation_filepath, sep="\t", dtype={0: str})
    node_to_attributes.set_index(node_to_attributes.columns[0], drop=True, inplace=True)
    # Force all values to numeric
    node_to_attributes = node_to_attributes.apply(pd.to_numeric, downcast="float", errors="coerce")
    node_to_attributes = node_to_attributes.apply(pd.to_numeric, errors="coerce")
    node_to_attributes.columns = np.arange(len(node_to_attributes.columns))
    # Averaging duplicate rows
    if not node_to_attributes.index.is_unique:
        print("Duplicate nodes found. Removing duplicates...")
        node_to_attributes = node_to_attributes.groupby(node_to_attributes.index, axis=0).mean()
    # Get node label order...
    node_label_order = (
        list(nx.get_node_attributes(network, annotation_key_colname).values())
        or node_to_attributes.index.values
    )
    node_to_attributes = node_to_attributes.reindex(index=node_label_order, fill_value=fill_value)

    # Load attributes
    attributes = pd.DataFrame(
        {"id": np.arange(len(node_to_attributes.columns)), "name": node_to_attributes.columns}
    )
    # Force attribute names to be strings
    attributes["name"] = attributes["name"].astype(str)

    return node_to_attributes, node_label_order, attributes


def plot_network(
    G,
    ax=None,
    foreground_color="#ffffff",
    background_color="#000000",
    random_sampling_edges_min=30000,
    title="Network",
):
    """
    Plot/draw a network.

    Note:
        The default attribute names
        gene ids: label_orf
        gene symbols: label
    """
    if background_color == "#ffffff":
        foreground_color = "#000000"

    node_xy = get_node_coordinates(G)

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(20, 10), facecolor=background_color, edgecolor=foreground_color
        )
        fig.set_facecolor(background_color)

    # Randomly sample a fraction of the edges (when network is too big)
    edges = tuple(G.edges())
    if len(edges) >= random_sampling_edges_min:
        logging.warning(
            f"Edges are randomly sampled because the network (edges={len(edges)}) is too big (random_sampling_edges_min={random_sampling_edges_min})."
        )
        edges = random.sample(edges, int(len(edges) * 0.1))

    nx.draw(
        G,
        ax=ax,
        pos=node_xy,
        edgelist=edges,
        node_color=foreground_color,
        edge_color=foreground_color,
        node_size=10,
        width=1,
        alpha=0.2,
    )

    ax.set_aspect("equal")
    ax.set_facecolor(background_color)

    ax.grid(False)
    ax.invert_yaxis()
    ax.margins(0.1, 0.1)

    ax.set_title(title, color=foreground_color)

    plt.axis("off")

    try:
        fig.set_facecolor(background_color)
    except NameError:
        pass

    return ax


def plot_network_contour(graph, ax, background_color="#000000"):
    foreground_color = "#ffffff"
    if background_color == "#ffffff":
        foreground_color = "#000000"

    x = dict(graph.nodes.data("x"))
    y = dict(graph.nodes.data("y"))

    ds = [x, y]
    pos = {}
    for k in x:
        pos[k] = np.array([d[k] for d in ds])

    # Compute the convex hull to delineate the network
    hull = ConvexHull(np.vstack(list(pos.values())))

    vertices_x = [pos.get(v)[0] for v in hull.vertices]
    vertices_y = [pos.get(v)[1] for v in hull.vertices]

    vertices_x = np.array(vertices_x)
    vertices_y = np.array(vertices_y)

    # Find center of mass and radius to approximate the hull with a circle
    xm = np.nanmean(vertices_x)
    ym = np.nanmean(vertices_y)

    rm = np.nanmean(np.sqrt((vertices_x - xm) ** 2 + (vertices_y - ym) ** 2))

    # Best fit a circle to these points
    def err(x0):
        [w, v, r] = x0
        pts = [np.linalg.norm([x - w, y - v]) - r for x, y in zip(vertices_x, vertices_y)]
        return (np.array(pts) ** 2).sum()

    [xf, yf, rf] = fmin(err, [xm, ym, rm], disp=False)

    circ = plt.Circle((xf, yf), radius=rf * 1.01, color=foreground_color, linewidth=1, fill=False)
    ax.add_patch(circ)

    return xf, yf, rf


def plot_costanzo2016_network_annotations(
    graph,
    ax,
    path_to_data,
    colors=True,
    clabels=False,
    foreground_color="#ffffff",
    background_color="#000000",
):
    if background_color == "#ffffff":
        foreground_color = "#000000"

    path_to_network_annotations = (
        "other/Data File S5_SAFE analysis_Gene cluster identity and functional enrichments.xlsx"
    )
    filename = os.path.join(path_to_data, path_to_network_annotations)

    costanzo2016 = pd.read_excel(filename, sheet_name="Global net. cluster gene list")
    processes = costanzo2016["Global Similarity Network Region name"].unique()
    processes = processes[pd.notnull(processes)]

    process_colors = pd.read_csv(
        os.path.join(path_to_data, "other/costanzo_2016_colors.txt"), sep="\t"
    )
    if colors:
        process_colors = process_colors[["R", "G", "B"]].values / 256
    else:
        if foreground_color == "#ffffff":
            process_colors = np.ones((process_colors.shape[0], 3))
        else:
            process_colors = np.zeros((process_colors.shape[0], 3))

    labels = nx.get_node_attributes(graph, "label")
    labels_dict = {k: v for v, k in labels.items()}

    x = list(dict(graph.nodes.data("x")).values())
    y = list(dict(graph.nodes.data("y")).values())

    pos = {}
    for idx, k in enumerate(x):
        pos[idx] = np.array([x[idx], y[idx]])

    for n_process, process in enumerate(processes):
        nodes = costanzo2016.loc[
            costanzo2016["Global Similarity Network Region name"] == process, "Gene/Allele"
        ]
        nodes_indices = [labels_dict[node] for node in nodes if node in labels_dict.keys()]

        pos3 = {idx: pos[node_index] for idx, node_index in enumerate(nodes_indices)}
        pos3 = np.vstack(list(pos3.values()))

        kernel = gaussian_kde(pos3.T)
        [X, Y] = np.mgrid[np.min(x) : np.max(x) : 100j, np.min(y) : np.max(y) : 100j]
        positions = np.vstack([X.ravel(), Y.ravel()])
        Z = np.reshape(kernel(positions).T, X.shape)

        C = ax.contour(X, Y, Z, [1e-6], colors=[tuple(process_colors[n_process, :])], alpha=1)

        if clabels:
            C.levels = [n_process + 1]
            plt.clabel(C, C.levels, inline=True, fmt="%d", fontsize=16)
            logging.info("%d -- %s" % (n_process + 1, process))


def mark_nodes(
    x,
    y,
    kind: list,
    ax=None,
    foreground_color="#ffffff",
    background_color="#000000",
    labels=None,  # subset the nodes by labels
    label_va="center",
    legend_label: str = None,
    test=False,
    **kws,
):
    """
    Show nodes.

    Parameters:
        s (str): legend name (defaults to '').
        kind (str): 'mark' if the nodes should be marked, 'label' if nodes should be marked and labeled.
    """
    if ax is None:
        ax = plt.gca()  # get current axes i.e. subplot
    if isinstance(kind, str):
        kind = [kind]

    if "mark" in kind:
        ## mark the selected nodes with the marker +
        sn1 = ax.scatter(x, y, **kws)

    if "label" in kind:
        ## show labels e.g. gene names
        if test:
            print(x, y, labels)
        assert len(x) == len(labels), f"len(x)!=len(labels): {len(x)}!={len(labels)}"
        if test:
            ax.plot(x, y, "r*")
        for i, label in enumerate(labels):
            ax.text(
                x[i],
                y[i],
                label,
                fontdict={
                    "color": "white" if background_color == "#000000" else "k",
                    "size": 14,
                    "weight": "bold",
                },
                # bbox={'facecolor': 'black', 'alpha': 0.5, 'pad': 3},
                ha="center",
                va=label_va,
            )

    if not legend_label is None:
        # Legend
        leg = ax.legend(
            [sn1],
            [legend_label],
            loc="upper left",
            bbox_to_anchor=(0, 1),
            title="Significance",
            scatterpoints=1,
            fancybox=False,
            facecolor=background_color,
            edgecolor=background_color,
        )

        for leg_txt in leg.get_texts():
            leg_txt.set_color(foreground_color)

        leg_title = leg.get_title()
        leg_title.set_color(foreground_color)

    return ax


def get_node_coordinates(graph, labels=[]):
    x = dict(graph.nodes.data("x"))
    y = dict(graph.nodes.data("y"))

    ds = [x, y]
    pos = {}
    for k in x:
        pos[k] = np.array([d[k] for d in ds])

    node_xy_list = list(pos.values())

    if len(labels) == 0:
        return np.vstack(node_xy_list)
    else:
        # Get the co-ordinates of the nodes
        node_labels = nx.get_node_attributes(graph, "label")
        node_labels_dict = {k: v for v, k in node_labels.items()}

        # TODOs: avoid determining the x and y again.
        x = list(dict(graph.nodes.data("x")).values())
        y = list(dict(graph.nodes.data("y")).values())

        # x_offset = (np.nanmax(x) - np.nanmin(x))*0.01

        idx = [node_labels_dict[x] for x in labels if x in node_labels_dict.keys()]

        # Labels found in the data
        labels_found = [x for x in labels if x in node_labels_dict.keys()]
        x_idx = [x[i] for i in idx]
        y_idx = [y[i] for i in idx]

        # Print out labels not found
        labels_missing = [x for x in labels if x not in node_labels_dict.keys()]
        if labels_missing:
            labels_missing_str = ", ".join(labels_missing)
            logging.warning(
                "These labels are missing from the network (case sensitive): %s"
                % labels_missing_str
            )

        node_xy_list = [x_idx, y_idx]

        return np.vstack(node_xy_list).T, labels_found


def chop_and_filter(s):
    single_str = s.str.cat(sep=" ")
    single_list = re.findall(r"[\w']+", single_str)

    single_list_count = dict(Counter(single_list))
    single_list_count = [
        k for k in sorted(single_list_count, key=single_list_count.get, reverse=True)
    ]

    to_exclude = ["of", "a", "the", "an", ",", "via", "to", "into", "from"]
    single_list_words = [w for w in single_list_count if w not in to_exclude]

    return ", ".join(single_list_words[:5])
