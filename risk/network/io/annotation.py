"""
risk/network/io/annotation
~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import json

import networkx as nx
import pandas as pd


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
    if annotation_pivot.notnull().sum().sum() == 0:
        raise ValueError(
            "No annotations found in the annotation file for the nodes in the network."
        )

    ordered_nodes = tuple(annotation_pivot.index)
    ordered_annotations = tuple(annotation_pivot.columns)
    return {
        "ordered_row_nodes": ordered_nodes,
        "ordered_column_annotations": ordered_annotations,
        "annotation_matrix": annotation_pivot.to_numpy(),
    }
