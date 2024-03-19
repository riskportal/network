"""
risk/network/io/tidy
~~~~~~~~~~~~~~~~~~~~
"""

import warnings

import networkx as nx


def remove_invalid_graph_properties(G, min_edges_per_node=0):
    # Remove nodes with `min_edges_per_node` or fewer edges
    nodes_with_few_edges = [node for node in G.nodes() if G.degree(node) <= min_edges_per_node]
    G.remove_nodes_from(nodes_with_few_edges)
    self_loops = list(nx.selfloop_edges(G))
    G.remove_edges_from(self_loops)


def validate_graph(graph):
    # Check if every node has 'x', 'y', and 'label' attributes
    for node, attrs in graph.nodes(data=True):
        assert (
            "x" in attrs and "y" in attrs
        ), f"Node {node} is missing 'x' or 'y' position attributes."
        assert "label" in attrs, f"Node {node} is missing a 'label' attribute."

    # Check if edges have weights, warn if any are missing
    missing_weights = [edge for edge in graph.edges(data=True) if "weight" not in edge[2]]
    if missing_weights:
        warnings.warn(
            "Some edges are missing weights; default weight of 1 will be used for missing weights."
        )
