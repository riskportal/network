"""
risk/network/io/tidy
~~~~~~~~~~~~~~~~~~~~
"""

import networkx as nx


def remove_invalid_graph_properties(G, min_edges_per_node=0):
    # Remove nodes with `min_edges_per_node` or fewer edges
    nodes_with_few_edges = [node for node in G.nodes() if G.degree(node) <= min_edges_per_node]
    G.remove_nodes_from(nodes_with_few_edges)
    self_loops = list(nx.selfloop_edges(G))
    G.remove_edges_from(self_loops)
