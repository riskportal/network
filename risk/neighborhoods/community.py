"""
risk/neighborhoods/community
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import community as community_louvain
import networkx as nx
import numpy as np
import markov_clustering as mc
from networkx.algorithms.community import asyn_lpa_communities, greedy_modularity_communities


def calculate_greedy_modularity_neighborhoods(network: nx.Graph) -> np.ndarray:
    """Calculate neighborhoods using the Greedy Modularity method.

    Args:
        network (nx.Graph): The network graph to analyze for community structure.

    Returns:
        np.ndarray: A binary neighborhood matrix where nodes in the same community have 1, and others have 0.
    """
    # Detect communities using the Greedy Modularity method
    communities = greedy_modularity_communities(network)
    # Create a binary neighborhood matrix
    n_nodes = network.number_of_nodes()
    neighborhoods = np.zeros((n_nodes, n_nodes), dtype=int)
    # Create a mapping from node to index in the matrix
    node_index = {node: i for i, node in enumerate(network.nodes())}
    # Fill in the neighborhood matrix for nodes in the same community
    for community in communities:
        # Iterate through all pairs of nodes in the same community
        for node_i in community:
            idx_i = node_index[node_i]
            for node_j in community:
                idx_j = node_index[node_j]
                # Set them as neighbors (1) in the binary matrix
                neighborhoods[idx_i, idx_j] = 1

    return neighborhoods


def calculate_label_propagation_neighborhoods(network: nx.Graph) -> np.ndarray:
    """Apply Label Propagation to the network to detect communities.

    Args:
        network (nx.Graph): The network graph.

    Returns:
        np.ndarray: Binary neighborhood matrix on Label Propagation.
    """
    # Apply Label Propagation for community detection
    communities = nx.algorithms.community.label_propagation.label_propagation_communities(network)
    # Create a binary neighborhood matrix
    num_nodes = network.number_of_nodes()
    neighborhoods = np.zeros((num_nodes, num_nodes), dtype=int)
    # Create a mapping from node to index in the matrix
    node_index = {node: i for i, node in enumerate(network.nodes())}
    # Assign neighborhoods based on community labels
    for community in communities:
        for node_i in community:
            idx_i = node_index[node_i]
            for node_j in community:
                idx_j = node_index[node_j]
                neighborhoods[idx_i, idx_j] = 1

    return neighborhoods


def calculate_louvain_neighborhoods(
    network: nx.Graph, resolution: float, random_seed: int = 888
) -> np.ndarray:
    """Calculate neighborhoods using the Louvain method.

    Args:
        network (nx.Graph): The network graph.
        resolution (float): Resolution parameter for the Louvain method.
        random_seed (int, optional): Random seed for reproducibility. Defaults to 888.

    Returns:
        np.ndarray: Binary neighborhood matrix on the Louvain method.
    """
    # Apply Louvain method to partition the network
    partition = community_louvain.best_partition(
        network, resolution=resolution, random_state=random_seed
    )
    # Create a binary neighborhood matrix
    num_nodes = network.number_of_nodes()
    neighborhoods = np.zeros((num_nodes, num_nodes), dtype=int)
    # Create a mapping from node to index in the matrix
    node_index = {node: i for i, node in enumerate(network.nodes())}
    # Group nodes by community
    community_groups = {}
    for node, community in partition.items():
        community_groups.setdefault(community, []).append(node)

    # Assign neighborhoods based on community partitions
    for community, nodes in community_groups.items():
        for node_i in nodes:
            idx_i = node_index[node_i]
            for node_j in nodes:
                idx_j = node_index[node_j]
                neighborhoods[idx_i, idx_j] = 1

    return neighborhoods


def calculate_markov_clustering_neighborhoods(network: nx.Graph) -> np.ndarray:
    """Apply Markov Clustering (MCL) to the network.

    Args:
        network (nx.Graph): The network graph.

    Returns:
        np.ndarray: Binary neighborhood matrix on Markov Clustering.
    """
    # Convert the graph to an adjacency matrix
    adjacency_matrix = nx.to_numpy_array(network)
    # Run Markov Clustering (MCL)
    result = mc.run_mcl(adjacency_matrix)  # MCL with default parameters
    # Get clusters (communities) from MCL result
    clusters = mc.get_clusters(result)
    # Create a binary neighborhood matrix
    num_nodes = network.number_of_nodes()
    neighborhoods = np.zeros((num_nodes, num_nodes), dtype=int)
    # Create a mapping from node to index in the matrix
    node_index = {node: i for i, node in enumerate(network.nodes())}
    # Assign neighborhoods based on MCL clusters
    for cluster in clusters:
        for node_i in cluster:
            idx_i = node_index[node_i]
            for node_j in cluster:
                idx_j = node_index[node_j]
                neighborhoods[idx_i, idx_j] = 1

    return neighborhoods


def calculate_spinglass_neighborhoods(network: nx.Graph) -> np.ndarray:
    """Apply Spin Glass Community Detection to the network.

    Args:
        network (nx.Graph): The network graph.

    Returns:
        np.ndarray: Binary neighborhood matrix on Spin Glass communities.
    """
    # Apply Asynchronous Label Propagation (LPA)
    communities = asyn_lpa_communities(network)
    # Create a binary neighborhood matrix
    num_nodes = network.number_of_nodes()
    neighborhoods = np.zeros((num_nodes, num_nodes), dtype=int)
    # Create a mapping from node to index in the matrix
    node_index = {node: i for i, node in enumerate(network.nodes())}
    # Assign neighborhoods based on community labels from LPA
    for community in communities:
        for node_i in community:
            idx_i = node_index[node_i]
            for node_j in community:
                idx_j = node_index[node_j]
                neighborhoods[idx_i, idx_j] = 1

    return neighborhoods


def calculate_walktrap_neighborhoods(network: nx.Graph) -> np.ndarray:
    """Apply Walktrap Community Detection to the network.

    Args:
        network (nx.Graph): The network graph.

    Returns:
        np.ndarray: Binary neighborhood matrix on Walktrap communities.
    """
    # Apply Asynchronous Label Propagation (LPA)
    communities = asyn_lpa_communities(network)
    # Create a binary neighborhood matrix
    num_nodes = network.number_of_nodes()
    neighborhoods = np.zeros((num_nodes, num_nodes), dtype=int)
    # Create a mapping from node to index in the matrix
    node_index = {node: i for i, node in enumerate(network.nodes())}
    # Assign neighborhoods based on community labels from LPA
    for community in communities:
        for node_i in community:
            idx_i = node_index[node_i]
            for node_j in community:
                idx_j = node_index[node_j]
                neighborhoods[idx_i, idx_j] = 1

    return neighborhoods
