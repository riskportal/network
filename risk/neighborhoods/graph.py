"""
risk/neighborhoods/graph
~~~~~~~~~~~~~~~~~~~~~~~~
"""

import community as community_louvain
import networkx as nx
import numpy as np
import markov_clustering as mc
from networkx.algorithms.community import asyn_lpa_communities


def calculate_dijkstra_neighborhoods(network: nx.Graph) -> np.ndarray:
    """Calculate neighborhoods using Dijkstra's shortest path distances.

    Args:
        network (nx.Graph): The network graph.

    Returns:
        np.ndarray: Neighborhood matrix based on Dijkstra's distances.
    """
    # Compute Dijkstra's distance for all pairs of nodes in the network
    all_dijkstra_paths = dict(nx.all_pairs_dijkstra_path_length(network, weight="length"))
    neighborhoods = np.zeros((network.number_of_nodes(), network.number_of_nodes()), dtype=int)

    # Populate the neighborhoods matrix based on Dijkstra's distances
    for source, targets in all_dijkstra_paths.items():
        for target, length in targets.items():
            neighborhoods[source, target] = (
                1 if np.isnan(length) or length == 0 else np.sqrt(1 / length)
            )

    return neighborhoods


def calculate_label_propagation_neighborhoods(network: nx.Graph) -> np.ndarray:
    """Apply Label Propagation to the network to detect communities.

    Args:
        network (nx.Graph): The network graph.

    Returns:
        np.ndarray: Neighborhood matrix based on Label Propagation.
    """
    # Apply Label Propagation
    communities = nx.algorithms.community.label_propagation.label_propagation_communities(network)

    # Create a mapping from node to community
    community_dict = {}
    for community_id, community in enumerate(communities):
        for node in community:
            community_dict[node] = community_id

    # Create a neighborhood matrix
    num_nodes = network.number_of_nodes()
    neighborhoods = np.zeros((num_nodes, num_nodes), dtype=int)

    # Assign neighborhoods based on community labels
    for node_i, community_i in community_dict.items():
        for node_j, community_j in community_dict.items():
            if community_i == community_j:
                neighborhoods[node_i, node_j] = 1

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
        np.ndarray: Neighborhood matrix based on the Louvain method.
    """
    # Apply Louvain method to partition the network
    partition = community_louvain.best_partition(
        network, resolution=resolution, random_state=random_seed
    )
    neighborhoods = np.zeros((network.number_of_nodes(), network.number_of_nodes()), dtype=int)

    # Assign neighborhoods based on community partitions
    for node_i, community_i in partition.items():
        for node_j, community_j in partition.items():
            if community_i == community_j:
                neighborhoods[node_i, node_j] = 1

    return neighborhoods


def calculate_markov_clustering_neighborhoods(network: nx.Graph) -> np.ndarray:
    """Apply Markov Clustering (MCL) to the network.

    Args:
        network (nx.Graph): The network graph.

    Returns:
        np.ndarray: Neighborhood matrix based on Markov Clustering.
    """
    # Convert the graph to an adjacency matrix
    adjacency_matrix = nx.to_numpy_array(network)
    # Run Markov Clustering
    result = mc.run_mcl(adjacency_matrix)  # Run MCL with default parameters
    # Get clusters
    clusters = mc.get_clusters(result)

    # Create a community label for each node
    community_dict = {}
    for community_id, community in enumerate(clusters):
        for node in community:
            community_dict[node] = community_id

    # Create a neighborhood matrix
    num_nodes = network.number_of_nodes()
    neighborhoods = np.zeros((num_nodes, num_nodes), dtype=int)

    # Assign neighborhoods based on community labels
    for node_i, community_i in community_dict.items():
        for node_j, community_j in community_dict.items():
            if community_i == community_j:
                neighborhoods[node_i, node_j] = 1

    return neighborhoods


def calculate_spinglass_neighborhoods(network: nx.Graph) -> np.ndarray:
    """Apply Spin Glass Community Detection to the network.

    Args:
        network (nx.Graph): The network graph.

    Returns:
        np.ndarray: Neighborhood matrix based on Spin Glass communities.
    """
    # Use the asynchronous label propagation algorithm as a proxy for Spin Glass
    communities = asyn_lpa_communities(network)

    # Create a community label for each node
    community_dict = {}
    for community_id, community in enumerate(communities):
        for node in community:
            community_dict[node] = community_id

    # Create a neighborhood matrix
    num_nodes = network.number_of_nodes()
    neighborhoods = np.zeros((num_nodes, num_nodes), dtype=int)

    # Assign neighborhoods based on community labels
    for node_i, community_i in community_dict.items():
        for node_j, community_j in community_dict.items():
            if community_i == community_j:
                neighborhoods[node_i, node_j] = 1

    return neighborhoods


def calculate_walktrap_neighborhoods(network: nx.Graph) -> np.ndarray:
    """Apply Walktrap Community Detection to the network.

    Args:
        network (nx.Graph): The network graph.

    Returns:
        np.ndarray: Neighborhood matrix based on Walktrap communities.
    """
    # Use the asynchronous label propagation algorithm as a proxy for Walktrap
    communities = asyn_lpa_communities(network)

    # Create a community label for each node
    community_dict = {}
    for community_id, community in enumerate(communities):
        for node in community:
            community_dict[node] = community_id

    # Create a neighborhood matrix
    num_nodes = network.number_of_nodes()
    neighborhoods = np.zeros((num_nodes, num_nodes), dtype=int)

    # Assign neighborhoods based on community labels
    for node_i, community_i in community_dict.items():
        for node_j, community_j in community_dict.items():
            if community_i == community_j:
                neighborhoods[node_i, node_j] = 1

    return neighborhoods
