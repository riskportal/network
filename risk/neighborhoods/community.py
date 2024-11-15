"""
risk/neighborhoods/community
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import community as community_louvain
import igraph as ig
import markov_clustering as mc
import networkx as nx
import numpy as np
from leidenalg import find_partition, RBConfigurationVertexPartition
from networkx.algorithms.community import greedy_modularity_communities


def calculate_greedy_modularity_neighborhoods(network: nx.Graph) -> np.ndarray:
    """Calculate neighborhoods using the Greedy Modularity method.

    Args:
        network (nx.Graph): The network graph to analyze for community structure.

    Returns:
        np.ndarray: A binary neighborhood matrix where nodes in the same community have 1, and others have 0.
    """
    # Detect communities using the Greedy Modularity method
    communities = greedy_modularity_communities(network)
    # Get the list of nodes in the original NetworkX graph
    nodes = list(network.nodes())
    node_index_map = {node: idx for idx, node in enumerate(nodes)}
    # Create a binary neighborhood matrix
    n_nodes = len(nodes)
    neighborhoods = np.zeros((n_nodes, n_nodes), dtype=int)
    # Fill in the neighborhood matrix for nodes in the same community
    for community in communities:
        # Iterate through all pairs of nodes in the same community
        for node_i in community:
            for node_j in community:
                idx_i = node_index_map[node_i]
                idx_j = node_index_map[node_j]
                # Set them as neighbors (1) in the binary matrix
                neighborhoods[idx_i, idx_j] = 1

    return neighborhoods


def calculate_label_propagation_neighborhoods(network: nx.Graph) -> np.ndarray:
    """Apply Label Propagation to the network to detect communities.

    Args:
        network (nx.Graph): The network graph.

    Returns:
        np.ndarray: A binary neighborhood matrix on Label Propagation.
    """
    # Apply Label Propagation for community detection
    communities = nx.algorithms.community.label_propagation.label_propagation_communities(network)
    # Get the list of nodes in the network
    nodes = list(network.nodes())
    node_index_map = {node: idx for idx, node in enumerate(nodes)}
    # Create a binary neighborhood matrix
    num_nodes = len(nodes)
    neighborhoods = np.zeros((num_nodes, num_nodes), dtype=int)
    # Assign neighborhoods based on community labels using the mapped indices
    for community in communities:
        for node_i in community:
            for node_j in community:
                idx_i = node_index_map[node_i]
                idx_j = node_index_map[node_j]
                neighborhoods[idx_i, idx_j] = 1

    return neighborhoods


def calculate_leiden_neighborhoods(
    network: nx.Graph, resolution: float = 1.0, random_seed: int = 888
) -> np.ndarray:
    """Calculate neighborhoods using the Leiden method.

    Args:
        network (nx.Graph): The network graph.
        resolution (float, optional): Resolution parameter for the Leiden method. Defaults to 1.0.
        random_seed (int, optional): Random seed for reproducibility. Defaults to 888.

    Returns:
        np.ndarray: A binary neighborhood matrix where nodes in the same community have 1, and others have 0.
    """
    # Convert NetworkX graph to iGraph
    igraph_network = ig.Graph.from_networkx(network)
    # Apply Leiden algorithm using RBConfigurationVertexPartition, which supports resolution
    partition = find_partition(
        igraph_network,
        partition_type=RBConfigurationVertexPartition,
        resolution_parameter=resolution,
        seed=random_seed,
    )
    # Get the list of nodes in the original NetworkX graph
    nodes = list(network.nodes())
    node_index_map = {node: idx for idx, node in enumerate(nodes)}
    # Create a binary neighborhood matrix
    num_nodes = len(nodes)
    neighborhoods = np.zeros((num_nodes, num_nodes), dtype=int)
    # Assign neighborhoods based on community partitions using the mapped indices
    for community in partition:
        for node_i in community:
            for node_j in community:
                idx_i = node_index_map[igraph_network.vs[node_i]["_nx_name"]]
                idx_j = node_index_map[igraph_network.vs[node_j]["_nx_name"]]
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
        np.ndarray: A binary neighborhood matrix on the Louvain method.
    """
    # Apply Louvain method to partition the network
    partition = community_louvain.best_partition(
        network, resolution=resolution, random_state=random_seed
    )
    # Get the list of nodes in the network and create a mapping to indices
    nodes = list(network.nodes())
    node_index_map = {node: idx for idx, node in enumerate(nodes)}
    # Create a binary neighborhood matrix
    num_nodes = len(nodes)
    neighborhoods = np.zeros((num_nodes, num_nodes), dtype=int)
    # Group nodes by community
    community_groups = {}
    for node, community in partition.items():
        community_groups.setdefault(community, []).append(node)

    # Assign neighborhoods based on community partitions using the mapped indices
    for community, nodes in community_groups.items():
        for node_i in nodes:
            for node_j in nodes:
                idx_i = node_index_map[node_i]
                idx_j = node_index_map[node_j]
                neighborhoods[idx_i, idx_j] = 1

    return neighborhoods


def calculate_markov_clustering_neighborhoods(network: nx.Graph) -> np.ndarray:
    """Apply Markov Clustering (MCL) to the network.

    Args:
        network (nx.Graph): The network graph.

    Returns:
        np.ndarray: A binary neighborhood matrix on Markov Clustering.
    """
    # Step 1: Convert the graph to an adjacency matrix
    nodes = list(network.nodes())
    node_index_map = {node: idx for idx, node in enumerate(nodes)}
    # Step 2: Create a reverse mapping from index to node
    index_node_map = {idx: node for node, idx in node_index_map.items()}
    adjacency_matrix = nx.to_numpy_array(network, nodelist=nodes)
    # Step 3: Run Markov Clustering (MCL) on the adjacency matrix
    result = mc.run_mcl(adjacency_matrix)
    # Step 4: Get clusters (communities) from MCL result
    clusters = mc.get_clusters(result)
    # Step 5: Create a binary neighborhood matrix
    num_nodes = len(nodes)
    neighborhoods = np.zeros((num_nodes, num_nodes), dtype=int)
    # Step 6: Assign neighborhoods based on MCL clusters using the original node labels
    for cluster in clusters:
        for node_i in cluster:
            for node_j in cluster:
                # Map the matrix indices back to the original node labels
                original_node_i = index_node_map[node_i]
                original_node_j = index_node_map[node_j]
                idx_i = node_index_map[original_node_i]
                idx_j = node_index_map[original_node_j]
                neighborhoods[idx_i, idx_j] = 1

    return neighborhoods


def calculate_spinglass_neighborhoods(network: nx.Graph) -> np.ndarray:
    """Apply Spinglass Community Detection to the network, handling disconnected components.

    Args:
        network (nx.Graph): The input network graph with 'x' and 'y' attributes for node positions.

    Returns:
        np.ndarray: A binary neighborhood matrix based on Spinglass communities.
    """
    # Step 1: Find connected components in the graph
    components = list(nx.connected_components(network))
    # Prepare to store community results
    nodes = list(network.nodes())
    node_index_map = {node: idx for idx, node in enumerate(nodes)}
    num_nodes = len(nodes)
    neighborhoods = np.zeros((num_nodes, num_nodes), dtype=int)
    # Step 2: Run Spinglass on each connected component
    for component in components:
        # Extract the subgraph corresponding to the current component
        subgraph = network.subgraph(component)
        # Convert the subgraph to an iGraph object
        igraph_subgraph = ig.Graph.from_networkx(subgraph)
        # Ensure the subgraph is connected before running Spinglass
        if not igraph_subgraph.is_connected():
            print("Warning: Subgraph is not connected. Skipping...")
            continue

        # Apply Spinglass community detection
        try:
            communities = igraph_subgraph.community_spinglass()
        except Exception as e:
            print(f"Error running Spinglass on component: {e}")
            continue

        # Step 3: Assign neighborhoods based on community labels
        for community in communities:
            for node_i in community:
                for node_j in community:
                    idx_i = node_index_map[igraph_subgraph.vs[node_i]["_nx_name"]]
                    idx_j = node_index_map[igraph_subgraph.vs[node_j]["_nx_name"]]
                    neighborhoods[idx_i, idx_j] = 1

    return neighborhoods


def calculate_walktrap_neighborhoods(network: nx.Graph) -> np.ndarray:
    """Apply Walktrap Community Detection to the network.

    Args:
        network (nx.Graph): The network graph.

    Returns:
        np.ndarray: A binary neighborhood matrix on Walktrap communities.
    """
    # Convert NetworkX graph to iGraph
    igraph_network = ig.Graph.from_networkx(network)
    # Apply Walktrap community detection
    communities = igraph_network.community_walktrap().as_clustering()
    # Get the list of nodes in the original NetworkX graph
    nodes = list(network.nodes())
    node_index_map = {node: idx for idx, node in enumerate(nodes)}
    # Create a binary neighborhood matrix
    num_nodes = len(nodes)
    neighborhoods = np.zeros((num_nodes, num_nodes), dtype=int)
    # Assign neighborhoods based on community labels
    for community in communities:
        for node_i in community:
            for node_j in community:
                idx_i = node_index_map[igraph_network.vs[node_i]["_nx_name"]]
                idx_j = node_index_map[igraph_network.vs[node_j]["_nx_name"]]
                neighborhoods[idx_i, idx_j] = 1

    return neighborhoods
