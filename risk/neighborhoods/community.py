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


def calculate_greedy_modularity_neighborhoods(
    network: nx.Graph, fraction_shortest_edges: float = 1.0
) -> np.ndarray:
    """Calculate neighborhoods using the Greedy Modularity method.

    Args:
        network (nx.Graph): The network graph.
        fraction_shortest_edges (float, optional): Shortest edge rank fraction threshold for creating
            subgraphs before clustering.

    Returns:
        np.ndarray: A binary neighborhood matrix where nodes in the same community have 1, and others have 0.
    """
    # Create a subgraph with the shortest edges based on the rank fraction
    subnetwork = _create_percentile_limited_subgraph(
        network, fraction_shortest_edges=fraction_shortest_edges
    )
    # Detect communities using the Greedy Modularity method
    communities = greedy_modularity_communities(subnetwork)
    # Get the list of nodes in the original NetworkX graph
    nodes = list(network.nodes())
    node_index_map = {node: idx for idx, node in enumerate(nodes)}
    # Create a binary neighborhood matrix
    num_nodes = len(nodes)
    # Initialize neighborhoods with zeros and set self-self entries to 1
    neighborhoods = np.eye(num_nodes, dtype=int)
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


def calculate_label_propagation_neighborhoods(
    network: nx.Graph, fraction_shortest_edges: float = 1.0
) -> np.ndarray:
    """Apply Label Propagation to the network to detect communities.

    Args:
        network (nx.Graph): The network graph.
        fraction_shortest_edges (float, optional): Shortest edge rank fraction threshold for creating
            subgraphs before clustering.

    Returns:
        np.ndarray: A binary neighborhood matrix on Label Propagation.
    """
    # Create a subgraph with the shortest edges based on the rank fraction
    subnetwork = _create_percentile_limited_subgraph(
        network, fraction_shortest_edges=fraction_shortest_edges
    )
    # Apply Label Propagation for community detection
    communities = nx.algorithms.community.label_propagation.label_propagation_communities(
        subnetwork
    )
    # Get the list of nodes in the network
    nodes = list(network.nodes())
    node_index_map = {node: idx for idx, node in enumerate(nodes)}
    # Create a binary neighborhood matrix
    num_nodes = len(nodes)
    # Initialize neighborhoods with zeros and set self-self entries to 1
    neighborhoods = np.eye(num_nodes, dtype=int)
    # Assign neighborhoods based on community labels using the mapped indices
    for community in communities:
        for node_i in community:
            for node_j in community:
                idx_i = node_index_map[node_i]
                idx_j = node_index_map[node_j]
                neighborhoods[idx_i, idx_j] = 1

    return neighborhoods


def calculate_leiden_neighborhoods(
    network: nx.Graph,
    resolution: float = 1.0,
    fraction_shortest_edges: float = 1.0,
    random_seed: int = 888,
) -> np.ndarray:
    """Calculate neighborhoods using the Leiden method.

    Args:
        network (nx.Graph): The network graph.
        resolution (float, optional): Resolution parameter for the Leiden method. Defaults to 1.0.
        fraction_shortest_edges (float, optional): Shortest edge rank fraction threshold for creating
            subgraphs before clustering.
        random_seed (int, optional): Random seed for reproducibility. Defaults to 888.

    Returns:
        np.ndarray: A binary neighborhood matrix where nodes in the same community have 1, and others have 0.
    """
    # Create a subgraph with the shortest edges based on the rank fraction
    subnetwork = _create_percentile_limited_subgraph(
        network, fraction_shortest_edges=fraction_shortest_edges
    )
    # Convert NetworkX graph to iGraph
    igraph_network = ig.Graph.from_networkx(subnetwork)
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
    # Initialize neighborhoods with zeros and set self-self entries to 1
    neighborhoods = np.eye(num_nodes, dtype=int)
    # Assign neighborhoods based on community partitions using the mapped indices
    for community in partition:
        for node_i in community:
            for node_j in community:
                idx_i = node_index_map[igraph_network.vs[node_i]["_nx_name"]]
                idx_j = node_index_map[igraph_network.vs[node_j]["_nx_name"]]
                neighborhoods[idx_i, idx_j] = 1

    return neighborhoods


def calculate_louvain_neighborhoods(
    network: nx.Graph,
    resolution: float = 0.1,
    fraction_shortest_edges: float = 1.0,
    random_seed: int = 888,
) -> np.ndarray:
    """Calculate neighborhoods using the Louvain method.

    Args:
        network (nx.Graph): The network graph.
        resolution (float, optional): Resolution parameter for the Louvain method. Defaults to 0.1.
        fraction_shortest_edges (float, optional): Shortest edge rank fraction threshold for creating
            subgraphs before clustering.
        random_seed (int, optional): Random seed for reproducibility. Defaults to 888.

    Returns:
        np.ndarray: A binary neighborhood matrix on the Louvain method.
    """
    # Create a subgraph with the shortest edges based on the rank fraction
    subnetwork = _create_percentile_limited_subgraph(
        network, fraction_shortest_edges=fraction_shortest_edges
    )
    # Apply Louvain method to partition the network
    partition = community_louvain.best_partition(
        subnetwork, resolution=resolution, random_state=random_seed
    )
    # Get the list of nodes in the network and create a mapping to indices
    nodes = list(network.nodes())
    node_index_map = {node: idx for idx, node in enumerate(nodes)}
    # Create a binary neighborhood matrix
    num_nodes = len(nodes)
    # Initialize neighborhoods with zeros and set self-self entries to 1
    neighborhoods = np.eye(num_nodes, dtype=int)
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


def calculate_markov_clustering_neighborhoods(
    network: nx.Graph, fraction_shortest_edges: float = 1.0
) -> np.ndarray:
    """Apply Markov Clustering (MCL) to the network and return a binary neighborhood matrix.

    Args:
        network (nx.Graph): The network graph.
        fraction_shortest_edges (float, optional): Shortest edge rank fraction threshold for creating
            subgraphs before clustering.

    Returns:
        np.ndarray: A binary neighborhood matrix on Markov Clustering.
    """
    # Create a subgraph with the shortest edges based on the rank fraction
    subnetwork = _create_percentile_limited_subgraph(
        network, fraction_shortest_edges=fraction_shortest_edges
    )
    # Step 1: Convert the subnetwork to an adjacency matrix
    subnetwork_nodes = list(subnetwork.nodes())
    adjacency_matrix = nx.to_numpy_array(subnetwork, nodelist=subnetwork_nodes)
    # Step 2: Run Markov Clustering (MCL) on the subnetwork's adjacency matrix
    result = mc.run_mcl(adjacency_matrix)
    clusters = mc.get_clusters(result)
    # Step 3: Prepare the original network nodes and indices
    nodes = list(network.nodes())
    node_index_map = {node: idx for idx, node in enumerate(nodes)}
    num_nodes = len(nodes)
    # Step 4: Initialize the neighborhood matrix for the original network
    neighborhoods = np.eye(num_nodes, dtype=int)
    # Step 5: Fill the neighborhoods matrix using the clusters from the subnetwork
    for cluster in clusters:
        for node_i in cluster:
            for node_j in cluster:
                # Map the indices back to the original network's node indices
                original_node_i = subnetwork_nodes[node_i]
                original_node_j = subnetwork_nodes[node_j]

                if original_node_i in node_index_map and original_node_j in node_index_map:
                    idx_i = node_index_map[original_node_i]
                    idx_j = node_index_map[original_node_j]
                    neighborhoods[idx_i, idx_j] = 1

    return neighborhoods


def calculate_spinglass_neighborhoods(
    network: nx.Graph, fraction_shortest_edges: float = 1.0
) -> np.ndarray:
    """Apply Spinglass Community Detection to the network, handling disconnected components.

    Args:
        network (nx.Graph): The network graph.
        fraction_shortest_edges (float, optional): Shortest edge rank fraction threshold for creating
            subgraphs before clustering.

    Returns:
        np.ndarray: A binary neighborhood matrix based on Spinglass communities.
    """
    # Create a subgraph with the shortest edges based on the rank fraction
    subnetwork = _create_percentile_limited_subgraph(
        network, fraction_shortest_edges=fraction_shortest_edges
    )
    # Step 1: Find connected components in the graph
    components = list(nx.connected_components(subnetwork))
    # Prepare to store community results
    nodes = list(network.nodes())
    node_index_map = {node: idx for idx, node in enumerate(nodes)}
    num_nodes = len(nodes)
    # Initialize neighborhoods with zeros and set self-self entries to 1
    neighborhoods = np.eye(num_nodes, dtype=int)
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


def calculate_walktrap_neighborhoods(
    network: nx.Graph, fraction_shortest_edges: float = 1.0
) -> np.ndarray:
    """Apply Walktrap Community Detection to the network.

    Args:
        network (nx.Graph): The network graph.
        fraction_shortest_edges (float, optional): Shortest edge rank fraction threshold for creating
            subgraphs before clustering.

    Returns:
        np.ndarray: A binary neighborhood matrix on Walktrap communities.
    """
    # Create a subgraph with the shortest edges based on the rank fraction
    subnetwork = _create_percentile_limited_subgraph(
        network, fraction_shortest_edges=fraction_shortest_edges
    )
    # Convert NetworkX graph to iGraph
    igraph_network = ig.Graph.from_networkx(subnetwork)
    # Apply Walktrap community detection
    communities = igraph_network.community_walktrap().as_clustering()
    # Get the list of nodes in the original NetworkX graph
    nodes = list(network.nodes())
    node_index_map = {node: idx for idx, node in enumerate(nodes)}
    # Create a binary neighborhood matrix
    num_nodes = len(nodes)
    # Initialize neighborhoods with zeros and set self-self entries to 1
    neighborhoods = np.eye(num_nodes, dtype=int)
    # Assign neighborhoods based on community labels
    for community in communities:
        for node_i in community:
            for node_j in community:
                idx_i = node_index_map[igraph_network.vs[node_i]["_nx_name"]]
                idx_j = node_index_map[igraph_network.vs[node_j]["_nx_name"]]
                neighborhoods[idx_i, idx_j] = 1

    return neighborhoods


def _create_percentile_limited_subgraph(G: nx.Graph, fraction_shortest_edges: float) -> nx.Graph:
    """Create a subgraph containing the shortest edges based on the specified rank fraction
    of all edge lengths in the input graph.

    Args:
        G (nx.Graph): The input graph with 'length' attributes on edges.
        fraction_shortest_edges (float): The rank fraction (between 0 and 1) to filter edges.

    Returns:
        nx.Graph: A subgraph with nodes and edges where the edges are within the shortest
        specified rank fraction.
    """
    # Step 1: Extract edges with their lengths
    edges_with_length = [(u, v, d) for u, v, d in G.edges(data=True) if "length" in d]
    if not edges_with_length:
        raise ValueError(
            "No edge lengths found in the graph. Ensure edges have 'length' attributes."
        )

    # Step 2: Sort edges by length in ascending order
    edges_with_length.sort(key=lambda x: x[2]["length"])
    # Step 3: Calculate the cutoff index for the given rank fraction
    cutoff_index = int(fraction_shortest_edges * len(edges_with_length))
    if cutoff_index == 0:
        raise ValueError("The rank fraction is too low, resulting in no edges being included.")

    # Step 4: Create the subgraph by selecting only the shortest edges within the rank fraction
    subgraph = nx.Graph()
    subgraph.add_nodes_from(G.nodes(data=True))  # Retain all nodes from the original graph
    subgraph.add_edges_from(edges_with_length[:cutoff_index])
    # Step 5: Remove nodes with no edges
    subgraph.remove_nodes_from(list(nx.isolates(subgraph)))
    # Step 6: Check if the resulting subgraph has no edges and issue a warning
    if subgraph.number_of_edges() == 0:
        raise Warning("The resulting subgraph has no edges. Consider adjusting the rank fraction.")

    return subgraph
